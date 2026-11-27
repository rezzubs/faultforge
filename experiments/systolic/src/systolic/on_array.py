"""Running the matrix multiplications of a model on a systolic array."""

lazy from typing import Any, final, override

lazy import torch
lazy from torch import Tensor

# `TorchDispatchMode` is a private API, which is why torch is pinned to one
# minor version.
lazy from torch.utils._python_dispatch import TorchDispatchMode

lazy from systolic._rust import Fault
lazy from systolic.array import ArrayMatmul
lazy from systolic.evaluator import Evaluator
lazy from systolic.im2col import conv2d_as_matmul

# ATen is torch's library of basic operations. Every torch function runs as
# one or more of them, and these are the operations a dispatch mode sees.
#
# An operation can have several signatures, called overloads. For example
# `_aten.mm` groups `_aten.mm.default`, which returns a new tensor, and
# `_aten.mm.out`, which writes into a tensor given as `out=`. `default` is the
# name of the plain signature, the one a call like `torch.mm(a, b)` uses.
_aten = torch.ops.aten

# Ops that contain a matrix multiplication. These are groups of overloads, so
# that any overload that `OnArray` doesn't handle raises. A multiplication
# can't silently skip the array.
_MATMUL_OPS = frozenset(
    [
        _aten._addmm_activation,
        _aten._convolution,
        _aten._convolution_mode,
        _aten._int_mm,
        _aten._scaled_dot_product_attention_math,
        _aten._scaled_dot_product_cudnn_attention,
        _aten._scaled_dot_product_efficient_attention,
        _aten._scaled_dot_product_flash_attention,
        _aten._scaled_dot_product_flash_attention_for_cpu,
        _aten._scaled_mm,
        _aten._trilinear,
        _aten.addbmm,
        _aten.addmm,
        _aten.addmv,
        _aten.addr,
        _aten.baddbmm,
        _aten.bilinear,
        _aten.bmm,
        _aten.chain_matmul,
        _aten.conv1d,
        _aten.conv2d,
        _aten.conv3d,
        _aten.conv_tbc,
        _aten.conv_transpose1d,
        _aten.conv_transpose2d,
        _aten.conv_transpose3d,
        _aten.convolution,
        _aten.cudnn_convolution,
        _aten.cudnn_convolution_transpose,
        _aten.dot,
        _aten.einsum,
        _aten.inner,
        _aten.linalg_multi_dot,
        _aten.linear,
        _aten.matmul,
        _aten.miopen_convolution,
        _aten.mkldnn_convolution,
        _aten.mkldnn_linear,
        _aten.mm,
        _aten.mv,
        _aten.outer,
        _aten.scaled_dot_product_attention,
        _aten.tensordot,
        _aten.vdot,
    ]
)


@final
class OnArray(TorchDispatchMode):
    """Runs the matrix multiplications inside it on an array.

    Use it around a model's forward pass:

    ```python
    with torch.no_grad(), OnArray(evaluator, fault):
        output = model(inputs)
    ```

    `fault` is present in the array for every multiplication, `None` means a
    fault-free array.

    - The right operand of a multiplication is stationary in the array.
    - A convolution runs as one multiplication with its kernel stationary.
    - A bias is added after the array, so the fault doesn't affect it.
    - Operations without a matrix multiplication run as usual.

    Supported operations:

    - `mm`, `addmm`, `bmm` and `baddbmm`. The multiplications in a batch of
      `bmm` or `baddbmm` run one after another, all with the same fault.
    - `convolution` in 2D, if it isn't transposed and has groups of one.

    Raises:
        TypeError: If the kind of `fault` isn't supported.
        ValueError: If `fault` is outside the array.

    Inside the mode, a matrix multiplication raises:

    - `NotImplementedError` if the operation isn't supported.
    - `ValueError` if its operands aren't float32.
    - `RuntimeError` if gradients are enabled or float32 matrix
      multiplications may run at a reduced precision.
    """

    def __init__(self, evaluator: Evaluator, fault: Fault | None) -> None:
        super().__init__()
        if fault is not None:
            evaluator.array.check_fault(fault)
        self._evaluator = evaluator
        self._fault = fault

    @override
    def __torch_dispatch__(
        self,
        func: torch._ops.OpOverload,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        # Torch calls this for every ATen operation inside the mode, instead of
        # running it. The return value becomes the result of the operation.
        #
        # - `func` is the overload being called, such as `_aten.mm.default`.
        # - `types` holds the tensor subclasses among the arguments, which
        #   `OnArray` doesn't use.
        # - `args` and `kwargs` are the arguments in the order of the
        #   overload's signature. Keyword-only parameters, like `beta` and
        #   `alpha` of `addmm`, are in `kwargs`.
        kwargs = kwargs or {}

        # While this method runs, torch takes this mode off its stack. So the
        # torch ops it calls, including the evaluator's own `mm`, run as usual
        # instead of coming back here.
        #
        # Only the plain signatures are handled. Their arguments are, for
        # example, `mm(self, mat2)` and
        # `addmm(self, mat1, mat2, *, beta=1, alpha=1)`, where `self` is the
        # first operand, or the bias for `addmm`.
        if func in (_aten.mm.default, _aten.bmm.default):
            return self._matmul(*args, **kwargs)
        if func in (_aten.addmm.default, _aten.baddbmm.default):
            return self._added_to_bias(func, *args, **kwargs)
        if func is _aten.convolution.default:
            return self._convolution(*args, **kwargs)

        # `overloadpacket` is the group that `func` belongs to, such as
        # `_aten.mm` for `_aten.mm.out`. Any other overload of a matmul op
        # gets here.
        if func.overloadpacket in _MATMUL_OPS:
            message = f"{func} is not supported on the array"
            # Torch decomposes `linear` and `matmul` into the supported ops
            # only outside inference mode.
            if torch.is_inference_mode_enabled():
                message += "; use torch.no_grad() instead of torch.inference_mode()"
            raise NotImplementedError(message)

        # Everything else, such as additions and softmax, runs as it would
        # without the mode.
        return func(*args, **kwargs)

    def _added_to_bias(
        self,
        func: torch._ops.OpOverload,
        bias: Tensor,
        left: Tensor,
        right: Tensor,
        *,
        beta: Any = 1,
        alpha: Any = 1,
    ) -> Tensor:
        """`bias + left @ right`, for `addmm` and `baddbmm`."""
        if beta != 1 or alpha != 1:
            raise NotImplementedError(
                f"{func} with beta={beta} and alpha={alpha} is not supported on "
                "the array, only beta=1 and alpha=1"
            )
        return bias + self._matmul(left, right)

    def _convolution(
        self,
        inputs: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        transposed: bool,
        output_padding: list[int],
        groups: int,
    ) -> Tensor:
        # Each of these would need its own way of becoming a matmul.
        if transposed:
            raise NotImplementedError(
                "transposed convolutions are not supported on the array"
            )
        if groups != 1:
            raise NotImplementedError(
                f"convolutions with groups={groups} are not supported on the "
                "array, only groups=1"
            )
        if inputs.dim() != 4:
            raise NotImplementedError(
                "only 2D convolutions are supported on the array, got an input "
                f"of shape {tuple(inputs.shape)}"
            )

        output = conv2d_as_matmul(
            inputs,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            matmul=self._evaluate,
        )
        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)
        # See `_on_array`.
        return output.contiguous()

    def _matmul(self, left: Tensor, right: Tensor) -> Tensor:
        """`left @ right` on the array."""
        # Each row of `left` is one input, such as a token, and every input
        # is multiplied by the same `right`, like the weights of a linear
        # layer. So `right` is the operand held in the array.
        return self._on_array(streamed=left, stationary=right)

    def _on_array(self, *, streamed: Tensor, stationary: Tensor) -> Tensor:
        """`streamed @ stationary` on the array, with `stationary` held in it.

        Both operands have the same number of dimensions, 2 or 3.
        """
        # Torch multiplies row vectors: each row of `streamed` is one input,
        # and `stationary` maps it to a row of the output. The array
        # multiplies column vectors, with one input per column of the
        # activations. Transposing converts between the two, since
        # `(streamed @ stationary)^T = stationary^T @ streamed^T`.
        output = self._evaluate(weights=stationary.mT, activations=streamed.mT).mT
        # The replaced ops return contiguous tensors, which callers rely on,
        # for example by calling `view` on them.
        # It's free for an evaluator that computes in torch's layout, see
        # `matmul_in_torch_layout`.
        return output.contiguous()

    def _evaluate(self, weights: Tensor, activations: Tensor) -> Tensor:
        """`weights @ activations` on the array, with `weights` stationary."""
        _check_inference_only()
        _check_full_precision()

        matmul = ArrayMatmul(weights=weights, activations=activations)
        return self._evaluator.evaluate(matmul, self._fault)


def _check_inference_only() -> None:
    """Raise if gradients are enabled.

    A backward pass inside `OnArray` would put its matrix multiplications on
    the array too, and gradients through a faulty array don't mean anything.
    """
    if torch.is_grad_enabled():
        raise RuntimeError(
            "OnArray only supports inference, run the model under torch.no_grad()"
        )


def _check_full_precision() -> None:
    """Raise if float32 matrix multiplications may run at a reduced precision.

    Such as TF32 on CUDA or bf16 on the CPU. The array multiplies in full
    float32, so the multiplications of the evaluators must too.
    """
    # Torch adds some of these attributes at runtime, so ty can't see them.
    global_precision = torch.backends.fp32_precision  # ty: ignore[unresolved-attribute]
    for backend, precision in (
        ("CUDA", torch.backends.cuda.matmul.fp32_precision),
        ("oneDNN", torch.backends.mkldnn.matmul.fp32_precision),  # ty: ignore[unresolved-attribute]
    ):
        # A backend set to "none" uses the global setting, and a global
        # "none" means full precision.
        effective_precision = global_precision if precision == "none" else precision
        if effective_precision not in ("none", "ieee"):
            raise RuntimeError(
                f"float32 matrix multiplications on {backend} run at "
                f"{effective_precision} precision, but OnArray needs full "
                'precision ("ieee")'
            )
