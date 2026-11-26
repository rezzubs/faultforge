"""Layers whose matmul is routed through a SystolicBackend."""

lazy from typing import final, override

lazy from torch import Tensor, nn

lazy from faultforge.progress import Progress, stage
lazy from systolic.backend import SystolicBackend
lazy from systolic.im2col import conv2d_as_matmul, convolution_output_size


def _layer_label(name: str, position: str, kind: str, shape: str) -> str:
    """Build a per-layer progress label: `"features.7 [8/20] Conv2d(576x3136x8)"`.

    `name` is the dotted path within the wrapped model and `position` its place
    among the model's mapped layers, both empty for a layer built outside a
    `BackendModel`. `shape` is the backend matmul's `out x in x batch`
    dimensions, the number that actually predicts how slow the layer will be.

    `position` reflects where a layer sits in the module tree, not how far a
    forward pass has got: a model whose `forward` visits its children out of
    registration order will show them out of order.
    """
    parts = [part for part in (name, f"[{position}]" if position else "") if part]
    prefix = f"{' '.join(parts)} " if parts else ""
    return f"{prefix}{kind}({shape})"


@final
class MappedLinear(nn.Module):
    """Routes an `nn.Linear`'s matmul through a `SystolicBackend`."""

    # Deliberately not a @dataclass: nn.Module.__setattr__ needs
    # super().__init__() to run before any attribute assignment, which a
    # dataclass-generated __init__ has no hook to do, and dataclass's auto
    # __eq__/__repr__ would conflict with nn.Module's own (tensor comparisons in
    # __eq__, tree-printing repr).
    def __init__(
        self,
        inner: nn.Linear,
        backend: SystolicBackend,
        *,
        name: str = "",
        position: str = "",
        progress: Progress | None = None,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.backend = backend
        self.name = name
        self.position = position
        self.progress = progress

    @override
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2:
            raise ValueError("MappedLinear only supports 2d inputs")
        out_features, in_features = self.inner.weight.shape
        label = _layer_label(
            self.name,
            self.position,
            "Linear",
            f"{out_features}x{in_features}x{x.shape[0]}",
        )
        with stage(self.progress, label):
            # nn.Linear.weight is already (out_features, in_features); only the
            # activations need transposing to the backend's (in_features, batch)
            # convention, then the result transposed back to (batch, out_features).
            out = self.backend.matmul(self.inner.weight, x.T).T
            if self.inner.bias is not None:  # None when nn.Linear(..., bias=False)
                out = out + self.inner.bias
        return out


@final
class MappedConv2d(nn.Module):
    """Routes an `nn.Conv2d`'s matmul through a `SystolicBackend` via im2col.

    See `conv2d_as_matmul` for how a convolution becomes a matmul.

    Only `groups=1` convolutions are supported. A grouped convolution is a
    block-diagonal operation (independent per-group matmuls), which the
    current systolic array model has no representation for.
    """

    # See comment on MappedLinear `__init__`.
    def __init__(
        self,
        inner: nn.Conv2d,
        backend: SystolicBackend,
        *,
        name: str = "",
        position: str = "",
        progress: Progress | None = None,
    ) -> None:
        super().__init__()
        if inner.groups != 1:
            raise ValueError(
                f"MappedConv2d does not support grouped convolutions "
                f"(got groups={inner.groups}); only groups=1 is currently supported"
            )
        self.inner = inner
        self.backend = backend
        self.name = name
        self.position = position
        self.progress = progress

    @override
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected a 4d input, got {x.dim()}d")
        if isinstance(self.inner.padding, str):
            raise NotImplementedError("string padding is not supported by F.unfold")

        # x: (batch_size, input_channels, input_height, input_width)
        batch_size = x.shape[0]
        input_channels = self.inner.in_channels
        output_channels = self.inner.out_channels
        kernel_height, kernel_width = self.inner.kernel_size

        output_height = convolution_output_size(
            x.shape[2],
            kernel_height,
            stride=self.inner.stride[0],
            padding=self.inner.padding[0],
            dilation=self.inner.dilation[0],
        )
        output_width = convolution_output_size(
            x.shape[3],
            kernel_width,
            stride=self.inner.stride[1],
            padding=self.inner.padding[1],
            dilation=self.inner.dilation[1],
        )
        # The kernel visits one position per output pixel.
        output_position_count = output_height * output_width
        # How many numbers the kernel reads at each position it visits.
        flattened_patch_size = input_channels * kernel_height * kernel_width

        label = _layer_label(
            self.name,
            self.position,
            "Conv2d",
            f"{output_channels}x{flattened_patch_size}x{batch_size * output_position_count}",
        )
        with stage(self.progress, label):
            out = conv2d_as_matmul(
                x,
                self.inner.weight,
                stride=self.inner.stride,
                padding=self.inner.padding,
                dilation=self.inner.dilation,
                matmul=self.backend.matmul,
            )

            if self.inner.bias is not None:
                # Differs from MappedLinear: broadcasts over
                # (batch, channel, height, width), so it needs an explicit
                # (1, output_channels, 1, 1) reshape rather than relying on
                # trailing-dim broadcast.
                out = out + self.inner.bias.view(1, -1, 1, 1)
        return out
