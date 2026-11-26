"""Convolutions as matrix multiplications."""

lazy from collections.abc import Callable, Sequence

lazy import torch.nn.functional as F
lazy from torch import Tensor


def convolution_output_size(
    input_size: int, kernel_size: int, *, stride: int, padding: int, dilation: int
) -> int:
    """The size of a convolution's output along one axis.

    Standard convolution output-size formula: pad the input, subtract the
    kernel's footprint once dilation has spread it out, then see how many
    times the stride fits.
    """
    effective_input_size = 2 * padding + input_size
    effective_kernel_size = dilation * (kernel_size - 1) + 1
    return (effective_input_size - effective_kernel_size) // stride + 1


def conv2d_as_matmul(
    inputs: Tensor,
    weight: Tensor,
    *,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    matmul: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """A 2D convolution computed with one matrix multiplication, without bias.

    - `inputs` is `(batch, in_channels, height, width)`.
    - `weight` is `(out_channels, in_channels, kernel_height, kernel_width)`.
    - `stride`, `padding` and `dilation` have one value for each of height
      and width.
    - `matmul(weights, activations)` multiplies `(out_features, in_features)`
      weights with `(in_features, columns)` activations.

    Returns `(batch, out_channels, output_height, output_width)`.

    Raises:
        ValueError: If `weight` doesn't have as many input channels as
            `inputs`, as in a grouped convolution.
    """
    # `im2col` ("image to columns") turns a convolution into a single matrix
    # multiplication. Every position the kernel visits gets its own column
    # containing the flattened values under the kernel at that position and
    # the kernel's weights get flattened into a matrix. One matmul of that
    # weight matrix against the matrix of columns then computes every output
    # value at once. The steps below build that matmul, with each tensor's
    # shape spelled out in a comment as it changes.

    # inputs: (batch_size, input_channels, input_height, input_width)
    batch_size, input_channels, input_height, input_width = inputs.shape
    output_channels, weight_input_channels, kernel_height, kernel_width = weight.shape
    if weight_input_channels != input_channels:
        raise ValueError(
            f"weight {tuple(weight.shape)} doesn't match the "
            f"{input_channels} channels of inputs {tuple(inputs.shape)}"
        )

    output_height = convolution_output_size(
        input_height,
        kernel_height,
        stride=stride[0],
        padding=padding[0],
        dilation=dilation[0],
    )
    output_width = convolution_output_size(
        input_width,
        kernel_width,
        stride=stride[1],
        padding=padding[1],
        dilation=dilation[1],
    )
    # The kernel visits one position per output pixel.
    output_position_count = output_height * output_width
    # How many numbers the kernel reads at each position it visits.
    flattened_patch_size = input_channels * kernel_height * kernel_width

    # im2col: for every position the kernel visits, extract the
    # flattened_patch_size values under it into one column, for every
    # batch element at once.
    #
    # patches: (batch_size, flattened_patch_size, number_of_output_positions)
    patches = F.unfold(
        inputs,
        (kernel_height, kernel_width),
        dilation=tuple(dilation),
        padding=tuple(padding),
        stride=tuple(stride),
    )

    # Flatten the kernel's weights into a matrix that, multiplied by one
    # patch column, produces that position's output value across all
    # output channels at once.
    #
    # weight_matrix: (output_channels, flattened_patch_size)
    weight_matrix = weight.reshape(output_channels, flattened_patch_size)

    # The matmul only has one "batch"/"free" axis - weights
    # (out_features, in_features) @ activations (in_features, batch) ->
    # (out_features, batch)). To cover every batch element with a single
    # matmul call instead of one call per batch element, lay every batch
    # element's patch columns side by side along that axis.
    #
    # Tiling batches together only needs to preserve the values a
    # plain matmul would produce, which it does: the zero-padding
    # cycles added by the array's dataflow (`shift_activations` in
    # `crates/systolic/src/array.rs`) never contribute to a real output
    # under plain multiply-accumulate. Whether batching also preserves
    # fault behavior is a separate question, answered on `XorMaskHook`'s doc
    # comment for the hooks implemented today.
    #
    # patches_matrix: (flattened_patch_size, batch_size * number_of_output_positions)
    patches_matrix = patches.permute(1, 0, 2).reshape(
        flattened_patch_size, batch_size * output_position_count
    )

    # One matmul computes every output value, for every output channel,
    # for every batch element, all at once.
    #
    # raw_output: (output_channels, batch_size * number_of_output_positions)
    raw_output = matmul(weight_matrix, patches_matrix)

    # Undo the batch-tiling from above, then move the batch axis back to
    # the front to match PyTorch's (batch, channel, height, width)
    # convention.
    #
    # output: (batch_size, output_channels, number_of_output_positions)
    output = raw_output.reshape(
        output_channels, batch_size, output_position_count
    ).permute(1, 0, 2)

    # Unflatten number_of_output_positions back into a 2d
    # (output_height, output_width) grid.
    #
    # out: (batch_size, output_channels, output_height, output_width)
    return output.reshape(batch_size, output_channels, output_height, output_width)
