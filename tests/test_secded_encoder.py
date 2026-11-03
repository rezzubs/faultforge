import pytest
import torch

from faultforge import BitFlip
from faultforge._impl.encoding.secded import (
    SecdedEncoding,
    _decode_into,
)
from faultforge.encoding import SecdedEncoder


@pytest.mark.parametrize("bits_per_chunk", [-8, 0, 1, 12, 65])
def test_rejects_bits_per_chunk_not_positive_multiple_of_8(bits_per_chunk: int) -> None:
    with pytest.raises(ValueError, match="positive multiple of 8"):
        _ = SecdedEncoder(bits_per_chunk=bits_per_chunk)


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().flatten().view(torch.uint8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_tensors_that_share_a_chunk_decode_separately(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(0)
    inputs = [
        torch.randn(3, 5, generator=generator).to(dtype),
        torch.randn(7, generator=generator).to(dtype),
        torch.randn(4, 2, generator=generator).to(dtype),
    ]
    encoding = SecdedEncoder(bits_per_chunk=64).encode(inputs)
    assert isinstance(encoding, SecdedEncoding)

    # A double error in the chunk that holds the end of the first tensor and
    # the start of the second. Encoded bits 3 and 71 hold the first and last
    # data bit of a 64-bit chunk.
    straddling_chunk = _bytes(inputs[0]).numel() // 8
    chunk_start = straddling_chunk * 72
    encoding.apply_faults([(BitFlip(), chunk_start + 3), (BitFlip(), chunk_start + 71)])

    contiguous = [torch.empty_like(t) for t in inputs]
    _decode_into(encoding._encoded_data, contiguous)
    # Every other element of a larger tensor isn't contiguous.
    strided = [torch.empty((*t.shape, 2), dtype=dtype)[..., 0] for t in inputs]
    assert not any(t.is_contiguous() for t in strided)
    _decode_into(encoding._encoded_data, strided)

    for contiguous_tensor, strided_tensor in zip(contiguous, strided, strict=True):
        assert torch.equal(_bytes(contiguous_tensor), _bytes(strided_tensor))
    # The faults reach both tensors that share the chunk.
    assert not torch.equal(_bytes(contiguous[0]), _bytes(inputs[0]))
    assert not torch.equal(_bytes(contiguous[1]), _bytes(inputs[1]))
    assert torch.equal(_bytes(contiguous[2]), _bytes(inputs[2]))


def test_decode_into_rejects_wrong_destinations() -> None:
    encoding = SecdedEncoder(bits_per_chunk=64).encode([torch.zeros(4), torch.zeros(2)])

    with pytest.raises(ValueError, match="Expected 2 destination tensors"):
        encoding.decode_into([torch.zeros(4)])
    with pytest.raises(ValueError, match=r"has shape \(3,\)"):
        encoding.decode_into([torch.zeros(4), torch.zeros(3)])
    with pytest.raises(ValueError, match="dtype torch.float16"):
        encoding.decode_into([torch.zeros(4), torch.zeros(2, dtype=torch.float16)])
