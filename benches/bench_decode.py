"""Benchmarks for decoding encodings with faults.

Compares a full decode (`Encoding.decode_into`) with a decode of only what
faults changed (`Encoding.decode_changes_into`). Run with `just bench-python`.
"""

import functools
import random

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor

from faultforge import BitFlip
from faultforge.encoding import (
    Encoding,
    SecdedEncoder,
)

# 16 MiB of float32.
_SHAPES = [(1024, 1024)] * 4
_BIT_ERROR_RATES = [1e-4, 1e-3, 1e-2]
_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    ),
]


@functools.cache
def _encoding() -> Encoding:
    generator = torch.Generator().manual_seed(0)
    return SecdedEncoder(bits_per_chunk=64).encode(
        [torch.randn(shape, generator=generator) for shape in _SHAPES]
    )


@functools.cache
def _fault_free_decode() -> list[Tensor]:
    return _encoding().decode()


@functools.cache
def _faulty_encoding(bit_error_rate: float) -> Encoding:
    """A clone of `_encoding` with unique random bit flips, like an experiment run."""
    encoding = _encoding().clone()
    bit_count = encoding.bit_count()
    target_bits = random.Random(0).sample(
        range(bit_count), round(bit_error_rate * bit_count)
    )
    encoding.apply_faults([(BitFlip(), bit) for bit in target_bits])
    return encoding


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("bit_error_rate", _BIT_ERROR_RATES)
@pytest.mark.parametrize("mode", ["full", "changes"])
def bench_decode(
    benchmark: BenchmarkFixture, mode: str, bit_error_rate: float, device: str
) -> None:
    encoding = _faulty_encoding(bit_error_rate)
    benchmark.group = f"{device}, BER {bit_error_rate:g}"

    def setup() -> tuple[tuple[list[Tensor]], dict[str, object]]:
        # A decode of only what faults changed needs the decode without faults.
        # Both modes start from it, so they only differ in what they write.
        dest = [t.to(device, copy=True) for t in _fault_free_decode()]
        return (dest,), {}

    def decode(dest: list[Tensor]) -> None:
        match mode:
            case "full":
                encoding.decode_into(dest)
            case "changes":
                encoding.decode_changes_into(dest)
        # CUDA copies are asynchronous.
        if device == "cuda":
            torch.cuda.synchronize()

    benchmark.pedantic(decode, setup=setup, rounds=5, warmup_rounds=1)
