from typing import ClassVar

import numpy as np
import numpy.typing as npt

type ListOfArray[T: np.generic] = list[npt.NDArray[T]]

class BitPatternEncoding:
    def decode_bit_pattern_f32(
        self,
        decoded_array_element_counts: list[int],
    ) -> tuple[ListOfArray[np.uint16], list[bool]]: ...
    def decode_bit_pattern_u16(
        self,
        decoded_array_element_counts: list[int],
    ) -> tuple[ListOfArray[np.uint16], list[bool]]: ...
    def flip_n_bits(self, n: int) -> None: ...
    def clone(self) -> BitPatternEncoding: ...
    def bits_count(self) -> int: ...

class FullEncoding:
    def decode_full_f32(
        self,
    ) -> tuple[ListOfArray[np.float32], list[bool]]: ...
    def decode_full_u16(
        self,
    ) -> tuple[ListOfArray[np.uint16], list[bool]]: ...
    def flip_n_bits(self, n: int) -> None: ...
    def clone(self) -> FullEncoding: ...
    def bits_count(self) -> int: ...

class EpScheme:
    D3P1: ClassVar[EpScheme]
    D7P1: ClassVar[EpScheme]
    D15P1: ClassVar[EpScheme]

def f32_array_list_fi(
    input: ListOfArray[np.float32],
    faults_count: int,
) -> ListOfArray[np.float32]: ...
def u16_array_list_fi(
    input: ListOfArray[np.uint16],
    faults_count: int,
) -> ListOfArray[np.uint16]: ...
def u8_array_list_fi(
    input: ListOfArray[np.uint8],
    faults_count: int,
) -> ListOfArray[np.uint16]: ...
def compare_array_list_bitwise_f32(
    a: ListOfArray[np.float32],
    b: ListOfArray[np.float32],
) -> list[int]: ...
def compare_array_list_bitwise_u16(
    a: ListOfArray[np.float32],
    b: ListOfArray[np.float32],
) -> list[int]: ...
def encode_full_f32(
    input: ListOfArray[np.float32],
    bits_per_chunk: int,
) -> FullEncoding: ...
def encode_full_u16(
    input: ListOfArray[np.uint16],
    bits_per_chunk: int,
) -> FullEncoding: ...
def decode_full_f32(
    encoded: npt.NDArray[np.uint8],
    encoded_bits_count: int,
    bits_per_chunk: int,
    decoded_array_element_counts: list[int],
) -> tuple[ListOfArray[np.float32], list[bool]]: ...
def decode_full_u16(
    encoded: npt.NDArray[np.uint8],
    encoded_bits_count: int,
    bits_per_chunk: int,
    decoded_array_element_counts: list[int],
) -> tuple[ListOfArray[np.uint16], list[bool]]: ...
def encode_bit_pattern_f32(
    input: ListOfArray[np.float32],
    bit_pattern_bits: set[int],
    bit_pattern_length: int,
    bits_per_chunk: int,
) -> BitPatternEncoding: ...
def encode_bit_pattern_u16(
    input: ListOfArray[np.uint16],
    bit_pattern_bits: set[int],
    bit_pattern_length: int,
    bits_per_chunk: int,
) -> BitPatternEncoding: ...
def bit30_encode_f32(arr: npt.NDArray[np.float32]) -> None: ...
def bit30_decode_f32(arr: npt.NDArray[np.float32]) -> None: ...
def bit14_encode_u16(arr: npt.NDArray[np.uint16]) -> None: ...
def bit14_decode_u16(arr: npt.NDArray[np.uint16]) -> None: ...
def embedded_parity_encode_f32(
    arr: npt.NDArray[np.float32], scheme: EpScheme
) -> None: ...
def embedded_parity_decode_f32(
    arr: npt.NDArray[np.float32], scheme: EpScheme
) -> None: ...
def embedded_parity_encode_u16(
    arr: npt.NDArray[np.uint16], scheme: EpScheme
) -> None: ...
def embedded_parity_decode_u16(
    arr: npt.NDArray[np.uint16], scheme: EpScheme
) -> None: ...
