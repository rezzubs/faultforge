import numpy as np
import numpy.typing as npt

type ListOfArray[T: np.generic] = list[npt.NDArray[T]]

class FullEncoding:
    def decode_full_f32(
        self,
    ) -> tuple[ListOfArray[np.float32], list[bool]]: ...
    def decode_full_u16(
        self,
    ) -> tuple[ListOfArray[np.uint16], list[bool]]: ...
    def flip_n_bits(self, n: int) -> None: ...
    def clone(self) -> FullEncoding: ...
    def bit_count(self) -> int: ...

def encode_f32(
    input: ListOfArray[np.float32],
    bits_per_chunk: int,
) -> FullEncoding: ...
def encode_u16(
    input: ListOfArray[np.uint16],
    bits_per_chunk: int,
) -> FullEncoding: ...
def decode_f32(
    encoded: npt.NDArray[np.uint8],
    encoded_bit_count: int,
    bits_per_chunk: int,
    decoded_array_element_counts: list[int],
) -> tuple[ListOfArray[np.float32], list[bool]]: ...
def decode_u16(
    encoded: npt.NDArray[np.uint8],
    encoded_bit_count: int,
    bits_per_chunk: int,
    decoded_array_element_counts: list[int],
) -> tuple[ListOfArray[np.uint16], list[bool]]: ...
