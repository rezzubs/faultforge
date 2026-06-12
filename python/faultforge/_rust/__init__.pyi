import numpy as np
import numpy.typing as npt

type ListOfArray[T: np.generic] = list[npt.NDArray[T]]

class Fault:
    @staticmethod
    def stuck_at_0() -> Fault: ...
    @staticmethod
    def stuck_at_1() -> Fault: ...
    @staticmethod
    def flip() -> Fault: ...

def list_of_array_fault_f32(
    input: ListOfArray[np.float32],
    fault: Fault,
    target_bit: int,
) -> None: ...
def list_of_array_fault_u16(
    input: ListOfArray[np.uint16],
    fault: Fault,
    target_bit: int,
) -> None: ...
def list_of_array_fault_u8(
    input: ListOfArray[np.uint8],
    fault: Fault,
    target_bit: int,
) -> None: ...
def compare_array_list_bitwise_f32(
    a: ListOfArray[np.float32],
    b: ListOfArray[np.float32],
) -> list[int]: ...
def compare_array_list_bitwise_u16(
    a: ListOfArray[np.float32],
    b: ListOfArray[np.float32],
) -> list[int]: ...
