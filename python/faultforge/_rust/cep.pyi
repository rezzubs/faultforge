from typing import ClassVar

import numpy as np
import numpy.typing as npt

class CepScheme:
    D3P1: ClassVar[CepScheme]
    D7P1: ClassVar[CepScheme]
    D15P1: ClassVar[CepScheme]

def encode_f32(arr: npt.NDArray[np.float32], scheme: CepScheme) -> None: ...
def decode_f32(arr: npt.NDArray[np.float32], scheme: CepScheme) -> None: ...
def encode_u16(arr: npt.NDArray[np.uint16], scheme: CepScheme) -> None: ...
def decode_u16(arr: npt.NDArray[np.uint16], scheme: CepScheme) -> None: ...
