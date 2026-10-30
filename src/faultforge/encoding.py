"""Encoding of tensors.

An `Encoder` turns a list of tensors into an `Encoding`: a protected
representation that can be decoded back into tensors, and that has faults
applied to it directly by bit index.

`TensorEncoder`/`TensorEncoding` are the subset whose protected representation
is itself a list of tensors. This is what allows them to be composed with
`EncoderSequence`, since one encoder's output tensors can be fed into the next
encoder's input.

`InPlaceEncoder`/`InPlaceEncoding` are a convenience base class for
`TensorEncoder`/`TensorEncoding` implementations that encode each tensor
independently into a tensor of the same shape and dtype, e.g. `CepEncoder` and
`MsetEncoder`. Implementing `encode_float32`/`encode_float16` (and their
`decode_*` counterparts) is enough; batching over the input list, dtype
dispatch, and bit-count tracking are handled by the base class.

`EncodedModule` wraps a `torch.nn.Module` so its parameters are stored through
an `Encoder`, decoding them on demand and letting faults be applied to the
encoded memory directly rather than the live parameters.

See the various `*Encoder` classes for details on each technique.
"""

from faultforge._impl.encoding.abc import (
    Encoder,
    Encoding,
    InPlaceEncoder,
    InPlaceEncoding,
    TensorEncoder,
    TensorEncoding,
)
from faultforge._impl.encoding.cep import (
    CepEncoder,
    CepEncoding,
    CepScheme,
)
from faultforge._impl.encoding.identity import (
    IdentityEncoder,
    IdentityEncoding,
)
from faultforge._impl.encoding.mset import (
    MsetEncoder,
    MsetEncoding,
)
from faultforge._impl.encoding.nn import EncodedModule
from faultforge._impl.encoding.secded import (
    SecdedEncoder,
    SecdedEncoding,
)
from faultforge._impl.encoding.sequence import (
    EncoderSequence,
    EncodingSequence,
)

__all__ = [
    "CepEncoder",
    "CepEncoding",
    "CepScheme",
    "EncodedModule",
    "Encoder",
    "EncoderSequence",
    "Encoding",
    "EncodingSequence",
    "IdentityEncoder",
    "IdentityEncoding",
    "InPlaceEncoder",
    "InPlaceEncoding",
    "MsetEncoder",
    "MsetEncoding",
    "SecdedEncoder",
    "SecdedEncoding",
    "TensorEncoder",
    "TensorEncoding",
]
