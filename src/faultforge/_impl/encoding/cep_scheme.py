"""The chunking scheme of Chunked Embedded Parity (CEP)."""

# Kept apart from `cep.py` so it can be used without loading PyTorch. For
# example, a CLI option uses it as a default value.

lazy import enum

lazy from faultforge._rust import cep


class CepScheme(enum.StrEnum):
    """How many data bits to use per parity bit.

    D3P1 should result in the best accuracy in most cases because it results in
    the biggest number of chunks per parameter - it can mitigate more faults.
    """

    D3P1 = "d3p1"
    D7P1 = "d7p1"
    D15P1 = "d15p1"

    def _to_rust(self) -> cep.Scheme:
        match self:
            case CepScheme.D3P1:
                return cep.Scheme.D3P1
            case CepScheme.D7P1:
                return cep.Scheme.D7P1
            case CepScheme.D15P1:
                return cep.Scheme.D15P1

    @staticmethod
    def default() -> CepScheme:
        """Return the default scheme."""
        return CepScheme.D3P1
