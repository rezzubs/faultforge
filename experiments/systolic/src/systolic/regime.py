"""The profiled input regimes of a PE and the MAC inputs they carry."""

# NOTE: kept apart from `profiling_similarity` so the CLI can use `Regime` as
# an option type without loading numpy.

lazy import enum


class Regime(enum.StrEnum):
    """Which of the profiled input regimes to analyze.

    ZERO is absent on purpose: its inputs are structurally always
    `(0, 0, 0)`, so profiling records nothing for it.
    """

    First = "first"
    Active = "active"
    Drain = "drain"

    def variables_for(self) -> list[Variable]:
        """The variables that carry real information in `regime`.

        FIRST omits the partial sum and DRAIN omits activation and weight,
        because in those regimes they are structurally zero rather than
        observed.
        """
        match self:
            case Regime.First:
                return [Variable.Activation, Variable.Weight]
            case Regime.Active:
                return [Variable.Activation, Variable.Weight, Variable.PartialSum]
            case Regime.Drain:
                return [Variable.PartialSum]


class Variable(enum.StrEnum):
    """One of the three MAC inputs a PE sees."""

    Activation = "activation"
    Weight = "weight"
    PartialSum = "partial-sum"
