class DomainError(Exception):
    """Base class for all domain-level errors."""


class UnpinnedInputError(DomainError):
    """Raised when an ExperimentID is requested but some input is not pinned to an immutable id."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(
            "cannot compute ExperimentID: unpinned inputs: " + ", ".join(missing)
        )
