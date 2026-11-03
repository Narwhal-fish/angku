class DomainError(Exception):
    """Base class for all domain-level errors."""


class UnpinnedInputError(DomainError):
    """Raised when an ExperimentID is requested but some input is not pinned to an immutable id."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__("cannot compute ExperimentID: unpinned inputs: " + ", ".join(missing))


class BundleFormatError(DomainError, ValueError):
    def __init__(self, what: str, version: int, known: int) -> None:
        self.version = version
        self.known = known
        super().__init__(
            f"{what} was written by a newer angku (format {version}, this one reads up to {known}): upgrade angku"
        )
