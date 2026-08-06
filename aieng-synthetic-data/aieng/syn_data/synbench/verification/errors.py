"""Exception raised when a draft task fails verification."""


class VerificationError(Exception):
    """Carries the list of reasons a draft task was rejected."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or [message]
