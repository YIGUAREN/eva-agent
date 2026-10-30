"""AIG adapter errors; messages must never contain tokens or response bodies."""


class AigError(RuntimeError):
    """Base class for AIG failures."""


class AigUnavailable(AigError):
    """AIG could not be reached or returned a non-2xx response."""


class AigRejected(AigError):
    """AIG accepted the connection but rejected the task."""


class AigTimeout(AigError):
    """The scan did not finish inside the allowed wait window."""


class AigResultError(AigError):
    """AIG returned a result that cannot be parsed safely."""
