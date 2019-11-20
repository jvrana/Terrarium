import sys


class TerrariumError(Exception):
    """Generatic base terrarium exception."""


class AutoPlannerException(TerrariumError):
    """A generic autoplanner exception."""

    def __init__(self, message):
        self.traceback = sys.exc_info()
        super().__init__(message)


class AutoPlannerLoadingError(AutoPlannerException):
    """A generic error for loading errors."""


class TerrariumJSONParseError(TerrariumError):
    """Exception for input file parsing errors."""


class ValidationError(TerrariumError):
    pass
