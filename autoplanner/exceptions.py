import sys


class AutoPlannerException(Exception):
    """A generic autoplanner exception"""

    def __init__(self, message):
        self.traceback = sys.exc_info()
        super().__init__(message)


class AutoPlannerLoadingError(AutoPlannerException):
    """A generic error for loading errors"""
