"""Exceptions for the package."""


class UserConfigValidationException(Exception):
    """An exception indicating that some user configuration is not valid.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """
    _error_code = "Invalid Configuration"


class SystemException(Exception):
    """An exception indicating that some system exception happened during execution.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """
    _error_code = "System Error"
