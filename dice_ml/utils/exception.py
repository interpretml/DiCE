"""Exceptions for the package."""


class UserConfigValidationException(Exception):
    """An exception indicating that some user configuration is not valid.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """
    _error_code = "Invalid config"
