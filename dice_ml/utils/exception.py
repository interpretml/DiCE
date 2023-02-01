"""Exceptions for the package."""
import warnings


class UserConfigValidationException(Exception):
    """An exception indicating that some user configuration is not valid.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """
    _error_code = "Invalid Configuration"
    warnings.warn("UserConfigValidationException will be deprecated from dice_ml.utils. "
                  "Please import UserConfigValidationException from raiutils.exceptions.")


class SystemException(Exception):
    """An exception indicating that some system exception happened during execution.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """
    _error_code = "System Error"
