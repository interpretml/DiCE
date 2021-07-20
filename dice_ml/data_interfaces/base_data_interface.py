"""Module containing base class for data interfaces for dice-ml."""

from abc import ABC, abstractmethod


class _BaseData(ABC):

    def _validate_and_set_data_name(self, params):
        """Validate and set the data name."""
        if 'data_name' in params:
            self.data_name = params['data_name']
        else:
            self.data_name = 'mydata'

    def _validate_and_set_outcome_name(self, params):
        """Validate and set the outcome name."""
        if 'outcome_name' not in params:
            raise ValueError("should provide the name of outcome feature")

        if type(params['outcome_name']) is str:
            self.outcome_name = params['outcome_name']
        else:
            raise ValueError("should provide the name of outcome feature as a string")

    @abstractmethod
    def __init__(self, params):
        """The init method needs to be implemented by the inherting classes."""
        pass
