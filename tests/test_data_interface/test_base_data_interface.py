import pytest

from dice_ml.data_interfaces.base_data_interface import _BaseData


class TestBaseData:
    def test_base_data_initialization(self):
        with pytest.raises(TypeError):
            _BaseData({})
