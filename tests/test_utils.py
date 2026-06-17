import json
import pytest
import numpy as np

pytest.importorskip("torchaudio", reason="torchaudio not installed")

from src.utils import NumpyEncoder


class TestNumpyEncoder:
    def test_numpy_int(self):
        result = json.dumps({"value": np.int64(42)}, cls=NumpyEncoder)
        assert '"value": 42' in result

    def test_numpy_float(self):
        result = json.dumps({"value": np.float64(3.14)}, cls=NumpyEncoder)
        assert '"value": 3.14' in result

    def test_numpy_array(self):
        result = json.dumps({"value": np.array([1, 2, 3])}, cls=NumpyEncoder)
        assert "[1, 2, 3]" in result

    def test_regular_types_still_work(self):
        result = json.dumps({"name": "test", "count": 5}, cls=NumpyEncoder)
        assert '"name": "test"' in result
        assert '"count": 5' in result
