import numpy as np
import pytest
librosa = pytest.importorskip("librosa")
from src.analysis import GlobalAnalyzer


class TestKeyEstimation:
    def setup_method(self):
        self.analyzer = GlobalAnalyzer()

    def _make_tone(self, root_pitch_class, sr=22050, duration=1.0):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = 440.0 * (2 ** (root_pitch_class / 12.0))
        return 0.5 * np.sin(2 * np.pi * freq * t), sr

    def test_returns_string(self):
        y, sr = self._make_tone(0)
        key = self.analyzer._estimate_key(y, sr)
        assert isinstance(key, str) and len(key) > 0

    def test_silence_returns_string(self):
        key = self.analyzer._estimate_key(np.zeros(22050), 22050)
        assert isinstance(key, str)


class TestTuningEstimation:
    def setup_method(self):
        self.analyzer = GlobalAnalyzer()

    def test_defaults_to_440_on_silence(self):
        assert self.analyzer._estimate_tuning_standard(np.zeros(22050), 22050) == 440.0

    def test_returns_float(self):
        val = self.analyzer._estimate_tuning_standard(np.random.randn(22050) * 0.01, 22050)
        assert isinstance(val, float)
