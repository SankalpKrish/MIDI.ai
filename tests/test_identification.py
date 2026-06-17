import pytest

pytest.importorskip("tensorflow_hub", reason="tensorflow-hub not installed")

from src.identification import InstrumentClassifier


class TestGmProgramMap:
    def setup_method(self):
        self.classifier = InstrumentClassifier()

    def test_piano_program(self):
        assert self.classifier.gm_program_map.get("Piano") == 0

    def test_electric_guitar_program(self):
        assert self.classifier.gm_program_map.get("Electric guitar") == 27

    def test_known_program_name(self):
        assert self.classifier._get_gm_program_name(0) == "Acoustic Grand Piano"
