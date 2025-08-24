from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import detect_infer as di


def test_tokenize_basic():
    tokens = di.tokenize_basic("salut mon pote")
    assert tokens == ["salut", "mon", "pote"]
