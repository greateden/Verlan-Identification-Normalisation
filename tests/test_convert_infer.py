from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import convert_infer as ci


def test_module_import():
    assert hasattr(ci, "generate_once")
