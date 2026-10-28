"""O shader WebGL2 (realtime/) concorda com o render offline."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realtime"))
import validate  # noqa: E402


@pytest.mark.skipif(validate.find_chrome() is None, reason="Chrome não instalado")
def test_shader_matches_offline_shadow():
    assert validate.main() == 0
