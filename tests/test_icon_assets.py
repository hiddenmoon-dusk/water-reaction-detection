import subprocess
import sys
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "package-icon-assets.py"
EXPECTED_SIZES = {(size, size) for size in (16, 24, 32, 48, 64, 128, 256)}


def test_package_icon_assets_builds_multisize_ico(tmp_path):
    source = tmp_path / "source.png"
    target = tmp_path / "water-reaction-icon.ico"
    Image.new("RGBA", (1024, 1024), (0, 167, 181, 255)).save(source)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(target)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert target.is_file()
    with Image.open(target) as icon:
        assert EXPECTED_SIZES.issubset(icon.ico.sizes())
