from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_inno_setup_uses_shared_water_reaction_icon():
    installer = (ROOT / "packaging" / "windows" / "water-detection.iss").read_text(
        encoding="utf-8"
    )
    build_script = (ROOT / "scripts" / "build-windows-formal-release.ps1").read_text(
        encoding="utf-8"
    )

    assert 'GetEnv("WATER_WINDOWS_ICON_FILE")' in installer
    assert "SetupIconFile={#IconFile}" in installer
    assert "WATER_WINDOWS_ICON_FILE" in build_script
    assert "water-reaction-icon.ico" in build_script
