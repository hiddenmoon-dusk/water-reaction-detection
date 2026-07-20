from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_formal_builders_accept_reserved_generation_values():
    android_script = (
        ROOT / "android-app" / "scripts" / "build-formal-release.ps1"
    ).read_text(encoding="utf-8")
    windows_script = (
        ROOT / "scripts" / "build-windows-formal-release.ps1"
    ).read_text(encoding="utf-8")
    gradle = (ROOT / "android-app" / "app" / "build.gradle").read_text(
        encoding="utf-8"
    )

    assert "[int]$ModelGeneration = 0" in android_script
    assert "[int]$DatasetGeneration = 0" in android_script
    assert "[int]$ModelGeneration = 0" in windows_script
    assert "[int]$DatasetGeneration = 0" in windows_script
    assert "release_batch_id" in gradle
    assert "formalAppReleaseId" in gradle
    assert "app_version_code" in gradle
    assert "中文使用说明.md" in windows_script
    assert "Compress-Archive -Path (Join-Path $packageRoot '*')" in windows_script
    assert "release_batch_id" in windows_script
