import pytest

from water_server import create_app
from water_server.config import default_config


def _set_valid_production_environment(monkeypatch):
    values = {
        "WATER_PRODUCTION": "true",
        "WATER_SECRET_KEY": "a" * 64,
        "WATER_ADMIN_INITIAL_PASSWORD": "production-admin-password",
        "WATER_BOOTSTRAP_TOKEN": "production-bootstrap-token",
        "WATER_PUBLIC_BASE_URL": "https://water.example.test",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_production_rejects_placeholder_secrets(monkeypatch):
    monkeypatch.setenv("WATER_PRODUCTION", "true")
    monkeypatch.setenv("WATER_PUBLIC_BASE_URL", "https://water.example.test")
    monkeypatch.delenv("WATER_SECRET_KEY", raising=False)
    monkeypatch.delenv("WATER_ADMIN_INITIAL_PASSWORD", raising=False)
    monkeypatch.delenv("WATER_BOOTSTRAP_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="WATER_SECRET_KEY"):
        create_app()


def test_production_rejects_insecure_public_url(monkeypatch):
    _set_valid_production_environment(monkeypatch)
    monkeypatch.setenv("WATER_PUBLIC_BASE_URL", "http://water.example.test")

    with pytest.raises(RuntimeError, match="HTTPS"):
        create_app()


def test_production_rejects_short_secret_key(monkeypatch):
    _set_valid_production_environment(monkeypatch)
    monkeypatch.setenv("WATER_SECRET_KEY", "too-short")

    with pytest.raises(RuntimeError, match="WATER_SECRET_KEY"):
        create_app()


def test_production_accepts_explicit_secrets(monkeypatch, tmp_path):
    _set_valid_production_environment(monkeypatch)

    app = create_app(
        {
            "DATABASE": str(tmp_path / "instance" / "app.db"),
            "STORAGE_ROOT": str(tmp_path / "storage"),
            "SESSION_COOKIE_SECURE": True,
        }
    )

    assert app.config["PRODUCTION"] is True
    assert app.config["PUBLIC_BASE_URL"] == "https://water.example.test"


def test_development_defaults_are_not_real_production_values(monkeypatch, tmp_path):
    for name in (
        "WATER_PRODUCTION",
        "WATER_SECRET_KEY",
        "WATER_ADMIN_INITIAL_PASSWORD",
        "WATER_BOOTSTRAP_TOKEN",
        "WATER_PUBLIC_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    config = default_config(str(tmp_path / "instance"))

    assert config["PRODUCTION"] is False
    assert config["ADMIN_INITIAL_PASSWORD"] == "change-me-before-use"
    assert config["PUBLIC_BASE_URL"] == "https://example.invalid"
