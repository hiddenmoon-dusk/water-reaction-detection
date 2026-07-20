from __future__ import annotations

from pathlib import Path

from flask import Flask

from .config import default_config, validate_production_config
from .db import close_db, initialize_database


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(default_config(app.instance_path))
    if test_config:
        app.config.update(test_config)
    validate_production_config(app.config)

    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    storage = Path(app.config["STORAGE_ROOT"])
    for relative in (
        "results/污水",
        "results/生活用水",
        "results/养殖水体",
        "releases",
        "temp",
        "backups",
    ):
        (storage / relative).mkdir(parents=True, exist_ok=True)

    app.teardown_appcontext(close_db)
    initialize_database(app)
    _register_blueprints(app)
    return app


def _register_blueprints(app: Flask) -> None:
    for module_name in ("uploads", "public", "admin"):
        try:
            module = __import__(f"water_server.{module_name}", fromlist=["bp"])
        except ModuleNotFoundError:
            continue
        app.register_blueprint(module.bp)
