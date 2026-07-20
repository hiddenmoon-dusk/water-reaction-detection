from __future__ import annotations

import hmac
import secrets
import shutil
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import Blueprint, current_app, g, jsonify, request

from .archive import InvalidArchive, read_result_archive
from .db import get_db
from .security import generate_token, hash_token, verify_token
from .validation import InvalidPayload, validate_result_payload


bp = Blueprint("uploads", __name__)


def _state(db):
    return db.execute("SELECT * FROM app_state WHERE id = 1").fetchone()


def _current_release(db, platform: str):
    row = db.execute(
        """
        SELECT p.release_id, b.model_generation
        FROM platform_releases AS p
        JOIN release_batches AS b ON b.batch_id = p.batch_id
        WHERE p.platform = ? AND p.is_current = 1
        """,
        (platform,),
    ).fetchone()
    if row is not None or platform == "android":
        return row
    state = _state(db)
    return {
        "release_id": state["current_release_id"],
        "model_generation": state["model_generation"],
    }


def installation_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        installation_id = request.headers.get("X-Installation-ID", "")
        authorization = request.headers.get("Authorization", "")
        if not authorization.startswith("Bearer ") or not installation_id:
            return jsonify(code="authentication_required"), 401
        token = authorization.removeprefix("Bearer ").strip()
        installation = get_db().execute(
            "SELECT * FROM installations WHERE installation_id = ? AND active = 1",
            (installation_id,),
        ).fetchone()
        if installation is None or not verify_token(token, installation["token_hash"]):
            return jsonify(code="invalid_credentials"), 401
        g.installation = installation
        return view(*args, **kwargs)

    return wrapped


@bp.get("/api/v1/client/config")
def client_config():
    state = _state(get_db())
    return jsonify(
        dataset_generation=state["dataset_generation"],
        model_generation=state["model_generation"],
        current_release_id=state["current_release_id"],
        max_upload_bytes=current_app.config["MAX_RESULT_UPLOAD_BYTES"],
    )


@bp.post("/api/v1/client/register")
def register_client():
    payload = request.get_json(silent=True) or {}
    bootstrap = str(payload.get("bootstrap_token", ""))
    if not hmac.compare_digest(bootstrap, current_app.config["BOOTSTRAP_TOKEN"]):
        return jsonify(code="invalid_bootstrap_token"), 403

    db = get_db()
    platform = payload.get("client_platform", "desktop")
    if platform not in {"desktop", "android"}:
        return jsonify(code="invalid_client_platform"), 400
    release = _current_release(db, platform)
    if (
        release is None
        or payload.get("app_release_id") != release["release_id"]
        or payload.get("model_generation") != release["model_generation"]
    ):
        return jsonify(code="client_release_expired"), 409

    installation_id = secrets.token_hex(16)
    token = generate_token()
    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        """
        INSERT INTO installations (
            installation_id, token_hash, app_release_id,
            model_generation, client_platform, active, created_at
        ) VALUES (?, ?, ?, ?, ?, 1, ?)
        """,
        (
            installation_id,
            hash_token(token),
            release["release_id"],
            release["model_generation"],
            platform,
            now,
        ),
    )
    db.commit()
    return jsonify(installation_id=installation_id, token=token), 201


@bp.post("/api/v1/results")
@installation_required
def upload_result():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify(code="missing_file"), 400
    max_bytes = current_app.config["MAX_RESULT_UPLOAD_BYTES"]
    archive_data = uploaded.read(max_bytes + 1)
    if len(archive_data) > max_bytes:
        return jsonify(code="result_archive_too_large"), 413
    try:
        files = read_result_archive(
            archive_data,
            current_app.config["MAX_ARCHIVE_UNCOMPRESSED"],
        )
        payload = validate_result_payload(files["result.json"])
    except InvalidArchive as exc:
        return jsonify(code="invalid_archive", message=str(exc)), 400
    except InvalidPayload as exc:
        return jsonify(code="invalid_payload", message=str(exc)), 400

    db = get_db()
    state = _state(db)
    release = _current_release(db, g.installation["client_platform"])
    if payload["dataset_generation"] != state["dataset_generation"]:
        return jsonify(code="generation_expired"), 409
    if (
        release is None
        or payload["model_generation"] != release["model_generation"]
        or payload["app_release_id"] != release["release_id"]
    ):
        return jsonify(code="client_release_expired"), 409
    if (
        g.installation["model_generation"] != release["model_generation"]
        or g.installation["app_release_id"] != release["release_id"]
    ):
        return jsonify(code="client_release_expired"), 409

    existing = db.execute(
        "SELECT positive_count, negative_count FROM uploads WHERE upload_id = ?",
        (payload["upload_id"],),
    ).fetchone()
    if existing is not None:
        return (
            jsonify(
                status="already_received",
                positive_count=existing["positive_count"],
                negative_count=existing["negative_count"],
            ),
            208,
        )

    positive = sum(item["label"] == "已反应" for item in payload["results"])
    negative = len(payload["results"]) - positive
    storage_root = Path(current_app.config["STORAGE_ROOT"])
    final_dir = storage_root / "results" / payload["water_type"] / payload["upload_id"]
    temp_dir = storage_root / "temp" / f"{payload['upload_id']}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    try:
        for name, content in files.items():
            (temp_dir / name).write_bytes(content)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.replace(final_dir)

        now = datetime.now(timezone.utc).isoformat()
        db.execute("BEGIN")
        db.execute(
            """
            INSERT INTO uploads (
                upload_id, installation_id, water_type, mode, captured_at,
                received_at, app_release_id, model_generation,
                dataset_generation, storage_path, positive_count, negative_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["upload_id"],
                g.installation["installation_id"],
                payload["water_type"],
                payload["mode"],
                payload["captured_at"],
                now,
                payload["app_release_id"],
                payload["model_generation"],
                payload["dataset_generation"],
                str(final_dir),
                positive,
                negative,
            ),
        )
        db.executemany(
            """
            INSERT INTO tube_results (
                upload_id, tube_index, label, confidence,
                x1, y1, x2, y2
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    payload["upload_id"],
                    item["id"],
                    item["label"],
                    item["confidence"],
                    item["x1"],
                    item["y1"],
                    item["x2"],
                    item["y2"],
                )
                for item in payload["results"]
            ],
        )
        db.execute(
            "UPDATE installations SET last_seen_at = ? WHERE installation_id = ?",
            (now, g.installation["installation_id"]),
        )
        db.execute(
            "UPDATE app_state SET updated_at = ? WHERE id = 1",
            (now,),
        )
        db.commit()
    except Exception:
        db.rollback()
        shutil.rmtree(final_dir, ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return jsonify(
        status="created",
        upload_id=payload["upload_id"],
        positive_count=positive,
        negative_count=negative,
    ), 201


@bp.get("/api/v1/uploads/<upload_id>")
@installation_required
def upload_status(upload_id: str):
    row = get_db().execute(
        """
        SELECT upload_id, received_at, positive_count, negative_count
        FROM uploads
        WHERE upload_id = ?
        """,
        (upload_id,),
    ).fetchone()
    if row is None:
        return jsonify(code="not_found"), 404
    return jsonify(dict(row))
