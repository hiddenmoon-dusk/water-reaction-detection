from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from .db import get_db
from .exports import build_results_export, clear_results
from .release_batches import (
    InvalidReleaseBatch,
    _safe_original_filename,
    publish_bundle,
    reserve_batch_with_callback,
)
from .releases import InvalidRelease, publish_desktop_release
from .security import generate_csrf_token, hash_password, verify_password


bp = Blueprint("admin", __name__, url_prefix="/admin")


def _source_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    return (forwarded.split(",")[0].strip() if forwarded else request.remote_addr) or "unknown"


def _audit(action: str, detail: str = "", *, commit: bool = True) -> None:
    db = get_db()
    db.execute(
        """
        INSERT INTO admin_audit(action, source_ip, detail, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (action, _source_ip(), detail[:1000], datetime.now(timezone.utc).isoformat()),
    )
    if commit:
        db.commit()


def admin_required(view=None, *, json_response: bool = False):
    def decorator(protected_view):
        @wraps(protected_view)
        def wrapped(*args, **kwargs):
            if not session.get("admin"):
                if json_response:
                    return jsonify(error="authentication_required"), 401
                return redirect(url_for("admin.login"))
            return protected_view(*args, **kwargs)

        return wrapped

    return decorator if view is None else decorator(view)


def csrf_required(view=None, *, json_response: bool = False):
    def decorator(protected_view):
        @wraps(protected_view)
        def wrapped(*args, **kwargs):
            expected = session.get("csrf_token", "")
            supplied = request.headers.get("X-CSRF-Token") or request.form.get(
                "csrf_token", ""
            )
            if (
                not expected
                or not supplied
                or not secrets.compare_digest(expected, supplied)
            ):
                if json_response:
                    return jsonify(error="csrf_failed"), 403
                abort(400, description="CSRF validation failed")
            return protected_view(*args, **kwargs)

        return wrapped

    return decorator if view is None else decorator(view)


def _too_many_failures(db, source_ip: str) -> bool:
    threshold = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    count = db.execute(
        """
        SELECT COUNT(*) FROM login_attempts
        WHERE source_ip = ? AND successful = 0 AND attempted_at >= ?
        """,
        (source_ip, threshold),
    ).fetchone()[0]
    return count >= 5


@bp.route("/login", methods=("GET", "POST"))
def login():
    if request.method == "GET":
        return render_template("admin_login.html", error=None)

    db = get_db()
    source_ip = _source_ip()
    if _too_many_failures(db, source_ip):
        return render_template("admin_login.html", error="尝试次数过多，请稍后再试。"), 429

    password = request.form.get("password", "")
    state = db.execute("SELECT admin_password_hash FROM app_state WHERE id = 1").fetchone()
    successful = verify_password(password, state["admin_password_hash"])
    db.execute(
        """
        INSERT INTO login_attempts(source_ip, successful, attempted_at)
        VALUES (?, ?, ?)
        """,
        (source_ip, int(successful), datetime.now(timezone.utc).isoformat()),
    )
    db.commit()
    if not successful:
        _audit("login_failed")
        return render_template("admin_login.html", error="密码错误。"), 401

    session.clear()
    session["admin"] = True
    session["csrf_token"] = generate_csrf_token()
    session.permanent = True
    _audit("login_success")
    return redirect(url_for("admin.dashboard"))


@bp.get("")
@admin_required
def dashboard():
    db = get_db()
    state = db.execute("SELECT * FROM app_state WHERE id = 1").fetchone()
    release = db.execute(
        "SELECT * FROM desktop_releases WHERE is_current = 1"
    ).fetchone()
    upload_count = db.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
    storage_root = Path(current_app.config["STORAGE_ROOT"]) / "results"
    storage_bytes = sum(
        path.stat().st_size for path in storage_root.rglob("*") if path.is_file()
    )
    return render_template(
        "admin_dashboard.html",
        state=state,
        release=release,
        upload_count=upload_count,
        storage_bytes=storage_bytes,
        csrf_token=session["csrf_token"],
    )


@bp.post("/logout")
@admin_required
@csrf_required
def logout():
    session.clear()
    return redirect(url_for("public.index"))


@bp.post("/password")
@admin_required
@csrf_required
def change_password():
    current_password = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    db = get_db()
    state = db.execute("SELECT admin_password_hash FROM app_state WHERE id = 1").fetchone()
    if not verify_password(current_password, state["admin_password_hash"]):
        return jsonify(code="wrong_password"), 403
    if len(new_password) < 12:
        return jsonify(code="weak_password"), 400
    db.execute(
        "UPDATE app_state SET admin_password_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
        (hash_password(new_password),),
    )
    db.commit()
    _audit("password_changed")
    return jsonify(status="password_changed")


@bp.get("/results.zip")
@admin_required
def download_results():
    path = build_results_export(get_db(), Path(current_app.config["STORAGE_ROOT"]))
    _audit("results_downloaded")
    response = send_file(
        path,
        as_attachment=True,
        download_name=f"检测结果-{datetime.now():%Y%m%d-%H%M%S}.zip",
        mimetype="application/zip",
    )
    response.call_on_close(lambda: path.unlink(missing_ok=True))
    return response


@bp.post("/releases/desktop")
@admin_required
@csrf_required
def upload_desktop_release():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify(code="missing_file"), 400
    temp_dir = Path(current_app.config["STORAGE_ROOT"]) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    source = temp_dir / f"release-upload-{secrets.token_hex(8)}.zip"
    try:
        uploaded.save(source)

        def insert_audit(activated):
            _audit(
                "desktop_release_uploaded",
                activated[0]["release_id"],
                commit=False,
            )

        result = publish_desktop_release(
            source,
            get_db(),
            original_filename=_safe_original_filename(
                uploaded.filename,
                "release.zip",
            ),
            before_commit=insert_audit,
        )
    except InvalidRelease as exc:
        return jsonify(code="invalid_release", message=str(exc)), 400
    finally:
        source.unlink(missing_ok=True)
    return jsonify(
        release_id=result["release_id"],
        model_generation=result["model_generation"],
        sha256=result["sha256"],
    ), 201


@bp.post("/releases/batches/reserve")
@admin_required(json_response=True)
@csrf_required(json_response=True)
def reserve_release_batch():
    def insert_audit(reserved):
        _audit("release_batch_reserved", reserved.batch_id, commit=False)

    reserved = reserve_batch_with_callback(get_db(), insert_audit)
    return jsonify(
        batch_id=reserved.batch_id,
        android_release_id=reserved.android_release_id,
        desktop_release_id=reserved.desktop_release_id,
        model_generation=reserved.model_generation,
        dataset_generation=reserved.dataset_generation,
        expires_at=reserved.expires_at.isoformat(),
    ), 201


@bp.post("/releases/bundle")
@admin_required(json_response=True)
@csrf_required(json_response=True)
def upload_release_bundle():
    allowed_form_fields = {"batch_id", "csrf_token"}
    allowed_file_fields = {"desktop", "android"}
    if set(request.form) - allowed_form_fields or set(request.files) - allowed_file_fields:
        return jsonify(
            error="invalid_release", message="unknown multipart field"
        ), 400
    if len(request.form.getlist("batch_id")) != 1 or any(
        len(request.form.getlist(name)) > 1
        for name in request.form
        if name != "batch_id"
    ):
        return jsonify(
            error="invalid_release", message="duplicate or missing batch_id field"
        ), 400
    if any(len(request.files.getlist(name)) != 1 for name in request.files):
        return jsonify(
            error="invalid_release", message="duplicate artifact field"
        ), 400

    batch_id = request.form.get("batch_id", "").strip()
    uploads = {
        platform: request.files.get(platform)
        for platform in ("desktop", "android")
    }
    uploads = {
        platform: uploaded
        for platform, uploaded in uploads.items()
        if uploaded is not None and uploaded.filename
    }
    temp_paths: dict[str, Path] = {}
    try:
        temp_dir = Path(current_app.config["STORAGE_ROOT"]) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        for platform, uploaded in uploads.items():
            suffix = ".zip" if platform == "desktop" else ".apk"
            path = temp_dir / f"bundle-{platform}-{secrets.token_hex(16)}{suffix}"
            temp_paths[platform] = path
            uploaded.save(path)

        def insert_audits(activated):
            for item in activated:
                _audit(
                    f"{item['platform']}_release_uploaded",
                    item["release_id"],
                    commit=False,
                )

        result = publish_bundle(
            get_db(),
            batch_id,
            desktop_path=temp_paths.get("desktop"),
            android_path=temp_paths.get("android"),
            original_filenames={
                platform: uploaded.filename
                for platform, uploaded in uploads.items()
            },
            before_commit=insert_audits,
        )
    except InvalidReleaseBatch as exc:
        return jsonify(error="invalid_release", message=str(exc)), 400
    finally:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)

    return jsonify(result), 201


@bp.post("/releases/mobile")
@admin_required
@csrf_required
def upload_mobile_release():
    return jsonify(code="mobile_not_implemented", message="功能筹备中"), 501


@bp.post("/results/clear")
@admin_required
@csrf_required
def clear_all_results():
    password = request.form.get("password", "")
    db = get_db()
    state = db.execute("SELECT admin_password_hash FROM app_state WHERE id = 1").fetchone()
    if not verify_password(password, state["admin_password_hash"]):
        return jsonify(code="wrong_password"), 403
    generation = clear_results(db, Path(current_app.config["STORAGE_ROOT"]))
    _audit("results_cleared", f"dataset_generation={generation}")
    return jsonify(status="cleared", dataset_generation=generation)
