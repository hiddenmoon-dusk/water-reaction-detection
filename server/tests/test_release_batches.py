import io
import json
import os
import sqlite3
import stat
import struct
import subprocess
import warnings
import zipfile
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread

import pytest
import water_server.release_batches as release_batches_module
from werkzeug.datastructures import MultiDict

from water_server.db import get_db
from water_server.release_batches import (
    InvalidReleaseBatch,
    publish_bundle,
    read_android_artifact,
    read_desktop_artifact,
    reserve_batch,
    verify_apk_signature,
)


def _manifest(reserved, platform, **overrides):
    payload = {
        "schema_version": 1,
        "release_batch_id": reserved.batch_id,
        "app_release_id": f"{reserved.batch_id}-{platform}",
        "model_generation": reserved.model_generation,
        "dataset_generation": reserved.dataset_generation,
        "app_version_code": 17,
        "app_version_name": "1.7.0",
    }
    payload.update(overrides)
    return payload


def _desktop_artifact(tmp_path, reserved, *, name="desktop.zip", **overrides):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "release.json",
            json.dumps(_manifest(reserved, "desktop", **overrides)),
        )
        archive.writestr("WaterApp.exe", b"exe")
        archive.writestr("models/reaction_classifier.h5", b"classifier")
        archive.writestr("models/yolov8n.pt", b"detector")
    return path


def _android_artifact(tmp_path, reserved, *, name="mobile.apk", **overrides):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "assets/model-manifest.json",
            json.dumps(_manifest(reserved, "android", **overrides)),
        )
        archive.writestr("assets/detector.tflite", b"detector")
        archive.writestr("assets/classifier.tflite", b"classifier")
    return path


def _append_member(path, filename, data=b"unsafe", *, mode=None):
    with zipfile.ZipFile(path, "a") as archive:
        if mode is None:
            archive.writestr(filename, data)
        else:
            info = zipfile.ZipInfo(filename)
            info.create_system = 3
            info.external_attr = mode << 16
            archive.writestr(info, data)


def _corrupt_stored_member(path, filename, *, compressed_offset=0):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(filename)
    with path.open("r+b") as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
        fields = struct.unpack("<IHHHHHIIIHH", header)
        data_offset = info.header_offset + 30 + fields[9] + fields[10]
        stream.seek(data_offset + compressed_offset)
        original = stream.read(1)
        stream.seek(data_offset + compressed_offset)
        stream.write(bytes([original[0] ^ 0xFF]))


def _desktop_with_compressed_classifier(tmp_path, reserved, compression):
    path = tmp_path / f"desktop-{compression}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "release.json", json.dumps(_manifest(reserved, "desktop"))
        )
        archive.writestr("WaterApp.exe", b"exe")
        archive.writestr(
            "models/reaction_classifier.h5",
            b"classifier-model-data" * 4096,
            compress_type=compression,
        )
        archive.writestr("models/yolov8n.pt", b"detector")
    return path


def _reserve(app):
    with app.app_context():
        return reserve_batch(get_db())


def _allow_test_signer(monkeypatch):
    monkeypatch.setattr(
        "water_server.release_batches.verify_apk_signature",
        lambda _path: "aa" * 32,
    )


def test_admin_can_reserve_release_batch(admin_client):
    statements = []
    with admin_client.application.app_context():
        db = get_db()
        db.set_trace_callback(statements.append)
        try:
            response = admin_client.post("/admin/releases/batches/reserve")
        finally:
            db.set_trace_callback(None)

        audit = db.execute(
            """
            SELECT action, detail
            FROM admin_audit
            WHERE action = 'release_batch_reserved'
            """
        ).fetchone()

    assert response.status_code == 201
    payload = response.get_json()
    batch_id = payload["batch_id"]
    assert payload["model_generation"] == 2
    assert payload["dataset_generation"] == 1
    assert len(batch_id) == 32
    assert payload["android_release_id"] == f"{batch_id}-android"
    assert payload["desktop_release_id"] == f"{batch_id}-desktop"
    assert payload["expires_at"].endswith("+00:00")
    commits = [statement for statement in statements if statement == "COMMIT"]
    assert len(commits) == 1
    assert audit is not None
    assert audit["detail"] == batch_id


def test_audit_failure_rolls_back_release_batch(admin_client):
    app = admin_client.application
    app.config["PROPAGATE_EXCEPTIONS"] = False
    with app.app_context():
        db = get_db()
        db.execute("DELETE FROM admin_audit")
        db.execute(
            """
            CREATE TRIGGER fail_release_batch_audit
            BEFORE INSERT ON admin_audit
            WHEN NEW.action = 'release_batch_reserved'
            BEGIN
                SELECT RAISE(FAIL, 'injected audit failure');
            END
            """
        )
        db.commit()

    response = admin_client.post("/admin/releases/batches/reserve")

    assert response.status_code == 500
    with app.app_context():
        db = get_db()
        batch_count = db.execute(
            "SELECT COUNT(*) FROM release_batches"
        ).fetchone()[0]
        audit_count = db.execute("SELECT COUNT(*) FROM admin_audit").fetchone()[0]
        db.execute("DROP TRIGGER fail_release_batch_audit")
        db.commit()

    assert batch_count == 0
    assert audit_count == 0

    retry = admin_client.post("/admin/releases/batches/reserve")

    assert retry.status_code == 201
    assert retry.get_json()["model_generation"] == 2


def test_release_batch_reservation_requires_login(client):
    response = client.post("/admin/releases/batches/reserve")

    assert response.status_code == 401
    assert response.mimetype == "application/json"
    assert response.get_json() == {"error": "authentication_required"}
    with client.application.app_context():
        batch_count = get_db().execute(
            "SELECT COUNT(*) FROM release_batches"
        ).fetchone()[0]
    assert batch_count == 0


@pytest.mark.parametrize("csrf_token", [None, "incorrect-token"])
def test_release_batch_reservation_requires_valid_csrf(admin_client, csrf_token):
    if csrf_token is None:
        admin_client.environ_base.pop("HTTP_X_CSRF_TOKEN")
    else:
        admin_client.environ_base["HTTP_X_CSRF_TOKEN"] = csrf_token

    response = admin_client.post("/admin/releases/batches/reserve")

    assert response.status_code == 403
    assert response.mimetype == "application/json"
    assert response.get_json() == {"error": "csrf_failed"}
    with admin_client.application.app_context():
        batch_count = get_db().execute(
            "SELECT COUNT(*) FROM release_batches"
        ).fetchone()[0]
    assert batch_count == 0


def test_reserve_batch_allocates_shared_platform_release_ids(app):
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)

    assert app.config["RELEASE_RESERVATION_HOURS"] == 24
    assert app.config["RELEASE_ORPHAN_GRACE_HOURS"] == 24

    with app.app_context():
        db = get_db()
        first = reserve_batch(db, now)
        second = reserve_batch(db, now + timedelta(seconds=1))
        rows = db.execute(
            """
            SELECT batch_id, model_generation, dataset_generation, status,
                   reserved_at, expires_at
            FROM release_batches
            ORDER BY model_generation
            """
        ).fetchall()

    assert first.model_generation == 2
    assert second.model_generation == 3
    assert first.dataset_generation == 1
    assert first.batch_id != second.batch_id
    assert first.android_release_id == f"{first.batch_id}-android"
    assert first.desktop_release_id == f"{first.batch_id}-desktop"
    assert first.reserved_at == now
    assert first.expires_at == now + timedelta(hours=24)
    assert [row["status"] for row in rows] == ["reserved", "reserved"]
    assert rows[0]["reserved_at"] == now.isoformat()
    assert rows[0]["expires_at"] == (now + timedelta(hours=24)).isoformat()


def test_reserve_batch_uses_highest_generation_as_floor(app):
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)

    with app.app_context():
        db = get_db()
        db.execute(
            """
            INSERT INTO release_batches (
                batch_id, model_generation, dataset_generation, status,
                reserved_at, expires_at, published_at
            ) VALUES ('historical', 5, 1, 'expired', ?, ?, NULL)
            """,
            (now.isoformat(), (now + timedelta(hours=24)).isoformat()),
        )
        db.commit()

        after_history = reserve_batch(db, now)

        db.execute("UPDATE app_state SET model_generation = 10 WHERE id = 1")
        db.commit()
        after_state = reserve_batch(db, now + timedelta(seconds=1))

        db.execute(
            """
            INSERT INTO desktop_releases (
                release_id, model_generation, original_filename, stored_path,
                sha256, uploaded_at, is_current
            ) VALUES ('legacy-high', 15, 'release.zip', 'release.zip',
                      'sha256', ?, 0)
            """,
            (now.isoformat(),),
        )
        db.commit()
        after_legacy_release = reserve_batch(db, now + timedelta(seconds=2))

        generations = [
            row["model_generation"]
            for row in db.execute(
                "SELECT model_generation FROM release_batches "
                "ORDER BY model_generation"
            )
        ]

    assert after_history.model_generation == 6
    assert after_state.model_generation == 11
    assert after_legacy_release.model_generation == 16
    assert generations == [5, 6, 11, 16]


def test_reserve_batch_rejects_outer_transaction_without_rolling_it_back(app):
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)

    with app.app_context():
        db = get_db()
        db.execute(
            """
            INSERT INTO admin_audit (action, detail, created_at)
            VALUES ('caller-owned', 'must survive', ?)
            """,
            (now.isoformat(),),
        )
        assert db.in_transaction

        with pytest.raises(RuntimeError, match="own transaction"):
            reserve_batch(db, now)

        assert db.in_transaction
        pending = db.execute(
            "SELECT action FROM admin_audit WHERE action = 'caller-owned'"
        ).fetchone()
        assert pending["action"] == "caller-owned"
        db.commit()
        committed = db.execute(
            "SELECT action FROM admin_audit WHERE action = 'caller-owned'"
        ).fetchone()

    assert committed["action"] == "caller-owned"


def test_reserve_batch_parses_string_reservation_hours(app):
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    app.config["RELEASE_RESERVATION_HOURS"] = "2"

    with app.app_context():
        reserved = reserve_batch(get_db(), now)

    assert reserved.expires_at == now + timedelta(hours=2)


@pytest.mark.parametrize(
    "hours",
    ["invalid", "0", 0, -1, True, False, 1.5, "+2", " 2", "2.0"],
)
def test_reserve_batch_rejects_invalid_reservation_hours(app, hours):
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    app.config["RELEASE_RESERVATION_HOURS"] = hours

    with app.app_context():
        with pytest.raises(ValueError, match="positive integer"):
            reserve_batch(get_db(), now)


def test_reserve_batch_rejects_naive_now(app):
    now = datetime(2026, 7, 16)

    with app.app_context():
        with pytest.raises(ValueError, match="timezone-aware"):
            reserve_batch(get_db(), now)


def test_reserve_batch_normalizes_now_to_utc(app):
    local_time = datetime(
        2026,
        7,
        16,
        8,
        tzinfo=timezone(timedelta(hours=8)),
    )
    expected = datetime(2026, 7, 16, tzinfo=timezone.utc)

    with app.app_context():
        db = get_db()
        reserved = reserve_batch(db, local_time)
        row = db.execute(
            "SELECT reserved_at FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()

    assert reserved.reserved_at == expected
    assert reserved.reserved_at.tzinfo is timezone.utc
    assert row["reserved_at"] == expected.isoformat()


def test_reserve_batch_reports_missing_app_state(app):
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)

    with app.app_context():
        db = get_db()
        db.execute("DELETE FROM app_state WHERE id = 1")
        db.commit()

        with pytest.raises(RuntimeError, match="app_state"):
            reserve_batch(db, now)

        assert not db.in_transaction


def test_reserve_batch_waits_for_writer_then_uses_committed_generation(app):
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    holder = sqlite3.connect(app.config["DATABASE"], timeout=5)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute(
        """
        INSERT INTO release_batches (
            batch_id, model_generation, dataset_generation, status,
            reserved_at, expires_at, published_at
        ) VALUES ('held-generation', 2, 1, 'reserved', ?, ?, NULL)
        """,
        (now.isoformat(), (now + timedelta(hours=24)).isoformat()),
    )

    attempting_lock = Event()
    finished = Event()
    results = []
    errors = []

    def reserve_from_second_connection():
        with app.app_context():
            worker_db = sqlite3.connect(app.config["DATABASE"], timeout=5)
            worker_db.row_factory = sqlite3.Row

            def trace(statement):
                if statement.startswith("BEGIN IMMEDIATE"):
                    attempting_lock.set()

            worker_db.set_trace_callback(trace)
            try:
                results.append(reserve_batch(worker_db, now))
            except BaseException as exc:
                errors.append(exc)
            finally:
                worker_db.close()
                finished.set()

    worker = Thread(target=reserve_from_second_connection)
    worker.start()
    try:
        assert attempting_lock.wait(timeout=2)
        assert not finished.wait(timeout=0.2)
        holder.commit()
        assert finished.wait(timeout=5)
        worker.join(timeout=1)
    finally:
        if holder.in_transaction:
            holder.rollback()
        holder.close()
        worker.join(timeout=1)

    assert not worker.is_alive()
    assert errors == []
    assert results[0].model_generation == 3


def test_bundle_route_publishes_both_platforms_atomically(
    admin_client, tmp_path, monkeypatch
):
    _allow_test_signer(monkeypatch)
    reserved = _reserve(admin_client.application)
    desktop = _desktop_artifact(tmp_path, reserved)
    android = _android_artifact(tmp_path, reserved)

    with desktop.open("rb") as desktop_stream, android.open("rb") as android_stream:
        response = admin_client.post(
            "/admin/releases/bundle",
            data={
                "batch_id": reserved.batch_id,
                "desktop": (desktop_stream, "desktop.zip"),
                "android": (android_stream, "mobile.apk"),
            },
        )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["batch_id"] == reserved.batch_id
    assert payload["status"] == "published"
    assert {item["platform"] for item in payload["platforms"]} == {
        "desktop",
        "android",
    }
    assert all(item["idempotent"] is False for item in payload["platforms"])
    with admin_client.application.app_context():
        db = get_db()
        rows = db.execute(
            "SELECT platform, release_id, is_current FROM platform_releases"
        ).fetchall()
        batch = db.execute(
            "SELECT status, published_at FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        audits = db.execute(
            "SELECT action, detail FROM admin_audit "
            "WHERE action IN ('desktop_release_uploaded', 'android_release_uploaded')"
        ).fetchall()

    assert {(row["platform"], row["is_current"]) for row in rows} == {
        ("desktop", 1),
        ("android", 1),
    }
    assert batch["status"] == "published"
    assert batch["published_at"] is not None
    assert {row["detail"] for row in audits} == {
        f"{reserved.batch_id}-desktop",
        f"{reserved.batch_id}-android",
    }


def test_android_then_desktop_transitions_partial_without_changing_legacy_state(
    app, tmp_path, monkeypatch
):
    _allow_test_signer(monkeypatch)
    reserved = _reserve(app)
    android = _android_artifact(tmp_path, reserved)
    desktop = _desktop_artifact(tmp_path, reserved)

    with app.app_context():
        db = get_db()
        before = dict(db.execute("SELECT * FROM app_state WHERE id = 1").fetchone())
        android_result = publish_bundle(
            db, reserved.batch_id, android_path=android
        )
        after_android = dict(
            db.execute("SELECT * FROM app_state WHERE id = 1").fetchone()
        )
        first_published_at = db.execute(
            "SELECT published_at FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()[0]
        desktop_result = publish_bundle(
            db, reserved.batch_id, desktop_path=desktop
        )
        final_batch = db.execute(
            "SELECT status, published_at FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()

    assert android_result["status"] == "partial"
    assert after_android["current_release_id"] == before["current_release_id"]
    assert after_android["model_generation"] == before["model_generation"]
    assert first_published_at is not None
    assert desktop_result["status"] == "published"
    assert final_batch["published_at"] == first_published_at
    assert after_android["dataset_generation"] == before["dataset_generation"]


def test_bundle_same_sha_is_idempotent_and_different_sha_is_rejected(
    app, tmp_path, monkeypatch
):
    _allow_test_signer(monkeypatch)
    reserved = _reserve(app)
    android = _android_artifact(tmp_path, reserved)

    with app.app_context():
        db = get_db()
        first = publish_bundle(db, reserved.batch_id, android_path=android)
        retry = publish_bundle(db, reserved.batch_id, android_path=android)
        changed = _android_artifact(
            tmp_path,
            reserved,
            name="changed.apk",
            app_version_name="1.7.1",
        )
        with pytest.raises(InvalidReleaseBatch, match="different artifact"):
            publish_bundle(db, reserved.batch_id, android_path=changed)
        rows = db.execute(
            "SELECT COUNT(*) FROM platform_releases WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()[0]

    assert first["platforms"][0]["idempotent"] is False
    assert retry["platforms"][0]["idempotent"] is True
    assert rows == 1


def test_bundle_failure_restores_both_platform_files_and_database(
    app, tmp_path, monkeypatch
):
    _allow_test_signer(monkeypatch)
    baseline = _reserve(app)
    replacement = _reserve(app)
    baseline_desktop = _desktop_artifact(
        tmp_path, baseline, name="baseline-desktop.zip"
    )
    baseline_android = _android_artifact(
        tmp_path, baseline, name="baseline.apk"
    )
    replacement_desktop = _desktop_artifact(
        tmp_path, replacement, name="replacement-desktop.zip"
    )
    replacement_android = _android_artifact(
        tmp_path, replacement, name="replacement.apk"
    )

    with app.app_context():
        db = get_db()
        publish_bundle(
            db,
            baseline.batch_id,
            desktop_path=baseline_desktop,
            android_path=baseline_android,
        )
        releases = Path(app.config["STORAGE_ROOT"]) / "releases"
        baseline_paths = {
            row["platform"]: Path(row["stored_path"])
            for row in db.execute(
                "SELECT platform, stored_path FROM platform_releases "
                "WHERE is_current = 1"
            )
        }
        desktop_before = baseline_paths["desktop"].read_bytes()
        android_before = baseline_paths["android"].read_bytes()

        class FailingConnection:
            def __init__(self, inner):
                self.inner = inner

            @property
            def in_transaction(self):
                return self.inner.in_transaction

            def execute(self, sql, parameters=()):
                if "INSERT INTO platform_releases" in sql and parameters[2] == "android":
                    raise sqlite3.OperationalError("injected android failure")
                return self.inner.execute(sql, parameters)

            def commit(self):
                return self.inner.commit()

            def rollback(self):
                return self.inner.rollback()

        with pytest.raises(sqlite3.OperationalError, match="injected android"):
            publish_bundle(
                FailingConnection(db),
                replacement.batch_id,
                desktop_path=replacement_desktop,
                android_path=replacement_android,
            )
        currents = db.execute(
            "SELECT platform, batch_id FROM platform_releases WHERE is_current = 1"
        ).fetchall()

    assert baseline_paths["desktop"].read_bytes() == desktop_before
    assert baseline_paths["android"].read_bytes() == android_before
    assert {(row["platform"], row["batch_id"]) for row in currents} == {
        ("desktop", baseline.batch_id),
        ("android", baseline.batch_id),
    }


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("bad_zip", "valid ZIP"),
        ("mismatch", "release batch"),
        ("expired", "expired"),
    ],
)
def test_invalid_bundle_does_not_change_current_files_or_rows(
    app, tmp_path, monkeypatch, kind, expected
):
    _allow_test_signer(monkeypatch)
    baseline = _reserve(app)
    candidate = _reserve(app)
    baseline_android = _android_artifact(tmp_path, baseline, name="baseline.apk")
    with app.app_context():
        db = get_db()
        publish_bundle(db, baseline.batch_id, android_path=baseline_android)
        current_path = Path(
            db.execute(
                "SELECT stored_path FROM platform_releases "
                "WHERE platform = 'android' AND is_current = 1"
            ).fetchone()[0]
        )
        before = current_path.read_bytes()
        if kind == "bad_zip":
            source = tmp_path / "invalid.apk"
            source.write_bytes(b"not a zip")
        else:
            source = _android_artifact(
                tmp_path,
                candidate,
                name=f"{kind}.apk",
                **(
                    {"release_batch_id": baseline.batch_id}
                    if kind == "mismatch"
                    else {}
                ),
            )
        if kind == "expired":
            db.execute(
                "UPDATE release_batches SET expires_at = ? WHERE batch_id = ?",
                ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), candidate.batch_id),
            )
            db.commit()
        with pytest.raises(InvalidReleaseBatch, match=expected):
            publish_bundle(db, candidate.batch_id, android_path=source)
        current = db.execute(
            "SELECT batch_id FROM platform_releases "
            "WHERE platform = 'android' AND is_current = 1"
        ).fetchone()[0]

    assert current_path.read_bytes() == before
    assert current == baseline.batch_id


def test_wrong_apk_signer_does_not_change_current(app, tmp_path, monkeypatch):
    reserved = _reserve(app)
    android = _android_artifact(tmp_path, reserved)
    monkeypatch.setattr(
        "water_server.release_batches.verify_apk_signature",
        lambda _path: (_ for _ in ()).throw(InvalidReleaseBatch("wrong signer")),
    )

    with app.app_context():
        with pytest.raises(InvalidReleaseBatch, match="wrong signer"):
            publish_bundle(get_db(), reserved.batch_id, android_path=android)
        count = get_db().execute("SELECT COUNT(*) FROM platform_releases").fetchone()[0]

    assert count == 0


@pytest.mark.parametrize(
    ("completed", "message"),
    [
        (subprocess.CompletedProcess([], 1, "", "bad signature"), "signature verification failed"),
        (subprocess.CompletedProcess([], 0, "Verified", ""), "exactly one signer"),
        (
            subprocess.CompletedProcess(
                [], 0, "Signer #1 certificate SHA-256 digest: 11:22", ""
            ),
            "fingerprint is invalid",
        ),
    ],
)
def test_verify_apk_signature_rejects_invalid_tool_results(
    app, tmp_path, monkeypatch, completed, message
):
    apk = tmp_path / "signed.apk"
    apk.write_bytes(b"apk")
    app.config["ANDROID_SIGNING_CERT_SHA256"] = "aa" * 32
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    with app.app_context(), pytest.raises(InvalidReleaseBatch, match=message):
        verify_apk_signature(apk)


def test_verify_apk_signature_rejects_timeout(app, tmp_path, monkeypatch):
    apk = tmp_path / "signed.apk"
    apk.write_bytes(b"apk")

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("apksigner", 120)

    monkeypatch.setattr(subprocess, "run", timeout)
    with app.app_context(), pytest.raises(InvalidReleaseBatch, match="timed out"):
        verify_apk_signature(apk)


@pytest.mark.parametrize(
    ("client_fixture", "csrf", "status", "payload"),
    [
        ("client", True, 401, {"error": "authentication_required"}),
        ("admin_client", False, 403, {"error": "csrf_failed"}),
    ],
)
def test_bundle_route_requires_auth_and_csrf(
    request, client_fixture, csrf, status, payload
):
    http_client = request.getfixturevalue(client_fixture)
    if not csrf:
        http_client.environ_base.pop("HTTP_X_CSRF_TOKEN")
    response = http_client.post("/admin/releases/bundle")
    assert response.status_code == status
    assert response.get_json() == payload


@pytest.mark.parametrize(
    ("data", "message"),
    [({}, "missing"), ({"batch_id": "missing"}, "unknown release batch")],
)
def test_bundle_route_rejects_missing_files_and_unknown_batch(
    admin_client, data, message
):
    response = admin_client.post("/admin/releases/bundle", data=data)
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "invalid_release"
    assert message in payload["message"]


def test_bundle_route_rejects_duplicate_batch_id(
    admin_client, tmp_path, monkeypatch
):
    _allow_test_signer(monkeypatch)
    reserved = _reserve(admin_client.application)
    android = _android_artifact(tmp_path, reserved)

    with android.open("rb") as stream:
        response = admin_client.post(
            "/admin/releases/bundle",
            data=MultiDict(
                [
                    ("batch_id", reserved.batch_id),
                    ("batch_id", reserved.batch_id),
                    ("android", (stream, "mobile.apk")),
                ]
            ),
        )

    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_release"


def test_bundle_rechecks_expiry_after_artifact_validation(
    app, tmp_path, monkeypatch
):
    import water_server.release_batches as release_batches

    _allow_test_signer(monkeypatch)
    reserved = _reserve(app)
    android = _android_artifact(tmp_path, reserved)
    before_expiry = datetime(2026, 7, 16, tzinfo=timezone.utc)
    after_expiry = before_expiry + timedelta(seconds=2)

    class AdvancingDateTime(datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            value = before_expiry if cls.calls == 1 else after_expiry
            return value if tz is not None else value.replace(tzinfo=None)

    with app.app_context():
        db = get_db()
        db.execute(
            "UPDATE release_batches SET expires_at = ? WHERE batch_id = ?",
            ((before_expiry + timedelta(seconds=1)).isoformat(), reserved.batch_id),
        )
        db.commit()
        monkeypatch.setattr(release_batches, "datetime", AdvancingDateTime)

        with pytest.raises(InvalidReleaseBatch, match="expired"):
            publish_bundle(db, reserved.batch_id, android_path=android)

        assert db.execute("SELECT COUNT(*) FROM platform_releases").fetchone()[0] == 0


def test_desktop_artifact_obeys_existing_upload_size_limit(app, tmp_path):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    app.config["MAX_CONTENT_LENGTH"] = desktop.stat().st_size - 1

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="too large"):
            read_desktop_artifact(desktop, row)


@pytest.mark.parametrize(
    "unsafe_name",
    ["/absolute.bin", "C:/drive-root.bin", "../escape.bin", "..\\escape.bin"],
)
def test_desktop_reader_rejects_unsafe_member_paths(
    app, tmp_path, unsafe_name
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    _append_member(desktop, unsafe_name)

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="unsafe"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_rejects_symlink_member(app, tmp_path):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    _append_member(desktop, "link", mode=stat.S_IFLNK | 0o777)

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="unsafe"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_rejects_encrypted_required_member(
    app, tmp_path, monkeypatch
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    original_infolist = zipfile.ZipFile.infolist

    def encrypted_infolist(archive):
        infos = original_infolist(archive)
        for info in infos:
            if info.filename.endswith("reaction_classifier.h5"):
                info.flag_bits |= 0x1
        return infos

    monkeypatch.setattr(zipfile.ZipFile, "infolist", encrypted_infolist)
    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="unsafe"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_rejects_more_than_20000_members(
    app, tmp_path, monkeypatch
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    original_infolist = zipfile.ZipFile.infolist

    def oversized_infolist(archive):
        infos = original_infolist(archive)
        return infos + [zipfile.ZipInfo(f"extra-{index}") for index in range(20_001)]

    monkeypatch.setattr(zipfile.ZipFile, "infolist", oversized_infolist)
    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="member count"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_rejects_declared_uncompressed_total(app, tmp_path):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    with zipfile.ZipFile(desktop) as archive:
        declared = sum(info.file_size for info in archive.infolist())
    app.config["MAX_RELEASE_UNCOMPRESSED"] = declared - 1

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="uncompressed size"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_counts_actual_streamed_bytes(
    app, tmp_path, monkeypatch
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    with zipfile.ZipFile(desktop) as archive:
        declared = sum(info.file_size for info in archive.infolist())
    app.config["MAX_RELEASE_UNCOMPRESSED"] = declared + 8
    original_open = zipfile.ZipFile.open

    def inflated_open(archive, member, *args, **kwargs):
        filename = member.filename if isinstance(member, zipfile.ZipInfo) else member
        if filename.endswith("reaction_classifier.h5"):
            return io.BytesIO(b"x" * (declared + 9))
        return original_open(archive, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", inflated_open)
    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="uncompressed size"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_bounds_actual_manifest_bytes(
    app, tmp_path, monkeypatch
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    app.config["MAX_RELEASE_UNCOMPRESSED"] = 2 * 1024 * 1024
    original_open = zipfile.ZipFile.open

    def oversized_manifest(archive, member, *args, **kwargs):
        filename = member.filename if isinstance(member, zipfile.ZipInfo) else member
        if filename == "release.json":
            return io.BytesIO(b" " * (1024 * 1024 + 1) + b"{}")
        return original_open(archive, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", oversized_manifest)
    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="manifest is too large"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_rejects_required_member_crc_corruption(app, tmp_path):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    _corrupt_stored_member(desktop, "models/reaction_classifier.h5")

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="invalid ZIP data"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_normalizes_lzma_required_member_corruption(
    app, tmp_path
):
    reserved = _reserve(app)
    desktop = _desktop_with_compressed_classifier(
        tmp_path, reserved, zipfile.ZIP_LZMA
    )
    _corrupt_stored_member(
        desktop,
        "models/reaction_classifier.h5",
        compressed_offset=12,
    )

    with app.app_context():
        db = get_db()
        row = db.execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="invalid ZIP data") as error:
            read_desktop_artifact(desktop, row)
        assert "Corrupt input data" not in str(error.value)
        assert db.execute("SELECT COUNT(*) FROM platform_releases").fetchone()[0] == 0
        current = (
            Path(app.config["STORAGE_ROOT"])
            / "releases"
            / "desktop-latest.zip"
        )
        assert not current.exists()


def test_desktop_reader_rejects_truncated_required_member(
    app, tmp_path, monkeypatch
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    original_open = zipfile.ZipFile.open

    class TruncatedStream(io.BytesIO):
        def read(self, size=-1):
            if self.tell() > 0:
                raise EOFError("truncated member")
            return super().read(1)

    def truncated_open(archive, member, *args, **kwargs):
        filename = member.filename if isinstance(member, zipfile.ZipInfo) else member
        if filename.endswith("yolov8n.pt"):
            return TruncatedStream(b"detector")
        return original_open(archive, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", truncated_open)
    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="invalid ZIP data"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_hides_decompressor_errors(app, tmp_path, monkeypatch):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    original_open = zipfile.ZipFile.open

    class BrokenCompressedStream(io.BytesIO):
        def read(self, _size=-1):
            raise zlib.error("sensitive decompressor detail")

    def broken_open(archive, member, *args, **kwargs):
        filename = member.filename if isinstance(member, zipfile.ZipInfo) else member
        if filename.endswith("reaction_classifier.h5"):
            return BrokenCompressedStream()
        return original_open(archive, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", broken_open)
    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="invalid ZIP data") as error:
            read_desktop_artifact(desktop, row)
    assert "sensitive" not in str(error.value)


def test_android_reader_rejects_source_over_apk_limit(app, tmp_path):
    reserved = _reserve(app)
    android = _android_artifact(tmp_path, reserved)
    app.config["MAX_ANDROID_APK_BYTES"] = android.stat().st_size - 1

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="too large"):
            read_android_artifact(android, row)


def test_bundle_route_cleans_partial_temp_when_filestorage_save_fails(
    admin_client, tmp_path, monkeypatch
):
    from werkzeug.datastructures import FileStorage

    reserved = _reserve(admin_client.application)
    android = _android_artifact(tmp_path, reserved)
    app = admin_client.application
    app.config["PROPAGATE_EXCEPTIONS"] = False
    temp_dir = Path(app.config["STORAGE_ROOT"]) / "temp"
    before = set(temp_dir.iterdir())

    def partial_save(_uploaded, destination, *_args, **_kwargs):
        Path(destination).write_bytes(b"partial")
        raise OSError("injected save failure")

    monkeypatch.setattr(FileStorage, "save", partial_save)
    with android.open("rb") as stream:
        response = admin_client.post(
            "/admin/releases/bundle",
            data={
                "batch_id": reserved.batch_id,
                "android": (stream, "mobile.apk"),
            },
        )

    assert response.status_code == 500
    assert set(temp_dir.iterdir()) == before
    with app.app_context():
        assert get_db().execute("SELECT COUNT(*) FROM platform_releases").fetchone()[0] == 0


def test_bundle_cleans_both_staging_files_when_second_copy_fails(
    app, tmp_path, monkeypatch
):
    import water_server.release_batches as release_batches

    _allow_test_signer(monkeypatch)
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    android = _android_artifact(tmp_path, reserved)
    original_copy = release_batches.shutil.copyfileobj
    calls = 0

    def fail_second_copy(source, target, length=0):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write(source.read(3))
            raise OSError("injected staging copy failure")
        return original_copy(source, target, length=length)

    monkeypatch.setattr(release_batches.shutil, "copyfileobj", fail_second_copy)
    with app.app_context():
        with pytest.raises(OSError, match="staging copy"):
            publish_bundle(
                get_db(),
                reserved.batch_id,
                desktop_path=desktop,
                android_path=android,
            )
        releases = Path(app.config["STORAGE_ROOT"]) / "releases"
        assert list(releases.glob(".*.staging")) == []
        assert (releases / f"{reserved.batch_id}-desktop.zip").is_file()
        assert get_db().execute("SELECT COUNT(*) FROM platform_releases").fetchone()[0] == 0


@pytest.mark.parametrize("platform", ["desktop", "android"])
def test_older_batch_cannot_replace_newer_platform_current(
    app, tmp_path, monkeypatch, platform
):
    _allow_test_signer(monkeypatch)
    older = _reserve(app)
    newer = _reserve(app)
    builder = _desktop_artifact if platform == "desktop" else _android_artifact
    path_argument = "desktop_path" if platform == "desktop" else "android_path"
    newer_artifact = builder(tmp_path, newer, name=f"newer-{platform}.zip")
    older_artifact = builder(tmp_path, older, name=f"older-{platform}.zip")

    with app.app_context():
        db = get_db()
        publish_bundle(db, newer.batch_id, **{path_argument: newer_artifact})
        current_before = db.execute(
            "SELECT release_id, stored_path FROM platform_releases "
            "WHERE platform = ? AND is_current = 1",
            (platform,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="newer generation"):
            publish_bundle(db, older.batch_id, **{path_argument: older_artifact})
        current_after = db.execute(
            "SELECT release_id, stored_path FROM platform_releases "
            "WHERE platform = ? AND is_current = 1",
            (platform,),
        ).fetchone()

    assert tuple(current_after) == tuple(current_before)


def test_partial_batch_late_android_cannot_replace_newer_android(
    app, tmp_path, monkeypatch
):
    _allow_test_signer(monkeypatch)
    older = _reserve(app)
    older_desktop = _desktop_artifact(tmp_path, older, name="older-desktop.zip")
    older_android = _android_artifact(tmp_path, older, name="older-android.apk")
    with app.app_context():
        db = get_db()
        publish_bundle(db, older.batch_id, desktop_path=older_desktop)
    newer = _reserve(app)
    newer_android = _android_artifact(tmp_path, newer, name="newer-android.apk")

    with app.app_context():
        db = get_db()
        publish_bundle(db, newer.batch_id, android_path=newer_android)
        with pytest.raises(InvalidReleaseBatch, match="newer generation"):
            publish_bundle(db, older.batch_id, android_path=older_android)
        rows = db.execute(
            "SELECT platform, release_id FROM platform_releases WHERE is_current = 1"
        ).fetchall()
        older_status = db.execute(
            "SELECT status FROM release_batches WHERE batch_id = ?",
            (older.batch_id,),
        ).fetchone()[0]

    assert {tuple(row) for row in rows} == {
        ("desktop", f"{older.batch_id}-desktop"),
        ("android", f"{newer.batch_id}-android"),
    }
    assert older_status == "partial"


def test_bundle_stores_immutable_release_file_and_fsyncs(
    app, tmp_path, monkeypatch
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    fsynced = []
    real_fsync = os.fsync

    def record_fsync(descriptor):
        fsynced.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr("water_server.release_batches.os.fsync", record_fsync)
    with app.app_context():
        result = publish_bundle(get_db(), reserved.batch_id, desktop_path=desktop)
        row = get_db().execute(
            "SELECT stored_path FROM platform_releases WHERE is_current = 1"
        ).fetchone()

    expected = (
        Path(app.config["STORAGE_ROOT"])
        / "releases"
        / f"{reserved.batch_id}-desktop.zip"
    )
    assert Path(row["stored_path"]) == expected
    assert expected.read_bytes() == desktop.read_bytes()
    assert not (expected.parent / "desktop-latest.zip").exists()
    assert result["platforms"][0]["release_id"] == f"{reserved.batch_id}-desktop"
    assert fsynced


@pytest.mark.parametrize(
    "conflicting_name",
    ["models/yolov8n.pt", "MODELS/YOLOV8N.PT"],
)
def test_desktop_reader_rejects_duplicate_or_casefold_path(
    app, tmp_path, conflicting_name
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _append_member(desktop, conflicting_name, b"conflict")
    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="duplicate.*path"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_rejects_unicode_normalization_collision(app, tmp_path):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    _append_member(desktop, "models/caf\u00e9.bin")
    _append_member(desktop, "models/cafe\u0301.bin")
    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="duplicate.*path"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_rejects_duplicate_json_key(app, tmp_path):
    reserved = _reserve(app)
    desktop = tmp_path / "duplicate-json-key.zip"
    manifest = json.dumps(_manifest(reserved, "desktop"))
    manifest = manifest[:-1] + ', "app_version_name": "duplicate"}'
    with zipfile.ZipFile(desktop, "w") as archive:
        archive.writestr("release.json", manifest)
        archive.writestr("WaterApp.exe", b"exe")
        archive.writestr("reaction_classifier.h5", b"classifier")
        archive.writestr("yolov8n.pt", b"detector")
    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="duplicate JSON key"):
            read_desktop_artifact(desktop, row)


def test_apk_signature_rejects_multiple_signers(app, tmp_path, monkeypatch):
    apk = tmp_path / "signed.apk"
    apk.write_bytes(b"apk")
    digest = "aa" * 32
    completed = subprocess.CompletedProcess(
        [],
        0,
        "\n".join(
            [
                f"Signer #1 certificate SHA-256 digest: {digest}",
                f"Signer #2 certificate SHA-256 digest: {'bb' * 32}",
            ]
        ),
        "",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)
    with app.app_context(), pytest.raises(
        InvalidReleaseBatch, match="exactly one signer"
    ):
        verify_apk_signature(apk)


@pytest.mark.parametrize(
    "configured",
    [
        "aa",
        "a" * 63,
        "a" * 65,
        "gg" * 32,
        "aa-bb" + ":cc" * 30,
        " " + "aa" * 32,
    ],
)
def test_apk_signature_rejects_malformed_configured_digest(
    app, tmp_path, monkeypatch, configured
):
    apk = tmp_path / "signed.apk"
    apk.write_bytes(b"apk")
    digest = "aa" * 32
    app.config["ANDROID_SIGNING_CERT_SHA256"] = configured
    completed = subprocess.CompletedProcess(
        [], 0, f"Signer #1 certificate SHA-256 digest: {digest}", ""
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)
    with app.app_context(), pytest.raises(
        InvalidReleaseBatch, match="configured.*fingerprint"
    ):
        verify_apk_signature(apk)


def test_apk_signature_requires_pin_outside_testing(app, tmp_path, monkeypatch):
    apk = tmp_path / "signed.apk"
    apk.write_bytes(b"apk")
    digest = "aa" * 32
    app.config["TESTING"] = False
    app.config["ANDROID_SIGNING_CERT_SHA256"] = ""
    completed = subprocess.CompletedProcess(
        [], 0, f"Signer #1 certificate SHA-256 digest: {digest}", ""
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)
    with app.app_context(), pytest.raises(InvalidReleaseBatch, match="required"):
        verify_apk_signature(apk)


def test_immutable_orphan_same_hash_can_be_retried(
    app, tmp_path
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)

    def fail_before_commit(_activated):
        raise sqlite3.OperationalError("injected callback failure")

    with app.app_context():
        db = get_db()
        with pytest.raises(sqlite3.OperationalError, match="callback"):
            publish_bundle(
                db,
                reserved.batch_id,
                desktop_path=desktop,
                before_commit=fail_before_commit,
            )
        immutable = (
            Path(app.config["STORAGE_ROOT"])
            / "releases"
            / f"{reserved.batch_id}-desktop.zip"
        )
        assert immutable.is_file()
        assert db.execute("SELECT COUNT(*) FROM platform_releases").fetchone()[0] == 0

        retried = publish_bundle(
            db, reserved.batch_id, desktop_path=desktop
        )

    assert retried["platforms"][0]["idempotent"] is False


def test_immutable_path_with_different_hash_is_rejected(app, tmp_path):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    immutable = (
        Path(app.config["STORAGE_ROOT"])
        / "releases"
        / f"{reserved.batch_id}-desktop.zip"
    )
    immutable.write_bytes(b"different artifact")

    with app.app_context():
        with pytest.raises(InvalidReleaseBatch, match="different artifact"):
            publish_bundle(
                get_db(), reserved.batch_id, desktop_path=desktop
            )
        assert get_db().execute("SELECT COUNT(*) FROM platform_releases").fetchone()[0] == 0


def test_published_immutable_android_is_made_read_only(
    app, tmp_path, monkeypatch
):
    reserved = _reserve(app)
    android = _android_artifact(tmp_path, reserved)
    _allow_test_signer(monkeypatch)
    chmod_calls = []
    monkeypatch.setattr(
        release_batches_module,
        "_POSIX_RELEASE_PERMISSIONS",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        release_batches_module.os,
        "chmod",
        lambda path, mode: chmod_calls.append((Path(path), mode)),
    )

    with app.app_context():
        publish_bundle(get_db(), reserved.batch_id, android_path=android)
        stored_path = Path(
            get_db().execute(
                "SELECT stored_path FROM platform_releases "
                "WHERE release_id = ?",
                (reserved.android_release_id,),
            ).fetchone()[0]
        )

    assert chmod_calls == [(stored_path, 0o444)]


def test_idempotent_retry_normalizes_legacy_stored_path(app, tmp_path):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    legacy_pointer = (
        Path(app.config["STORAGE_ROOT"]) / "releases" / "desktop-latest.zip"
    )
    legacy_pointer.write_bytes(desktop.read_bytes())

    with app.app_context():
        db = get_db()
        first = publish_bundle(db, reserved.batch_id, desktop_path=desktop)
        db.execute(
            "UPDATE platform_releases SET stored_path = ? WHERE release_id = ?",
            (str(legacy_pointer), first["platforms"][0]["release_id"]),
        )
        db.commit()

        retry = publish_bundle(db, reserved.batch_id, desktop_path=desktop)
        stored_path = db.execute(
            "SELECT stored_path FROM platform_releases WHERE release_id = ?",
            (first["platforms"][0]["release_id"],),
        ).fetchone()[0]

    expected = legacy_pointer.parent / f"{reserved.batch_id}-desktop.zip"
    assert retry["platforms"][0]["idempotent"] is True
    assert Path(stored_path) == expected


@pytest.mark.parametrize(
    "members",
    [
        ("models/x.bin", "models/./x.bin"),
        ("models//x.bin",),
        ("Model.pt", "model.pt. "),
    ],
)
def test_desktop_reader_rejects_windows_equivalent_or_empty_segments(
    app, tmp_path, members
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    for member in members:
        _append_member(desktop, member)

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="Windows|segment|path"):
            read_desktop_artifact(desktop, row)


@pytest.mark.parametrize(
    "member",
    ["CON.txt", "models/aux", "models/NUL. ", "com1.bin", "LPT9"],
)
def test_desktop_reader_rejects_windows_device_names(
    app, tmp_path, member
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    _append_member(desktop, member)

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="reserved device"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_rejects_windows_ads(app, tmp_path):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    _append_member(desktop, "models/file:stream")

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="alternate data stream"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_allows_safe_directory_entry(app, tmp_path):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    _append_member(desktop, "safe-directory/", b"")

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        artifact = read_desktop_artifact(desktop, row)

    assert artifact.release_id == f"{reserved.batch_id}-desktop"


def test_android_reader_does_not_apply_windows_device_rules(app, tmp_path):
    reserved = _reserve(app)
    android = _android_artifact(tmp_path, reserved)
    _append_member(android, "assets/CON.txt")

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        artifact = read_android_artifact(android, row)

    assert artifact.release_id == f"{reserved.batch_id}-android"


def test_bundle_route_rejects_windows_device_path_without_changing_current(
    admin_client, tmp_path
):
    app = admin_client.application
    baseline = _reserve(app)
    baseline_desktop = _desktop_artifact(tmp_path, baseline, name="baseline.zip")
    with app.app_context():
        publish_bundle(
            get_db(), baseline.batch_id, desktop_path=baseline_desktop
        )
    candidate = _reserve(app)
    candidate_desktop = _desktop_artifact(
        tmp_path, candidate, name="windows-device.zip"
    )
    _append_member(candidate_desktop, "models/CON.txt")

    with candidate_desktop.open("rb") as stream:
        response = admin_client.post(
            "/admin/releases/bundle",
            data={
                "batch_id": candidate.batch_id,
                "desktop": (stream, "desktop.zip"),
            },
        )

    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_release"
    with app.app_context():
        current = get_db().execute(
            "SELECT release_id FROM platform_releases "
            "WHERE platform = 'desktop' AND is_current = 1"
        ).fetchone()[0]
    assert current == f"{baseline.batch_id}-desktop"


@pytest.mark.parametrize("character", ['<', '>', ':', '"', '|', '?', '*'])
def test_desktop_reader_rejects_windows_invalid_filename_characters(
    app, tmp_path, character
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    _append_member(desktop, f"models/bad{character}name.bin")

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="Windows"):
            read_desktop_artifact(desktop, row)


@pytest.mark.parametrize("character", ["\x01", "\x1f"])
def test_desktop_reader_rejects_windows_control_characters(
    app, tmp_path, character
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    _append_member(desktop, f"models/bad{character}name.bin")

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="Windows"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_rejects_windows_nul_control_character(app, tmp_path):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    original = b"models/badXname.bin"
    replacement = b"models/bad\x00name.bin"
    _append_member(desktop, original.decode("ascii"))
    raw = desktop.read_bytes()
    assert raw.count(original) == 2
    desktop.write_bytes(raw.replace(original, replacement))

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="Windows"):
            read_desktop_artifact(desktop, row)


@pytest.mark.parametrize(
    "member",
    [
        "CONIN$",
        "models/conout$.log",
        "COM¹",
        "models/com².txt",
        "LPT³.data",
        "CON .txt",
        "NUL...txt",
    ],
)
def test_desktop_reader_rejects_additional_windows_device_names(
    app, tmp_path, member
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    _append_member(desktop, member)

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        with pytest.raises(InvalidReleaseBatch, match="reserved device"):
            read_desktop_artifact(desktop, row)


def test_desktop_reader_allows_com10_and_safe_unicode_names(app, tmp_path):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    _append_member(desktop, "models/COM10.bin")
    _append_member(desktop, "模型/安全な名前.bin")

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        artifact = read_desktop_artifact(desktop, row)

    assert artifact.release_id == f"{reserved.batch_id}-desktop"


def test_android_reader_allows_windows_invalid_member_names(app, tmp_path):
    reserved = _reserve(app)
    android = _android_artifact(tmp_path, reserved)
    _append_member(android, "assets/CONIN$.txt")
    _append_member(android, "assets/bad?.txt")
    _append_member(android, "assets/control\x01.bin")

    with app.app_context():
        row = get_db().execute(
            "SELECT * FROM release_batches WHERE batch_id = ?",
            (reserved.batch_id,),
        ).fetchone()
        artifact = read_android_artifact(android, row)

    assert artifact.release_id == f"{reserved.batch_id}-android"


@pytest.mark.parametrize(
    "member",
    ["models/bad?.txt", "models/bad\x01.txt", "CONIN$.txt", "CON .txt", "COM¹"],
)
def test_bundle_route_rejects_invalid_windows_member_without_changing_current(
    admin_client, tmp_path, member
):
    app = admin_client.application
    baseline = _reserve(app)
    baseline_desktop = _desktop_artifact(tmp_path, baseline, name="baseline.zip")
    with app.app_context():
        publish_bundle(
            get_db(), baseline.batch_id, desktop_path=baseline_desktop
        )
    candidate = _reserve(app)
    candidate_desktop = _desktop_artifact(
        tmp_path, candidate, name="invalid-windows-name.zip"
    )
    _append_member(candidate_desktop, member)

    with candidate_desktop.open("rb") as stream:
        response = admin_client.post(
            "/admin/releases/bundle",
            data={
                "batch_id": candidate.batch_id,
                "desktop": (stream, "desktop.zip"),
            },
        )

    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_release"
    with app.app_context():
        current = get_db().execute(
            "SELECT release_id FROM platform_releases "
            "WHERE platform = 'desktop' AND is_current = 1"
        ).fetchone()[0]
    assert current == f"{baseline.batch_id}-desktop"


def test_immutable_directory_is_flushed_before_database_transaction(
    app, tmp_path, monkeypatch
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    events = []
    real_fsync = os.fsync
    real_link = os.link

    def record_fsync(descriptor):
        events.append("file fsync")
        return real_fsync(descriptor)

    def record_link(source, destination):
        result = real_link(source, destination)
        events.append("link")
        return result

    def record_directory_flush(path):
        assert Path(path).name == "releases"
        events.append("directory fsync")
        return True

    class RecordingConnection:
        def __init__(self, inner):
            self.inner = inner

        @property
        def in_transaction(self):
            return self.inner.in_transaction

        def execute(self, statement, parameters=()):
            if statement.strip().startswith("BEGIN IMMEDIATE"):
                events.append("BEGIN IMMEDIATE")
            return self.inner.execute(statement, parameters)

        def commit(self):
            return self.inner.commit()

        def rollback(self):
            return self.inner.rollback()

    monkeypatch.setattr(release_batches_module.os, "fsync", record_fsync)
    monkeypatch.setattr(release_batches_module.os, "link", record_link)
    monkeypatch.setattr(
        release_batches_module,
        "_flush_release_directory",
        record_directory_flush,
        raising=False,
    )

    with app.app_context():
        publish_bundle(
            RecordingConnection(get_db()),
            reserved.batch_id,
            desktop_path=desktop,
        )

    assert events.index("file fsync") < events.index("link")
    assert events.index("link") < events.index("directory fsync")
    assert events.index("directory fsync") < events.index("BEGIN IMMEDIATE")


def test_directory_flush_failure_does_not_commit_release_pointer(
    app, tmp_path, monkeypatch
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)

    def fail_directory_flush(_path):
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(
        release_batches_module,
        "_flush_release_directory",
        fail_directory_flush,
        raising=False,
    )

    with app.app_context():
        db = get_db()
        before = tuple(
            db.execute(
                "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
            ).fetchone()
        )
        with pytest.raises(OSError, match="directory fsync"):
            publish_bundle(db, reserved.batch_id, desktop_path=desktop)
        after = tuple(
            db.execute(
                "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
            ).fetchone()
        )
        count = db.execute("SELECT COUNT(*) FROM platform_releases").fetchone()[0]

    assert after == before
    assert count == 0


def test_posix_release_directory_flush_opens_and_fsyncs_directory(
    app, tmp_path, monkeypatch
):
    releases = tmp_path / "releases"
    releases.mkdir()
    descriptor = 1729
    calls = []

    with monkeypatch.context() as patch:
        patch.setattr(release_batches_module.os, "name", "posix")
        patch.setattr(
            release_batches_module.os,
            "O_DIRECTORY",
            0x10000,
            raising=False,
        )
        patch.setattr(
            release_batches_module.os,
            "open",
            lambda path, flags: calls.append(("open", str(path), flags))
            or descriptor,
        )
        patch.setattr(
            release_batches_module.os,
            "fsync",
            lambda fd: calls.append(("fsync", fd)),
        )
        patch.setattr(
            release_batches_module.os,
            "close",
            lambda fd: calls.append(("close", fd)),
        )
        with app.app_context():
            supported = release_batches_module._flush_release_directory(releases)

    assert supported is True
    assert calls == [
        ("open", str(releases), os.O_RDONLY | 0x10000),
        ("fsync", descriptor),
        ("close", descriptor),
    ]


def test_windows_release_directory_flush_reports_unsupported(
    app, tmp_path, monkeypatch
):
    if os.name != "nt":
        pytest.skip("Windows fallback semantics")
    releases = tmp_path / "releases"
    releases.mkdir()
    monkeypatch.setattr(
        release_batches_module.os,
        "open",
        lambda *_args: pytest.fail("Windows fallback must not claim directory fsync"),
    )
    monkeypatch.setattr(
        release_batches_module.os,
        "fsync",
        lambda *_args: pytest.fail("Windows fallback must not fsync a directory"),
    )

    with app.app_context():
        supported = release_batches_module._flush_release_directory(releases)

    assert supported is False


def _insert_gc_batch(db, batch_id, model_generation, status, now, expires_at):
    db.execute(
        """
        INSERT INTO release_batches (
            batch_id, model_generation, dataset_generation, status,
            reserved_at, expires_at, published_at
        ) VALUES (?, ?, 1, ?, ?, ?, NULL)
        """,
        (
            batch_id,
            model_generation,
            status,
            (now - timedelta(days=3)).isoformat(),
            expires_at.isoformat(),
        ),
    )
    db.commit()


def _write_old_release(path, now):
    path.write_bytes(b"orphan")
    old_timestamp = (now - timedelta(hours=48)).timestamp()
    os.utime(path, (old_timestamp, old_timestamp))


@pytest.mark.parametrize(
    ("status", "expires_delta"),
    [("expired", timedelta(hours=1)), ("reserved", -timedelta(hours=1))],
)
def test_cleanup_deletes_old_unreferenced_expired_release(
    app, monkeypatch, status, expires_delta
):
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    batch_id = "a" * 32
    releases = Path(app.config["STORAGE_ROOT"]) / "releases"
    orphan = releases / f"{batch_id}-desktop.zip"
    _write_old_release(orphan, now)
    flushed = []
    monkeypatch.setattr(
        release_batches_module,
        "_flush_release_directory",
        lambda path: flushed.append(Path(path)) or True,
    )

    with app.app_context():
        db = get_db()
        _insert_gc_batch(
            db,
            batch_id,
            2,
            status,
            now,
            now + expires_delta,
        )
        deleted = release_batches_module.cleanup_orphan_release_files(db, now)

    assert deleted == 1
    assert not orphan.exists()
    assert flushed == [releases]


def test_cleanup_never_deletes_referenced_release(app):
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    batch_id = "b" * 32
    releases = Path(app.config["STORAGE_ROOT"]) / "releases"
    artifact = releases / f"{batch_id}-desktop.zip"
    _write_old_release(artifact, now)

    with app.app_context():
        db = get_db()
        _insert_gc_batch(
            db,
            batch_id,
            2,
            "expired",
            now,
            now - timedelta(hours=1),
        )
        db.execute(
            """
            INSERT INTO platform_releases (
                release_id, batch_id, platform, version_code, version_name,
                original_filename, stored_path, sha256, size_bytes,
                uploaded_at, is_current
            ) VALUES (?, ?, 'desktop', 2, '2', 'release.zip', ?, 'sha', 6, ?, 0)
            """,
            (
                f"{batch_id}-desktop",
                batch_id,
                str(artifact),
                now.isoformat(),
            ),
        )
        db.commit()
        deleted = release_batches_module.cleanup_orphan_release_files(db, now)

    assert deleted == 0
    assert artifact.is_file()


@pytest.mark.parametrize("status", ["reserved", "partial"])
def test_cleanup_keeps_unexpired_publishable_batch_artifact(app, status):
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    batch_id = ("c" if status == "reserved" else "d") * 32
    releases = Path(app.config["STORAGE_ROOT"]) / "releases"
    artifact = releases / f"{batch_id}-desktop.zip"
    _write_old_release(artifact, now)

    with app.app_context():
        db = get_db()
        _insert_gc_batch(
            db,
            batch_id,
            2,
            status,
            now,
            now + timedelta(hours=1),
        )
        deleted = release_batches_module.cleanup_orphan_release_files(db, now)

    assert deleted == 0
    assert artifact.is_file()


def test_cleanup_ignores_symlinks_directories_and_non_release_names(app):
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    batch_id = "e" * 32
    releases = Path(app.config["STORAGE_ROOT"]) / "releases"
    target = releases.parent / "outside-target.zip"
    target.write_bytes(b"outside")
    symlink = releases / f"{batch_id}-desktop.zip"
    try:
        symlink.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    directory = releases / f"{'f' * 32}-android.apk"
    directory.mkdir()
    unrelated = releases / "manual-backup.zip"
    _write_old_release(unrelated, now)

    with app.app_context():
        db = get_db()
        _insert_gc_batch(
            db,
            batch_id,
            2,
            "expired",
            now,
            now - timedelta(hours=1),
        )
        deleted = release_batches_module.cleanup_orphan_release_files(db, now)

    assert deleted == 0
    assert symlink.is_symlink()
    assert directory.is_dir()
    assert unrelated.is_file()
    assert target.is_file()


def test_cleanup_limits_release_directory_scan(app, monkeypatch):
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    releases = Path(app.config["STORAGE_ROOT"]) / "releases"
    artifacts = []

    with app.app_context():
        db = get_db()
        for index, letter in enumerate(("1", "2", "3"), start=2):
            batch_id = letter * 32
            _insert_gc_batch(
                db,
                batch_id,
                index,
                "expired",
                now,
                now - timedelta(hours=1),
            )
            artifact = releases / f"{batch_id}-desktop.zip"
            _write_old_release(artifact, now)
            artifacts.append(artifact)
        monkeypatch.setattr(
            release_batches_module,
            "_MAX_ORPHAN_SCAN_ENTRIES",
            2,
            raising=False,
        )
        monkeypatch.setattr(
            release_batches_module,
            "_MAX_ORPHAN_DELETIONS",
            2,
            raising=False,
        )
        deleted = release_batches_module.cleanup_orphan_release_files(db, now)

    assert deleted == 2
    assert sum(path.exists() for path in artifacts) == 1


@pytest.mark.parametrize(
    "grace_hours",
    ["invalid", "0", 0, -1, True, False, 1.5, "+2", " 2", "2.0"],
)
def test_cleanup_rejects_invalid_orphan_grace_hours(app, grace_hours):
    app.config["RELEASE_ORPHAN_GRACE_HOURS"] = grace_hours

    with app.app_context(), pytest.raises(ValueError, match="positive integer"):
        release_batches_module.cleanup_orphan_release_files(get_db())


def test_publish_runs_orphan_cleanup_nonfatally_before_and_after(
    app, tmp_path, monkeypatch
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    calls = []

    def fail_cleanup(_db, now=None):
        calls.append(now)
        raise OSError("injected orphan cleanup failure")

    monkeypatch.setattr(
        release_batches_module,
        "cleanup_orphan_release_files",
        fail_cleanup,
        raising=False,
    )

    with app.app_context():
        result = publish_bundle(get_db(), reserved.batch_id, desktop_path=desktop)

    assert result["platforms"][0]["release_id"] == reserved.desktop_release_id
    assert len(calls) == 2


def test_cleanup_interleaved_before_reserved_batch_commit_keeps_artifact(
    app, tmp_path, monkeypatch
):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    artifact_ready = Event()
    allow_publish = Event()
    errors = []
    real_create = release_batches_module._create_immutable_artifact

    def block_after_create(storage, platform, artifact, expected_sha256):
        result = real_create(storage, platform, artifact, expected_sha256)
        old_timestamp = (
            datetime.now(timezone.utc) - timedelta(hours=48)
        ).timestamp()
        os.utime(result, (old_timestamp, old_timestamp))
        artifact_ready.set()
        assert allow_publish.wait(timeout=5)
        return result

    monkeypatch.setattr(
        release_batches_module,
        "_create_immutable_artifact",
        block_after_create,
    )

    def publish_in_thread():
        try:
            with app.app_context():
                publish_bundle(
                    get_db(), reserved.batch_id, desktop_path=desktop
                )
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=publish_in_thread)
    worker.start()
    try:
        assert artifact_ready.wait(timeout=5)
        with app.app_context():
            deleted = release_batches_module.cleanup_orphan_release_files(
                get_db(), datetime.now(timezone.utc)
            )
        immutable = (
            Path(app.config["STORAGE_ROOT"])
            / "releases"
            / f"{reserved.desktop_release_id}.zip"
        )
        assert deleted == 0
        assert immutable.is_file()
    finally:
        allow_publish.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert errors == []


def test_bundle_route_records_sanitized_original_upload_filename(
    admin_client, tmp_path
):
    app = admin_client.application
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    original_name = "..\\private/" + ("水" * 300) + ".zip"

    with desktop.open("rb") as stream:
        response = admin_client.post(
            "/admin/releases/bundle",
            data={
                "batch_id": reserved.batch_id,
                "desktop": (stream, original_name),
            },
        )

    assert response.status_code == 201
    with app.app_context():
        row = get_db().execute(
            "SELECT original_filename, stored_path FROM platform_releases "
            "WHERE release_id = ?",
            (reserved.desktop_release_id,),
        ).fetchone()

    assert row["original_filename"] == ("水" * 251) + ".zip"
    assert Path(row["stored_path"]).name == f"{reserved.desktop_release_id}.zip"


def test_cleanup_never_deletes_published_batch_after_expiry(app):
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    batch_id = "9" * 32
    releases = Path(app.config["STORAGE_ROOT"]) / "releases"
    artifact = releases / f"{batch_id}-desktop.zip"
    _write_old_release(artifact, now)

    with app.app_context():
        db = get_db()
        _insert_gc_batch(
            db,
            batch_id,
            2,
            "published",
            now,
            now - timedelta(days=1),
        )
        deleted = release_batches_module.cleanup_orphan_release_files(db, now)

    assert deleted == 0
    assert artifact.is_file()


def test_cleanup_protects_normalized_stored_path_reference(app):
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    batch_id = "8" * 32
    releases = Path(app.config["STORAGE_ROOT"]) / "releases"
    artifact = releases / f"{batch_id}-desktop.zip"
    _write_old_release(artifact, now)
    normalized_alias = releases / "missing-directory" / ".." / artifact.name

    with app.app_context():
        db = get_db()
        _insert_gc_batch(
            db,
            batch_id,
            2,
            "expired",
            now,
            now - timedelta(days=1),
        )
        db.execute(
            """
            INSERT INTO platform_releases (
                release_id, batch_id, platform, version_code, version_name,
                original_filename, stored_path, sha256, size_bytes,
                uploaded_at, is_current
            ) VALUES (?, ?, 'desktop', 2, '2', 'release.zip', ?, 'sha', 6, ?, 0)
            """,
            (
                f"{batch_id}-desktop",
                batch_id,
                str(normalized_alias),
                now.isoformat(),
            ),
        )
        db.commit()
        deleted = release_batches_module.cleanup_orphan_release_files(db, now)

    assert deleted == 0
    assert artifact.is_file()


class _ReplacingReleaseConnection:
    def __init__(self, inner, artifact, replacement_bytes, old_timestamp):
        self.inner = inner
        self.artifact = artifact
        self.replacement_bytes = replacement_bytes
        self.old_timestamp = old_timestamp
        self.status_reads = 0
        self.replaced = False
        self.events = []
        self.identities = []

    @property
    def in_transaction(self):
        return self.inner.in_transaction

    def _replace_artifact(self):
        if self.replaced:
            return
        before = self.artifact.lstat()
        replacement = self.artifact.with_name(f".{self.artifact.name}.replacement")
        replacement.write_bytes(self.replacement_bytes)
        os.utime(replacement, (self.old_timestamp, self.old_timestamp))
        replacement_stat = replacement.lstat()
        os.replace(replacement, self.artifact)
        self.identities = [
            (before.st_dev, before.st_ino),
            (replacement_stat.st_dev, replacement_stat.st_ino),
        ]
        assert self.identities[0] != self.identities[1]
        self.replaced = True

    def execute(self, statement, parameters=()):
        normalized = " ".join(statement.split())
        if normalized.startswith("BEGIN IMMEDIATE"):
            self.events.append("begin")
            self._replace_artifact()
        elif (
            normalized.startswith("SELECT")
            and "FROM release_batches" in normalized
            and "WHERE batch_id = ?" in normalized
        ):
            self.status_reads += 1
            if self.status_reads == 2:
                self._replace_artifact()
        return self.inner.execute(statement, parameters)

    def commit(self):
        self.events.append("commit")
        return self.inner.commit()

    def rollback(self):
        self.events.append("rollback")
        return self.inner.rollback()


def test_cleanup_skips_recreated_inode_then_release_can_publish(app, tmp_path):
    reserved = _reserve(app)
    desktop = _desktop_artifact(tmp_path, reserved)
    now = datetime.now(timezone.utc)
    old_timestamp = (now - timedelta(hours=48)).timestamp()
    releases = Path(app.config["STORAGE_ROOT"]) / "releases"
    artifact = releases / f"{reserved.desktop_release_id}.zip"
    artifact.write_bytes(b"old orphan")
    os.utime(artifact, (old_timestamp, old_timestamp))

    with app.app_context():
        db = get_db()
        db.execute(
            "UPDATE release_batches SET status = 'expired' WHERE batch_id = ?",
            (reserved.batch_id,),
        )
        db.commit()
        controlled = _ReplacingReleaseConnection(
            db,
            artifact,
            desktop.read_bytes(),
            old_timestamp,
        )

        deleted = release_batches_module.cleanup_orphan_release_files(
            controlled, now
        )

        assert deleted == 0
        assert controlled.replaced is True
        assert controlled.events == ["begin", "commit"]
        assert artifact.read_bytes() == desktop.read_bytes()

        db.execute(
            "UPDATE release_batches SET status = 'reserved', expires_at = ? "
            "WHERE batch_id = ?",
            ((now + timedelta(hours=1)).isoformat(), reserved.batch_id),
        )
        db.commit()
        result = publish_bundle(
            db,
            reserved.batch_id,
            desktop_path=desktop,
        )

    assert result["platforms"][0]["release_id"] == reserved.desktop_release_id
    assert artifact.is_file()


def test_cleanup_flush_failure_rolls_back_coordination_transaction(
    app, monkeypatch
):
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    batch_id = "7" * 32
    releases = Path(app.config["STORAGE_ROOT"]) / "releases"
    artifact = releases / f"{batch_id}-desktop.zip"
    _write_old_release(artifact, now)

    def fail_flush(_storage):
        raise OSError("injected GC directory flush failure")

    monkeypatch.setattr(
        release_batches_module,
        "_flush_release_directory",
        fail_flush,
    )

    with app.app_context():
        db = get_db()
        _insert_gc_batch(
            db,
            batch_id,
            2,
            "expired",
            now,
            now - timedelta(days=1),
        )

        class RecordingConnection:
            @property
            def in_transaction(self):
                return db.in_transaction

            def execute(self, statement, parameters=()):
                if statement.strip().startswith("BEGIN IMMEDIATE"):
                    events.append("begin")
                return db.execute(statement, parameters)

            def commit(self):
                events.append("commit")
                return db.commit()

            def rollback(self):
                events.append("rollback")
                return db.rollback()

        events = []
        with pytest.raises(OSError, match="GC directory flush"):
            release_batches_module.cleanup_orphan_release_files(
                RecordingConnection(), now
            )
        in_transaction = db.in_transaction

    assert events == ["begin", "rollback"]
    assert in_transaction is False
    assert not artifact.exists()


def test_failed_legacy_cleanup_skips_recreated_inode_under_write_lock(app):
    now = datetime(2026, 7, 17, tzinfo=timezone.utc)
    batch_id = "6" * 32
    releases = Path(app.config["STORAGE_ROOT"]) / "releases"
    artifact = releases / f"{batch_id}-desktop.zip"
    old_timestamp = (now - timedelta(hours=48)).timestamp()
    artifact.write_bytes(b"old legacy orphan")
    os.utime(artifact, (old_timestamp, old_timestamp))

    with app.app_context():
        db = get_db()
        _insert_gc_batch(
            db,
            batch_id,
            2,
            "expired",
            now,
            now - timedelta(days=1),
        )
        controlled = _ReplacingReleaseConnection(
            db,
            artifact,
            b"new legacy artifact",
            old_timestamp,
        )

        deleted = release_batches_module.cleanup_failed_legacy_release_file(
            controlled, batch_id
        )

    assert deleted is False
    assert controlled.replaced is True
    assert controlled.events == ["begin", "commit"]
    assert artifact.read_bytes() == b"new legacy artifact"
