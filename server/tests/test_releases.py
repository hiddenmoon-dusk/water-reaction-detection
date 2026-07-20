import io
import json
import os
import sqlite3
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread, current_thread

import pytest
import water_server.release_batches as release_batches_module

from water_server.db import get_db
from water_server.release_batches import reserve_batch
from water_server.releases import InvalidRelease, publish_desktop_release


def release_zip(
    tmp_path,
    *,
    filename="release.zip",
    exe_data=b"exe-data",
    include_classifier=True,
    include_detector=True,
):
    path = tmp_path / filename
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("水体反应管检测系统.exe", exe_data)
        if include_classifier:
            archive.writestr("reaction_classifier.h5", b"classifier-data")
        if include_detector:
            archive.writestr("yolov8n.pt", b"detector-data")
    return path


def upload_release(client, path):
    with path.open("rb") as stream:
        return client.post(
            "/admin/releases/desktop",
            data={"file": (stream, path.name)},
            content_type="multipart/form-data",
        )


def test_release_missing_model_keeps_current_release(admin_client, tmp_path, db):
    bad = release_zip(tmp_path, include_detector=False)
    before = db.execute(
        "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
    ).fetchone()

    response = upload_release(admin_client, bad)

    after = db.execute(
        "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
    ).fetchone()
    assert response.status_code == 400
    assert tuple(after) == tuple(before)


def test_invalid_legacy_release_expires_its_reserved_generation(tmp_path, db):
    invalid = release_zip(tmp_path, include_detector=False)

    with pytest.raises(InvalidRelease):
        publish_desktop_release(invalid, db)

    failed_batch = db.execute(
        "SELECT model_generation, status FROM release_batches"
    ).fetchone()
    next_batch = reserve_batch(
        db,
        datetime(2026, 7, 16, tzinfo=timezone.utc),
    )

    assert tuple(failed_batch) == (2, "expired")
    assert next_batch.model_generation == 3


def test_invalid_release_keeps_existing_desktop_archive(
    admin_client, app, tmp_path
):
    valid = release_zip(tmp_path)
    published = upload_release(admin_client, valid)
    assert published.status_code == 201
    with app.app_context():
        final_path = Path(
            get_db().execute(
                "SELECT stored_path FROM platform_releases "
                "WHERE platform = 'desktop' AND is_current = 1"
            ).fetchone()[0]
        )
    original_archive = final_path.read_bytes()

    invalid = release_zip(tmp_path, include_detector=False)
    rejected = upload_release(admin_client, invalid)

    assert rejected.status_code == 400
    assert final_path.read_bytes() == original_archive


def test_database_failure_restores_existing_desktop_archive(app, tmp_path, db):
    first_archive = release_zip(tmp_path)
    first_release = publish_desktop_release(first_archive, db)
    original_archive = first_release["path"].read_bytes()
    original_state = tuple(
        db.execute(
            "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
        ).fetchone()
    )

    class FailingConnection:
        @property
        def in_transaction(self):
            return db.in_transaction

        def execute(self, statement, parameters=()):
            if "INSERT INTO platform_releases" in statement:
                raise sqlite3.OperationalError("injected release write failure")
            return db.execute(statement, parameters)

        def commit(self):
            return db.commit()

        def rollback(self):
            return db.rollback()

    second_archive = release_zip(tmp_path)
    with pytest.raises(sqlite3.OperationalError, match="injected"):
        publish_desktop_release(second_archive, FailingConnection())

    restored_state = tuple(
        db.execute(
            "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
        ).fetchone()
    )
    assert not db.in_transaction
    assert restored_state == original_state
    assert first_release["path"].read_bytes() == original_archive


def test_legacy_activation_uses_shared_immutable_hardlink_path(
    app, tmp_path, db, monkeypatch
):
    replacement = release_zip(
        tmp_path,
        filename="immutable-legacy.zip",
        exe_data=b"rename-replacement",
    )
    links = []
    real_link = os.link

    def record_link(source, destination):
        result = real_link(source, destination)
        links.append((Path(source), Path(destination)))
        return result

    monkeypatch.setattr("water_server.release_batches.os.link", record_link)

    result = publish_desktop_release(replacement, db)

    assert len(links) == 1
    assert links[0][0].name.endswith(".staging")
    assert links[0][1] == result["path"]
    assert result["path"].name == f"{result['release_id']}.zip"
    assert result["path"].is_file()


def test_hard_link_failure_keeps_current_release_and_expires_reservation(
    app, tmp_path, db, monkeypatch
):
    baseline = release_zip(
        tmp_path,
        filename="link-baseline.zip",
        exe_data=b"link-baseline",
    )
    baseline_result = publish_desktop_release(baseline, db)
    old_bytes = baseline_result["path"].read_bytes()
    old_state = tuple(
        db.execute(
            "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
        ).fetchone()
    )
    replacement = release_zip(
        tmp_path,
        filename="link-failure.zip",
        exe_data=b"link-failure",
    )

    def fail_link(source, destination):
        raise OSError("injected hard link failure")

    monkeypatch.setattr("water_server.release_batches.os.link", fail_link)

    with pytest.raises(OSError, match="hard link"):
        publish_desktop_release(replacement, db)

    state = tuple(
        db.execute(
            "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
        ).fetchone()
    )
    failed_batch = db.execute(
        """
        SELECT model_generation, status FROM release_batches
        ORDER BY model_generation DESC LIMIT 1
        """
    ).fetchone()
    hidden_files = list(baseline_result["path"].parent.glob(".desktop-*.zip"))

    assert baseline_result["path"].read_bytes() == old_bytes
    assert state == old_state
    assert tuple(failed_batch) == (3, "expired")
    assert hidden_files == []


def test_legacy_release_rejects_outer_transaction_without_rolling_it_back(
    tmp_path, db
):
    archive = release_zip(tmp_path)
    db.execute(
        """
        INSERT INTO admin_audit (action, detail, created_at)
        VALUES ('legacy-caller-owned', 'must survive',
                '2026-07-16T00:00:00+00:00')
        """
    )

    with pytest.raises(RuntimeError, match="own transaction"):
        publish_desktop_release(archive, db)

    assert db.in_transaction
    pending = db.execute(
        "SELECT action FROM admin_audit WHERE action = 'legacy-caller-owned'"
    ).fetchone()
    assert pending["action"] == "legacy-caller-owned"
    db.rollback()


def test_valid_release_becomes_downloadable_and_contains_release_json(
    admin_client, client, tmp_path, db
):
    archive = release_zip(tmp_path)

    response = upload_release(admin_client, archive)

    assert response.status_code == 201
    state = db.execute("SELECT * FROM app_state WHERE id = 1").fetchone()
    assert state["model_generation"] == 2
    assert state["current_release_id"] != "initial"

    download = client.get("/downloads/desktop")
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.data)) as published:
        names = set(published.namelist())
        assert "reaction_classifier.h5" in names
        assert "yolov8n.pt" in names
        release = json.loads(published.read("release.json").decode("utf-8"))
        assert release["app_release_id"] == state["current_release_id"]
        assert release["release_batch_id"] + "-desktop" == release["app_release_id"]
        assert release["model_generation"] == 2
        assert release["app_version_code"] > 0
        assert release["app_version_name"]
    assert db.execute("SELECT COUNT(*) FROM desktop_releases").fetchone()[0] == 0


def test_release_publication_streams_archive_and_hashing(app, tmp_path, db, monkeypatch):
    archive = release_zip(tmp_path)

    def reject_whole_member_reads(*args, **kwargs):
        raise AssertionError("release publication must stream ZIP members")

    def reject_whole_file_reads(*args, **kwargs):
        raise AssertionError("release hashing must stream the file")

    monkeypatch.setattr(zipfile.ZipFile, "read", reject_whole_member_reads)
    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_reads)

    with app.app_context():
        result = publish_desktop_release(archive, db)

    assert result["path"].is_file()


def test_legacy_release_and_reservations_share_authoritative_generations(
    app, tmp_path, db
):
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    archive = release_zip(tmp_path)

    first_reservation = reserve_batch(db, now)
    legacy_release = publish_desktop_release(archive, db)
    second_reservation = reserve_batch(db, now + timedelta(seconds=1))

    assert first_reservation.model_generation == 2
    assert legacy_release["model_generation"] == 3
    assert second_reservation.model_generation == 4

    with zipfile.ZipFile(legacy_release["path"]) as published:
        release = json.loads(published.read("release.json").decode("utf-8"))
    assert release["model_generation"] == 3


def test_interleaved_publish_cleanup_cannot_delete_another_backup(
    app, tmp_path, db, monkeypatch
):
    baseline = release_zip(
        tmp_path,
        filename="baseline.zip",
        exe_data=b"baseline",
    )
    publish_desktop_release(baseline, db)

    source_a = release_zip(tmp_path, filename="a.zip", exe_data=b"release-a")
    source_b = release_zip(tmp_path, filename="b.zip", exe_data=b"release-b")
    baseline_row = db.execute(
        "SELECT stored_path FROM platform_releases "
        "WHERE platform = 'desktop' AND is_current = 1"
    ).fetchone()
    final_path = Path(baseline_row["stored_path"])
    baseline_bytes = final_path.read_bytes()
    a_committed = Event()
    b_backup_created = Event()
    a_cleanup_completed = Event()
    a_finished = Event()
    b_finished = Event()
    backup_paths = {}
    expected_before_b = []
    results = {}
    errors = {}

    real_link = os.link
    real_unlink = Path.unlink

    def controlled_link(source, destination):
        role = current_thread().name
        source = Path(source)
        destination = Path(destination)
        if role == "publisher-b":
            expected_before_b.append(source.read_bytes())
        result = real_link(source, destination)
        backup_paths[role] = destination
        return result

    def controlled_unlink(path, *args, **kwargs):
        result = real_unlink(path, *args, **kwargs)
        if (
            current_thread().name == "publisher-a"
            and path == backup_paths.get("publisher-a")
        ):
            a_cleanup_completed.set()
        return result

    monkeypatch.setattr("water_server.release_batches.os.link", controlled_link)
    monkeypatch.setattr(Path, "unlink", controlled_unlink)

    class CoordinatedConnection:
        def __init__(self, connection, role):
            self.connection = connection
            self.role = role
            self.finalizing = False

        @property
        def in_transaction(self):
            return self.connection.in_transaction

        def execute(self, statement, parameters=()):
            if "INSERT INTO platform_releases" in statement:
                self.finalizing = True
            if (
                self.role == "publisher-b"
                and "INSERT INTO platform_releases" in statement
            ):
                raise sqlite3.OperationalError("injected interleaved failure")
            return self.connection.execute(statement, parameters)

        def commit(self):
            result = self.connection.commit()
            if self.role == "publisher-a" and self.finalizing:
                a_committed.set()
            return result

        def rollback(self):
            return self.connection.rollback()

    def run_publisher(role, source, finished):
        if role == "publisher-b" and not a_committed.wait(timeout=5):
            errors[role] = AssertionError("publisher A did not commit")
            finished.set()
            return
        with app.app_context():
            connection = sqlite3.connect(app.config["DATABASE"], timeout=5)
            connection.row_factory = sqlite3.Row
            try:
                results[role] = publish_desktop_release(
                    source,
                    CoordinatedConnection(connection, role),
                )
            except BaseException as exc:
                errors[role] = exc
            finally:
                connection.close()
                finished.set()

    publisher_a = Thread(
        target=run_publisher,
        args=("publisher-a", source_a, a_finished),
        name="publisher-a",
    )
    publisher_b = Thread(
        target=run_publisher,
        args=("publisher-b", source_b, b_finished),
        name="publisher-b",
    )
    publisher_a.start()
    publisher_b.start()
    try:
        assert a_finished.wait(timeout=10)
        assert b_finished.wait(timeout=10)
    finally:
        publisher_a.join(timeout=6)
        publisher_b.join(timeout=6)

    assert not publisher_a.is_alive()
    assert not publisher_b.is_alive()
    assert "publisher-a" in results
    assert "publisher-a" not in errors
    assert isinstance(errors.get("publisher-b"), sqlite3.OperationalError)
    assert backup_paths["publisher-a"] != backup_paths["publisher-b"]
    assert final_path.read_bytes() == baseline_bytes
    assert list(final_path.parent.glob(".*.staging")) == []


def test_failed_publish_restores_before_rollback_allows_next_publish(
    app, tmp_path, db
):
    baseline = release_zip(
        tmp_path,
        filename="ordering-baseline.zip",
        exe_data=b"ordering-baseline",
    )
    publish_desktop_release(baseline, db)
    source_a = release_zip(
        tmp_path,
        filename="ordering-a.zip",
        exe_data=b"ordering-a",
    )
    source_b = release_zip(
        tmp_path,
        filename="ordering-b.zip",
        exe_data=b"ordering-b",
    )
    b_result = {}

    class FailingConnection:
        rollback_triggered = False

        @property
        def in_transaction(self):
            return db.in_transaction

        def execute(self, statement, parameters=()):
            if "INSERT INTO platform_releases" in statement:
                raise sqlite3.OperationalError("injected publisher A failure")
            return db.execute(statement, parameters)

        def commit(self):
            return db.commit()

        def rollback(self):
            db.rollback()
            if not self.rollback_triggered:
                self.rollback_triggered = True
                connection_b = sqlite3.connect(app.config["DATABASE"], timeout=5)
                connection_b.row_factory = sqlite3.Row
                try:
                    b_result.update(publish_desktop_release(source_b, connection_b))
                finally:
                    connection_b.close()

    with pytest.raises(sqlite3.OperationalError, match="publisher A"):
        publish_desktop_release(source_a, FailingConnection())

    state = db.execute(
        "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
    ).fetchone()
    final_path = Path(
        db.execute(
            "SELECT stored_path FROM platform_releases "
            "WHERE platform = 'desktop' AND is_current = 1"
        ).fetchone()[0]
    )
    with zipfile.ZipFile(final_path) as published:
        release = json.loads(published.read("release.json").decode("utf-8"))

    assert state["current_release_id"] == b_result["release_id"]
    assert state["model_generation"] == b_result["model_generation"]
    assert release["app_release_id"] == b_result["release_id"]
    assert release["model_generation"] == b_result["model_generation"]


def test_release_hashing_does_not_hold_database_write_lock(
    app, tmp_path, monkeypatch
):
    source = release_zip(
        tmp_path,
        filename="unlocked-hash.zip",
        exe_data=b"unlocked-hash",
    )
    hash_started = Event()
    allow_hash = Event()
    finished = Event()
    results = []
    errors = []
    real_hash = release_batches_module._stream_sha256

    def blocked_hash(path):
        hash_started.set()
        if not allow_hash.wait(timeout=5):
            raise AssertionError("hash was not released")
        return real_hash(path)

    monkeypatch.setattr("water_server.release_batches._stream_sha256", blocked_hash)

    def publish_in_thread():
        with app.app_context():
            connection = sqlite3.connect(app.config["DATABASE"], timeout=5)
            connection.row_factory = sqlite3.Row
            try:
                results.append(publish_desktop_release(source, connection))
            except BaseException as exc:
                errors.append(exc)
            finally:
                connection.close()
                finished.set()

    publisher = Thread(target=publish_in_thread, name="hash-publisher")
    publisher.start()
    try:
        assert hash_started.wait(timeout=5)
        with app.app_context():
            probe = sqlite3.connect(app.config["DATABASE"], timeout=0)
            probe.row_factory = sqlite3.Row
            try:
                concurrent = reserve_batch(
                    probe,
                    datetime(2026, 7, 16, tzinfo=timezone.utc),
                )
            finally:
                probe.close()
    finally:
        allow_hash.set()
        assert finished.wait(timeout=10)
        publisher.join(timeout=1)

    assert not publisher.is_alive()
    assert errors == []
    assert results[0]["model_generation"] == 2
    assert concurrent.model_generation == 3


def test_stale_legacy_publish_cannot_replace_newer_release(
    app, tmp_path, db, monkeypatch
):
    source_a = release_zip(
        tmp_path,
        filename="stale-a.zip",
        exe_data=b"stale-a",
    )
    source_b = release_zip(
        tmp_path,
        filename="stale-b.zip",
        exe_data=b"stale-b",
    )
    a_hash_started = Event()
    allow_a_hash = Event()
    a_finished = Event()
    a_errors = []
    real_hash = release_batches_module._stream_sha256

    def controlled_hash(path):
        if current_thread().name == "stale-publisher-a":
            a_hash_started.set()
            if not allow_a_hash.wait(timeout=5):
                raise AssertionError("publisher A hash was not released")
        return real_hash(path)

    monkeypatch.setattr("water_server.release_batches._stream_sha256", controlled_hash)

    def publish_a():
        with app.app_context():
            connection = sqlite3.connect(app.config["DATABASE"], timeout=5)
            connection.row_factory = sqlite3.Row
            try:
                publish_desktop_release(source_a, connection)
            except BaseException as exc:
                a_errors.append(exc)
            finally:
                connection.close()
                a_finished.set()

    publisher_a = Thread(target=publish_a, name="stale-publisher-a")
    publisher_a.start()
    try:
        assert a_hash_started.wait(timeout=5)
        connection_b = sqlite3.connect(app.config["DATABASE"], timeout=0)
        connection_b.row_factory = sqlite3.Row
        try:
            b_result = publish_desktop_release(source_b, connection_b)
        finally:
            connection_b.close()
    finally:
        allow_a_hash.set()
        assert a_finished.wait(timeout=10)
        publisher_a.join(timeout=1)

    assert not publisher_a.is_alive()
    assert len(a_errors) == 1
    assert isinstance(a_errors[0], InvalidRelease)
    assert "newer generation" in str(a_errors[0])

    state = db.execute(
        "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
    ).fetchone()
    final_path = Path(
        db.execute(
            "SELECT stored_path FROM platform_releases "
            "WHERE platform = 'desktop' AND is_current = 1"
        ).fetchone()[0]
    )
    with zipfile.ZipFile(final_path) as published:
        release = json.loads(published.read("release.json").decode("utf-8"))
    next_reservation = reserve_batch(
        db,
        datetime(2026, 7, 16, tzinfo=timezone.utc),
    )
    batch_rows = db.execute(
        """
        SELECT model_generation, status
        FROM release_batches
        ORDER BY model_generation
        """
    ).fetchall()

    assert state["current_release_id"] == b_result["release_id"]
    assert state["model_generation"] == b_result["model_generation"] == 3
    assert release["app_release_id"] == b_result["release_id"]
    assert release["model_generation"] == 3
    assert next_reservation.model_generation == 4
    assert [tuple(row) for row in batch_rows] == [
        (2, "expired"),
        (3, "partial"),
        (4, "reserved"),
    ]


def test_database_failure_leaves_immutable_current_pointer_unchanged(
    app, tmp_path, db
):
    baseline = release_zip(
        tmp_path,
        filename="restore-baseline.zip",
        exe_data=b"restore-baseline",
    )
    baseline_result = publish_desktop_release(baseline, db)
    source = release_zip(
        tmp_path,
        filename="restore-failure.zip",
        exe_data=b"restore-failure",
    )
    class FailingConnection:
        @property
        def in_transaction(self):
            return db.in_transaction

        def execute(self, statement, parameters=()):
            if "INSERT INTO platform_releases" in statement:
                raise sqlite3.OperationalError("original database failure")
            return db.execute(statement, parameters)

        def commit(self):
            return db.commit()

        def rollback(self):
            return db.rollback()

    with pytest.raises(sqlite3.OperationalError, match="original database"):
        publish_desktop_release(source, FailingConnection())

    current = db.execute(
        "SELECT release_id, stored_path FROM platform_releases "
        "WHERE platform = 'desktop' AND is_current = 1"
    ).fetchone()
    assert current["release_id"] == baseline_result["release_id"]
    assert Path(current["stored_path"]) == baseline_result["path"]
    assert not db.in_transaction


def test_dataset_change_during_build_rejects_legacy_activation(
    app, tmp_path, db, monkeypatch
):
    baseline = release_zip(
        tmp_path,
        filename="dataset-baseline.zip",
        exe_data=b"dataset-baseline",
    )
    baseline_result = publish_desktop_release(baseline, db)
    baseline_bytes = baseline_result["path"].read_bytes()
    before = db.execute(
        """
        SELECT current_release_id, model_generation
        FROM app_state WHERE id = 1
        """
    ).fetchone()
    source = release_zip(
        tmp_path,
        filename="dataset-stale.zip",
        exe_data=b"dataset-stale",
    )
    hash_started = Event()
    allow_hash = Event()
    finished = Event()
    errors = []
    real_hash = release_batches_module._stream_sha256

    def blocked_hash(path):
        hash_started.set()
        if not allow_hash.wait(timeout=5):
            raise AssertionError("dataset test hash was not released")
        return real_hash(path)

    monkeypatch.setattr("water_server.release_batches._stream_sha256", blocked_hash)

    def publish_in_thread():
        with app.app_context():
            connection = sqlite3.connect(app.config["DATABASE"], timeout=5)
            connection.row_factory = sqlite3.Row
            try:
                publish_desktop_release(source, connection)
            except BaseException as exc:
                errors.append(exc)
            finally:
                connection.close()
                finished.set()

    publisher = Thread(target=publish_in_thread, name="dataset-publisher")
    publisher.start()
    try:
        assert hash_started.wait(timeout=5)
        db.execute(
            """
            UPDATE app_state
            SET dataset_generation = dataset_generation + 1
            WHERE id = 1
            """
        )
        db.commit()
    finally:
        allow_hash.set()
        assert finished.wait(timeout=10)
        publisher.join(timeout=1)

    after = db.execute(
        """
        SELECT current_release_id, model_generation, dataset_generation
        FROM app_state WHERE id = 1
        """
    ).fetchone()
    failed_batch = db.execute(
        """
        SELECT model_generation, status
        FROM release_batches
        ORDER BY model_generation DESC
        LIMIT 1
        """
    ).fetchone()

    assert not publisher.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], InvalidRelease)
    assert "dataset generation" in str(errors[0])
    assert tuple(after[:2]) == tuple(before)
    assert after["dataset_generation"] == 2
    assert baseline_result["path"].read_bytes() == baseline_bytes
    assert tuple(failed_batch) == (3, "expired")


def test_legacy_after_bundle_uses_same_platform_pointer_and_download(
    app, client, tmp_path, db
):
    from test_release_batches import _desktop_artifact
    from water_server.release_batches import publish_bundle

    reserved = reserve_batch(db)
    bundle = _desktop_artifact(tmp_path, reserved, name="bundle-first.zip")
    publish_bundle(db, reserved.batch_id, desktop_path=bundle)
    legacy_source = release_zip(
        tmp_path, filename="legacy-after.zip", exe_data=b"legacy-after"
    )

    legacy = publish_desktop_release(legacy_source, db)
    current = db.execute(
        "SELECT release_id, stored_path FROM platform_releases "
        "WHERE platform = 'desktop' AND is_current = 1"
    ).fetchone()
    download = client.get("/downloads/desktop")

    assert current["release_id"] == legacy["release_id"]
    assert Path(current["stored_path"]) == legacy["path"]
    assert download.status_code == 200
    assert download.data == legacy["path"].read_bytes()


def test_bundle_after_legacy_downloads_bundle_immutable_file(
    app, client, tmp_path, db
):
    from test_release_batches import _desktop_artifact
    from water_server.release_batches import publish_bundle

    legacy = publish_desktop_release(
        release_zip(tmp_path, filename="legacy-first.zip"), db
    )
    reserved = reserve_batch(db)
    bundle = _desktop_artifact(tmp_path, reserved, name="bundle-after.zip")
    publish_bundle(db, reserved.batch_id, desktop_path=bundle)
    current = db.execute(
        "SELECT release_id, stored_path FROM platform_releases "
        "WHERE platform = 'desktop' AND is_current = 1"
    ).fetchone()
    download = client.get("/downloads/desktop")

    assert current["release_id"] == f"{reserved.batch_id}-desktop"
    assert current["release_id"] != legacy["release_id"]
    assert download.status_code == 200
    assert download.data == Path(current["stored_path"]).read_bytes()


def test_legacy_release_keeps_android_installations_active(tmp_path, db):
    db.execute(
        """
        INSERT INTO installations (
            installation_id, token_hash, app_release_id, model_generation,
            client_platform, active, created_at
        ) VALUES ('android-install', 'hash', 'android-release', 1,
                  'android', 1, '2026-07-16T00:00:00+00:00')
        """
    )
    db.commit()

    publish_desktop_release(release_zip(tmp_path), db)

    active = db.execute(
        "SELECT active FROM installations WHERE installation_id = 'android-install'"
    ).fetchone()[0]
    assert active == 1


def test_legacy_admin_audit_failure_does_not_activate_release(
    admin_client, tmp_path
):
    app = admin_client.application
    app.config["PROPAGATE_EXCEPTIONS"] = False
    with app.app_context():
        db = get_db()
        db.execute(
            """
            CREATE TRIGGER fail_legacy_release_audit
            BEFORE INSERT ON admin_audit
            WHEN NEW.action = 'desktop_release_uploaded'
            BEGIN
                SELECT RAISE(FAIL, 'injected legacy audit failure');
            END
            """
        )
        db.commit()

    response = upload_release(admin_client, release_zip(tmp_path))

    assert response.status_code == 500
    with app.app_context():
        db = get_db()
        assert db.execute("SELECT COUNT(*) FROM platform_releases").fetchone()[0] == 0
        state = db.execute(
            "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
        ).fetchone()
        assert tuple(state) == ("initial", 1)


def test_repeated_legacy_audit_failures_remove_their_immutable_orphans(
    admin_client, tmp_path
):
    app = admin_client.application
    baseline = release_zip(
        tmp_path,
        filename="cleanup-baseline.zip",
        exe_data=b"cleanup-baseline",
    )
    with app.app_context():
        db = get_db()
        baseline_result = publish_desktop_release(baseline, db)
        original_state = tuple(
            db.execute(
                "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
            ).fetchone()
        )
        db.execute(
            """
            CREATE TRIGGER fail_repeated_legacy_release_audit
            BEFORE INSERT ON admin_audit
            WHEN NEW.action = 'desktop_release_uploaded'
            BEGIN
                SELECT RAISE(FAIL, 'injected repeated legacy audit failure');
            END
            """
        )
        db.commit()
    app.config["PROPAGATE_EXCEPTIONS"] = False

    responses = [
        upload_release(
            admin_client,
            release_zip(
                tmp_path,
                filename=f"failed-{index}.zip",
                exe_data=f"failed-{index}".encode(),
            ),
        )
        for index in range(3)
    ]

    assert [response.status_code for response in responses] == [500, 500, 500]
    with app.app_context():
        db = get_db()
        state = tuple(
            db.execute(
                "SELECT current_release_id, model_generation FROM app_state WHERE id = 1"
            ).fetchone()
        )
        failed_batches = db.execute(
            "SELECT batch_id FROM release_batches WHERE status = 'expired'"
        ).fetchall()
        release_count = db.execute(
            "SELECT COUNT(*) FROM platform_releases"
        ).fetchone()[0]

    releases = baseline_result["path"].parent
    failed_paths = [
        releases / f"{row['batch_id']}-desktop.zip" for row in failed_batches
    ]
    assert len(failed_paths) == 3
    assert all(not path.exists() for path in failed_paths)
    assert release_count == 1
    assert state == original_state
    assert baseline_result["path"].is_file()


def test_legacy_release_records_source_filename(tmp_path, db):
    source = release_zip(tmp_path, filename="管理员原始发布包.zip")

    result = publish_desktop_release(source, db)
    original_filename = db.execute(
        "SELECT original_filename FROM platform_releases WHERE release_id = ?",
        (result["release_id"],),
    ).fetchone()[0]

    assert original_filename == source.name


def test_invalid_legacy_entry_runs_orphan_cleanup_before_and_after(
    tmp_path, db, monkeypatch
):
    invalid = release_zip(tmp_path, include_detector=False)
    calls = []
    monkeypatch.setattr(
        "water_server.releases._run_orphan_cleanup_best_effort",
        lambda connection: calls.append(connection),
        raising=False,
    )

    with pytest.raises(InvalidRelease):
        publish_desktop_release(invalid, db)

    assert calls == [db, db]


def test_legacy_admin_route_preserves_original_upload_filename(
    admin_client, tmp_path
):
    source = release_zip(tmp_path, filename="用户原始包.zip")

    response = upload_release(admin_client, source)

    assert response.status_code == 201
    with admin_client.application.app_context():
        original_filename = get_db().execute(
            "SELECT original_filename FROM platform_releases "
            "WHERE platform = 'desktop' AND is_current = 1"
        ).fetchone()[0]
    assert original_filename == "用户原始包.zip"


def test_legacy_admin_route_sanitizes_path_and_control_characters(
    admin_client, tmp_path
):
    source = release_zip(tmp_path, filename="safe-source.zip")
    unsafe_name = "..\\private/用户\x01原始包.zip"

    with source.open("rb") as stream:
        response = admin_client.post(
            "/admin/releases/desktop",
            data={"file": (stream, unsafe_name)},
            content_type="multipart/form-data",
        )

    assert response.status_code == 201
    with admin_client.application.app_context():
        original_filename = get_db().execute(
            "SELECT original_filename FROM platform_releases "
            "WHERE platform = 'desktop' AND is_current = 1"
        ).fetchone()[0]
    assert original_filename == "用户原始包.zip"
    assert original_filename
    assert "/" not in original_filename
    assert "\\" not in original_filename


def test_legacy_admin_cleans_partial_temp_when_save_fails(
    admin_client, tmp_path, monkeypatch
):
    from werkzeug.datastructures import FileStorage

    app = admin_client.application
    app.config["PROPAGATE_EXCEPTIONS"] = False
    temp_dir = Path(app.config["STORAGE_ROOT"]) / "temp"
    before = set(temp_dir.iterdir())

    def partial_save(_uploaded, destination, *_args, **_kwargs):
        Path(destination).write_bytes(b"partial")
        raise OSError("injected legacy save failure")

    monkeypatch.setattr(FileStorage, "save", partial_save)
    response = upload_release(admin_client, release_zip(tmp_path))

    assert response.status_code == 500
    assert set(temp_dir.iterdir()) == before
