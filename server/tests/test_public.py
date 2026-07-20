import hashlib
import os
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import water_server.public as public_module

from test_uploads import post_zip


def _publish_android(app, tmp_path, monkeypatch, *, original_filename=None):
    from water_server.db import get_db
    from water_server.release_batches import publish_bundle, reserve_batch
    from test_release_batches import _android_artifact, _allow_test_signer

    _allow_test_signer(monkeypatch)
    with app.app_context():
        reserved = reserve_batch(get_db())
    android = _android_artifact(tmp_path, reserved)
    kwargs = {}
    if original_filename is not None:
        kwargs["original_filenames"] = {"android": original_filename}
    with app.app_context():
        publish_bundle(
            get_db(), reserved.batch_id, android_path=android, **kwargs
        )
        row = get_db().execute(
            "SELECT stored_path FROM platform_releases "
            "WHERE release_id = ?",
            (reserved.android_release_id,),
        ).fetchone()
    return reserved, android, Path(row["stored_path"])


def test_bundle_desktop_is_available_to_legacy_download_route(
    app, client, tmp_path
):
    from water_server.db import get_db
    from water_server.release_batches import publish_bundle, reserve_batch
    from test_release_batches import _desktop_artifact

    with app.app_context():
        reserved = reserve_batch(get_db())
    desktop = _desktop_artifact(tmp_path, reserved)
    with app.app_context():
        publish_bundle(get_db(), reserved.batch_id, desktop_path=desktop)

    response = client.get("/downloads/desktop")

    assert response.status_code == 200
    assert response.data == desktop.read_bytes()


def test_statistics_returns_all_water_types_without_details(
    auth_client, result_zip
):
    post_zip(auth_client, result_zip)

    response = auth_client.get("/api/v1/public/statistics")
    payload = response.get_json()

    assert response.status_code == 200
    assert [row["water_type"] for row in payload["water_types"]] == [
        "污水",
        "生活用水",
        "养殖水体",
    ]
    assert payload["water_types"][0]["positive_count"] == 2
    assert payload["water_types"][0]["negative_count"] == 1
    assert payload["water_types"][0]["positive_ratio"] == 2 / 3
    assert "storage_path" not in response.get_data(as_text=True)


def test_empty_water_type_has_null_ratios(client):
    payload = client.get("/api/v1/public/statistics").get_json()

    assert payload["water_types"][0]["positive_ratio"] is None
    assert payload["water_types"][0]["negative_ratio"] is None


def test_homepage_contains_three_charts_and_download_controls(client):
    html = client.get("/").get_data(as_text=True)

    assert html.count('class="donut"') == 3
    assert "下载电脑端检测程序" in html
    assert "下载手机端检测程序" in html
    assert "管理员登录" in html


def test_desktop_download_can_be_offloaded_to_nginx(
    app, client, tmp_path
):
    from water_server.db import get_db
    from test_releases import release_zip

    legacy_path = release_zip(tmp_path)
    with app.app_context():
        db = get_db()
        db.execute(
            """
            INSERT INTO desktop_releases (
                release_id, model_generation, original_filename, stored_path,
                sha256, uploaded_at, is_current
            ) VALUES ('legacy', 1, 'release.zip', ?, 'sha',
                      '2026-07-16T00:00:00+00:00', 1)
            """,
            (str(legacy_path),),
        )
        db.commit()
    app.config["USE_X_ACCEL_REDIRECT"] = True

    response = client.get("/downloads/desktop")

    assert response.status_code == 200
    assert (
        response.headers["X-Accel-Redirect"]
        == "/_desktop_release/desktop-latest.zip"
    )
    assert response.headers["Content-Type"] == "application/zip"


def test_mobile_release_metadata_and_download_use_published_immutable_apk(
    app, client, tmp_path, monkeypatch
):
    reserved, _source, stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    expected_bytes = stored_path.read_bytes()
    expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()

    metadata_response = client.get("/api/v1/mobile/releases/current")
    payload = metadata_response.get_json()
    download_response = client.get("/downloads/mobile")

    assert metadata_response.status_code == 200
    assert payload == {
        "release_id": reserved.android_release_id,
        "model_generation": reserved.model_generation,
        "dataset_generation": reserved.dataset_generation,
        "version_code": 17,
        "version_name": "1.7.0",
        "size_bytes": len(expected_bytes),
        "sha256": expected_sha256,
        "download_url": (
            f"https://example.invalid/downloads/mobile/"
            f"{reserved.android_release_id}.apk"
        ),
        "mandatory": False,
        "release_notes": "",
    }
    assert len(payload["sha256"]) == 64
    assert payload["download_url"].endswith(
        f"/downloads/mobile/{reserved.android_release_id}.apk"
    )
    assert download_response.status_code == 200
    assert (
        download_response.mimetype
        == "application/vnd.android.package-archive"
    )
    assert download_response.data == expected_bytes
    assert hashlib.sha256(download_response.data).hexdigest() == expected_sha256
    assert "attachment" in download_response.headers["Content-Disposition"]
    assert metadata_response.headers["Cache-Control"] == (
        "no-store, no-cache, must-revalidate"
    )
    assert download_response.headers["Cache-Control"] == (
        "no-store, no-cache, must-revalidate"
    )


@pytest.mark.parametrize(
    "route",
    ["/api/v1/mobile/releases/current", "/downloads/mobile"],
)
def test_mobile_release_routes_return_json_404_without_current(client, route):
    response = client.get(route)

    assert response.status_code == 404
    assert response.is_json
    assert response.get_json() == {
        "code": "not_found",
        "message": "mobile release not found",
    }


def test_mobile_release_uses_configured_public_base_url(
    app, client, tmp_path, monkeypatch
):
    _publish_android(app, tmp_path, monkeypatch)
    app.config["PUBLIC_BASE_URL"] = "https://downloads.example.test/root/"

    payload = client.get("/api/v1/mobile/releases/current").get_json()

    assert (
        payload["download_url"]
        == "https://downloads.example.test/root/downloads/mobile/"
        + payload["release_id"]
        + ".apk"
    )


def test_mobile_download_can_be_offloaded_to_immutable_nginx_path(
    app, client, tmp_path, monkeypatch
):
    reserved, _source, _stored_path = _publish_android(
        app, tmp_path, monkeypatch, original_filename="mobile-release.apk"
    )
    app.config["USE_X_ACCEL_REDIRECT"] = True
    app.config["TRUSTED_IMMUTABLE_RELEASES"] = True
    monkeypatch.setattr(
        public_module,
        "_is_posix_platform",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        public_module,
        "_release_can_be_offloaded",
        lambda _path, _file_stat: public_module._trusted_x_accel_permissions(
            SimpleNamespace(
                st_mode=stat.S_IFDIR | 0o555, st_uid=0, st_gid=0
            ),
            SimpleNamespace(
                st_mode=stat.S_IFREG | 0o444, st_uid=0, st_gid=0
            ),
            1000,
            {1000},
        ),
        raising=False,
    )

    response = client.get("/downloads/mobile")

    assert response.status_code == 200
    assert response.data == b""
    assert response.headers["X-Accel-Redirect"] == (
        f"/_mobile_release/{reserved.android_release_id}.apk"
    )
    assert (
        response.mimetype == "application/vnd.android.package-archive"
    )
    assert "mobile-release.apk" in response.headers["Content-Disposition"]


def test_mobile_download_defaults_to_direct_when_x_accel_is_not_trusted(
    app, client, tmp_path, monkeypatch
):
    _reserved, _source, stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    expected = stored_path.read_bytes()
    app.config["USE_X_ACCEL_REDIRECT"] = True
    assert app.config.get("TRUSTED_IMMUTABLE_RELEASES", False) is False

    response = client.get("/downloads/mobile")

    assert response.status_code == 200
    assert response.data == expected
    assert "X-Accel-Redirect" not in response.headers


@pytest.mark.parametrize(
    ("trusted", "posix", "permissions_trusted"),
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_mobile_download_requires_every_x_accel_trust_guard(
    app,
    client,
    tmp_path,
    monkeypatch,
    trusted,
    posix,
    permissions_trusted,
):
    _reserved, _source, stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    expected = stored_path.read_bytes()
    app.config["USE_X_ACCEL_REDIRECT"] = True
    app.config["TRUSTED_IMMUTABLE_RELEASES"] = trusted
    monkeypatch.setattr(
        public_module,
        "_is_posix_platform",
        lambda: posix,
        raising=False,
    )
    monkeypatch.setattr(
        public_module,
        "_release_can_be_offloaded",
        lambda _path, _file_stat: permissions_trusted,
        raising=False,
    )

    response = client.get("/downloads/mobile")

    assert response.status_code == 200
    assert response.data == expected
    assert "X-Accel-Redirect" not in response.headers


def test_x_accel_permissions_require_other_owner_and_no_effective_write():
    checker = getattr(public_module, "_trusted_x_accel_permissions", None)
    assert callable(checker)
    directory = SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o555, st_uid=0, st_gid=0
    )
    release = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o444, st_uid=0, st_gid=0
    )

    assert checker(directory, release, 1000, {1000}) is True
    assert checker(directory, release, 0, {0}) is False
    assert checker(directory, release, 0, {1000}) is False
    assert checker(directory, release, 1000, {0}) is True
    assert checker(
        SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o775, st_uid=0, st_gid=1000
        ),
        release,
        1000,
        {1000},
    ) is False
    assert checker(
        directory,
        SimpleNamespace(
            st_mode=stat.S_IFREG | 0o446, st_uid=0, st_gid=0
        ),
        1000,
        {1000},
    ) is False
    assert checker(
        directory,
        SimpleNamespace(
            st_mode=stat.S_IFREG | 0o444, st_uid=1000, st_gid=0
        ),
        1000,
        {1000},
    ) is False


def _synthetic_directory_stat(*, mode=0o555, uid=0, gid=0, inode=1):
    return SimpleNamespace(
        st_mode=stat.S_IFDIR | mode,
        st_uid=uid,
        st_gid=gid,
        st_dev=1,
        st_ino=inode,
    )


def _synthetic_release_stat(*, mode=0o444, uid=0, gid=0):
    return SimpleNamespace(
        st_mode=stat.S_IFREG | mode,
        st_uid=uid,
        st_gid=gid,
    )


def test_x_accel_ancestry_permissions_require_every_directory_trusted():
    checker = getattr(
        public_module, "_trusted_x_accel_ancestry_permissions", None
    )
    assert callable(checker)
    trusted_chain = [
        _synthetic_directory_stat(inode=index) for index in range(1, 6)
    ]
    release = _synthetic_release_stat()

    assert checker(trusted_chain, release, 1000, {1000}) is True

    writable_storage = list(trusted_chain)
    writable_storage[1] = _synthetic_directory_stat(
        mode=0o575, gid=1000, inode=2
    )
    assert checker(writable_storage, release, 1000, {1000}) is False

    same_uid_parent = list(trusted_chain)
    same_uid_parent[3] = _synthetic_directory_stat(uid=1000, inode=4)
    assert checker(same_uid_parent, release, 1000, {1000}) is False


def test_x_accel_guard_rejects_writable_storage_ancestor(
    app, tmp_path, monkeypatch
):
    releases = tmp_path / "storage" / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    trusted_release_directory = _synthetic_directory_stat(inode=1)
    writable_storage = _synthetic_directory_stat(
        mode=0o575, gid=1000, inode=2
    )
    monkeypatch.setattr(
        public_module,
        "_immutable_directory_stat_chain",
        lambda _configured, _resolved: [
            trusted_release_directory,
            writable_storage,
        ],
        raising=False,
    )
    monkeypatch.setattr(public_module.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.setattr(public_module.os, "getegid", lambda: 1000, raising=False)
    monkeypatch.setattr(public_module.os, "getgroups", lambda: [], raising=False)
    monkeypatch.setattr(
        Path,
        "stat",
        lambda _path, *_args, **_kwargs: trusted_release_directory,
    )

    app.config["STORAGE_ROOT"] = str(tmp_path / "storage")
    with app.app_context():
        trusted = public_module._release_can_be_offloaded(
            releases / "release.apk", _synthetic_release_stat()
        )

    assert trusted is False


def test_immutable_directory_chain_rejects_symlink(tmp_path):
    collector = getattr(
        public_module, "_immutable_directory_stat_chain", None
    )
    assert callable(collector)
    real_releases = tmp_path / "real-releases"
    real_releases.mkdir()
    linked_releases = tmp_path / "linked-releases"
    try:
        linked_releases.symlink_to(real_releases, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    assert collector(linked_releases, real_releases.resolve()) is None


def test_immutable_directory_chain_rejects_replaced_directory(
    tmp_path, monkeypatch
):
    collector = getattr(
        public_module, "_immutable_directory_stat_chain", None
    )
    assert callable(collector)
    releases = (tmp_path / "releases").absolute()
    releases.mkdir()
    real_lstat = releases.lstat()
    calls = 0

    def replacing_lstat(path):
        nonlocal calls
        if Path(path) == releases:
            calls += 1
            if calls == 2:
                return SimpleNamespace(
                    st_mode=real_lstat.st_mode,
                    st_uid=real_lstat.st_uid,
                    st_gid=real_lstat.st_gid,
                    st_dev=real_lstat.st_dev,
                    st_ino=real_lstat.st_ino + 1,
                )
        return Path(path).lstat()

    monkeypatch.setattr(
        public_module, "_path_lstat", replacing_lstat, raising=False
    )
    monkeypatch.setattr(
        public_module,
        "_path_resolve",
        lambda path: Path(path).absolute(),
        raising=False,
    )

    assert collector(releases, releases) is None
    assert calls == 2


def test_immutable_directory_chain_stops_at_filesystem_root():
    collector = getattr(
        public_module, "_immutable_directory_stat_chain", None
    )
    assert callable(collector)
    root = Path(Path.cwd().anchor)

    chain = collector(root, root.resolve(strict=True))

    assert chain is not None
    assert len(chain) == 1


def test_mobile_download_rejects_unsafe_original_filename_for_header(
    app, client, tmp_path, monkeypatch
):
    reserved, _source, _stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    with app.app_context():
        from water_server.db import get_db

        db = get_db()
        db.execute(
            "UPDATE platform_releases SET original_filename = ? "
            "WHERE release_id = ?",
            ("release.apk\r\nX-Evil: injected", reserved.android_release_id),
        )
        db.commit()
    app.config["USE_X_ACCEL_REDIRECT"] = True

    response = client.get("/downloads/mobile")

    disposition = response.headers["Content-Disposition"]
    assert response.status_code == 200
    assert "\r" not in disposition
    assert "\n" not in disposition
    assert "X-Evil" not in response.headers
    assert f"{reserved.android_release_id}.apk" in disposition


def _change_current_android(app, statement, parameters):
    from water_server.db import get_db

    with app.app_context():
        db = get_db()
        db.execute(statement, parameters)
        db.commit()


@pytest.mark.parametrize(
    "route",
    ["/api/v1/mobile/releases/current", "/downloads/mobile"],
)
def test_mobile_release_rejects_stored_path_outside_release_storage(
    app, client, tmp_path, monkeypatch, route
):
    reserved, _source, stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    outside = tmp_path / "outside" / stored_path.name
    outside.parent.mkdir()
    outside.write_bytes(stored_path.read_bytes())
    _change_current_android(
        app,
        "UPDATE platform_releases SET stored_path = ? WHERE release_id = ?",
        (str(outside), reserved.android_release_id),
    )

    response = client.get(route)

    assert response.status_code == 503
    assert response.get_json() == {
        "code": "mobile_release_unavailable",
        "message": "mobile release is unavailable",
    }
    assert str(outside) not in response.get_data(as_text=True)


def test_mobile_release_rejects_symlinked_apk(
    app, client, tmp_path, monkeypatch
):
    reserved, _source, stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    target = tmp_path / "outside.apk"
    target.write_bytes(stored_path.read_bytes())
    stored_path.unlink()
    try:
        os.symlink(target, stored_path)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")

    response = client.get("/downloads/mobile")

    assert response.status_code == 503
    assert response.is_json
    assert "outside.apk" not in response.get_data(as_text=True)


def test_mobile_release_rejects_size_mismatch(
    app, client, tmp_path, monkeypatch
):
    reserved, _source, _stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    _change_current_android(
        app,
        "UPDATE platform_releases SET size_bytes = size_bytes + 1 "
        "WHERE release_id = ?",
        (reserved.android_release_id,),
    )

    response = client.get("/api/v1/mobile/releases/current")

    assert response.status_code == 503
    assert response.is_json
    assert response.get_json()["code"] == "mobile_release_unavailable"


def _same_size_corruption(data):
    return bytes([data[0] ^ 0xFF]) + data[1:]


@pytest.mark.parametrize(
    ("route_kind", "use_x_accel"),
    [
        ("metadata", False),
        ("current", False),
        ("version", False),
        ("current", True),
        ("version", True),
    ],
)
def test_mobile_release_rejects_same_size_content_tampering(
    app, client, tmp_path, monkeypatch, route_kind, use_x_accel
):
    reserved, _source, stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    original = stored_path.read_bytes()
    stored_path.chmod(0o644)
    stored_path.write_bytes(_same_size_corruption(original))
    assert stored_path.stat().st_size == len(original)
    app.config["USE_X_ACCEL_REDIRECT"] = use_x_accel
    routes = {
        "metadata": "/api/v1/mobile/releases/current",
        "current": "/downloads/mobile",
        "version": f"/downloads/mobile/{reserved.android_release_id}.apk",
    }

    response = client.get(routes[route_kind])

    assert response.status_code == 503
    assert response.is_json
    assert response.get_json() == {
        "code": "mobile_release_unavailable",
        "message": "mobile release is unavailable",
    }
    assert "X-Accel-Redirect" not in response.headers


def test_mobile_integrity_rechecks_replaced_inode(
    app, client, tmp_path, monkeypatch
):
    _reserved, _source, stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    expected = stored_path.read_bytes()
    original_stat = stored_path.stat()

    assert client.get("/api/v1/mobile/releases/current").status_code == 200

    corrupt_replacement = stored_path.with_name("corrupt-replacement.apk")
    corrupt_replacement.write_bytes(_same_size_corruption(expected))
    os.utime(
        corrupt_replacement,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    stored_path.chmod(0o644)
    os.replace(corrupt_replacement, stored_path)

    assert client.get("/api/v1/mobile/releases/current").status_code == 503

    valid_replacement = stored_path.with_name("valid-replacement.apk")
    valid_replacement.write_bytes(expected)
    os.utime(
        valid_replacement,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    os.replace(valid_replacement, stored_path)

    assert client.get("/api/v1/mobile/releases/current").status_code == 200


@pytest.mark.parametrize("route_kind", ["metadata", "current", "version"])
def test_mobile_release_rehashes_same_inode_after_mtime_is_restored(
    app, client, tmp_path, monkeypatch, route_kind
):
    reserved, _source, stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    original = stored_path.read_bytes()
    original_stat = stored_path.stat()
    corruption = _same_size_corruption(original)

    assert client.get("/api/v1/mobile/releases/current").status_code == 200

    stored_path.chmod(0o644)
    stored_path.write_bytes(corruption)
    os.utime(
        stored_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    tampered_stat = stored_path.stat()
    assert tampered_stat.st_ino == original_stat.st_ino
    assert tampered_stat.st_size == original_stat.st_size
    assert tampered_stat.st_mtime_ns == original_stat.st_mtime_ns
    routes = {
        "metadata": "/api/v1/mobile/releases/current",
        "current": "/downloads/mobile",
        "version": f"/downloads/mobile/{reserved.android_release_id}.apk",
    }

    response = client.get(routes[route_kind])

    assert response.status_code == 503
    assert response.is_json
    assert response.get_json()["code"] == "mobile_release_unavailable"
    assert response.data != corruption


def _digest_open_file(opened_file):
    digest = hashlib.sha256()
    while chunk := opened_file.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _parallel_metadata_statuses(app, count=8):
    barrier = threading.Barrier(count)

    def fetch():
        barrier.wait(timeout=5)
        with app.test_client() as threaded_client:
            return threaded_client.get(
                "/api/v1/mobile/releases/current"
            ).status_code

    with ThreadPoolExecutor(max_workers=count) as executor:
        return list(executor.map(lambda _index: fetch(), range(count)))


def test_posix_mobile_hash_verification_is_single_flight(
    app, tmp_path, monkeypatch
):
    _publish_android(app, tmp_path, monkeypatch)
    monkeypatch.setattr(
        public_module,
        "_is_posix_platform",
        lambda: True,
        raising=False,
    )
    calls = 0
    calls_lock = threading.Lock()

    def delayed_hash(opened_file):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.2)
        return _digest_open_file(opened_file)

    monkeypatch.setattr(
        public_module,
        "_hash_open_mobile_file",
        delayed_hash,
        raising=False,
    )

    statuses = _parallel_metadata_statuses(app)

    assert statuses == [200] * 8
    assert calls == 1


def test_posix_mobile_hash_failure_wakes_waiters_and_allows_retry(
    app, client, tmp_path, monkeypatch
):
    _publish_android(app, tmp_path, monkeypatch)
    monkeypatch.setattr(
        public_module,
        "_is_posix_platform",
        lambda: True,
        raising=False,
    )
    calls = 0
    fail = True
    calls_lock = threading.Lock()

    def controlled_hash(opened_file):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.2)
        if fail:
            raise OSError("injected hash read failure")
        return _digest_open_file(opened_file)

    monkeypatch.setattr(
        public_module,
        "_hash_open_mobile_file",
        controlled_hash,
        raising=False,
    )

    statuses = _parallel_metadata_statuses(app)
    assert statuses == [503] * 8
    assert calls == 1

    fail = False
    response = client.get("/api/v1/mobile/releases/current")

    assert response.status_code == 200
    assert calls == 2


def test_posix_mobile_hash_interruption_releases_waiters_and_allows_retry(
    app, client, tmp_path, monkeypatch
):
    class VerificationInterrupted(BaseException):
        pass

    class BoundedTrackingEvent:
        def __init__(self):
            self._event = threading.Event()
            self._lock = threading.Lock()
            self._waiter_count = 0
            self.wait_results = []

        def wait(self, timeout=None):
            with self._lock:
                self._waiter_count += 1
                if self._waiter_count == 8:
                    all_waiters_ready.set()
            signaled = self._event.wait(timeout=1)
            with self._lock:
                self.wait_results.append(signaled)
            return signaled

        def set(self):
            self._event.set()

    _publish_android(app, tmp_path, monkeypatch)
    monkeypatch.setattr(
        public_module,
        "_is_posix_platform",
        lambda: True,
        raising=False,
    )
    leader_hashing = threading.Event()
    release_leader = threading.Event()
    all_waiters_ready = threading.Event()
    flight_events = []

    def flight_factory():
        event = BoundedTrackingEvent()
        flight_events.append(event)
        return SimpleNamespace(event=event, result=None)

    monkeypatch.setattr(public_module, "_IntegrityFlight", flight_factory)
    calls = 0
    calls_lock = threading.Lock()

    def interrupted_hash(opened_file):
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            leader_hashing.set()
            assert release_leader.wait(timeout=5)
            raise VerificationInterrupted("injected verification interruption")
        return _digest_open_file(opened_file)

    monkeypatch.setattr(
        public_module,
        "_hash_open_mobile_file",
        interrupted_hash,
        raising=False,
    )

    def fetch_status():
        with app.test_client() as threaded_client:
            return threaded_client.get(
                "/api/v1/mobile/releases/current"
            ).status_code

    with ThreadPoolExecutor(max_workers=9) as executor:
        leader = executor.submit(fetch_status)
        assert leader_hashing.wait(timeout=5)
        waiters = [executor.submit(fetch_status) for _index in range(8)]
        assert all_waiters_ready.wait(timeout=5)
        release_leader.set()

        with pytest.raises(
            VerificationInterrupted,
            match="injected verification interruption",
        ):
            leader.result(timeout=2)
        statuses = [waiter.result(timeout=2) for waiter in waiters]

    state = app.extensions["mobile_release_integrity"]
    assert statuses == [503] * 8
    assert calls == 1
    assert len(flight_events) == 1
    assert flight_events[0].wait_results == [True] * 8
    assert state.in_flight == {}

    response = client.get("/api/v1/mobile/releases/current")

    assert response.status_code == 200
    assert calls == 2


def test_windows_mobile_integrity_hashes_every_request(
    app, client, tmp_path, monkeypatch
):
    _publish_android(app, tmp_path, monkeypatch)
    monkeypatch.setattr(
        public_module,
        "_is_posix_platform",
        lambda: False,
        raising=False,
    )
    calls = 0

    def counted_hash(opened_file):
        nonlocal calls
        calls += 1
        return _digest_open_file(opened_file)

    monkeypatch.setattr(
        public_module,
        "_hash_open_mobile_file",
        counted_hash,
        raising=False,
    )

    assert client.get("/api/v1/mobile/releases/current").status_code == 200
    assert client.get("/api/v1/mobile/releases/current").status_code == 200
    assert calls == 2


def test_posix_mobile_integrity_cache_key_includes_ctime():
    builder = getattr(public_module, "_mobile_integrity_cache_key", None)
    assert callable(builder)
    before = SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_size=3,
        st_mtime_ns=4,
        st_ctime_ns=5,
    )
    after = SimpleNamespace(**{**before.__dict__, "st_ctime_ns": 6})

    assert builder(Path("release.apk"), before, "a" * 64) != builder(
        Path("release.apk"), after, "a" * 64
    )


def test_posix_mobile_integrity_cache_is_bounded(
    app, client, tmp_path, monkeypatch
):
    _reserved, _source, stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    expected = stored_path.read_bytes()
    monkeypatch.setattr(
        public_module,
        "_is_posix_platform",
        lambda: True,
        raising=False,
    )

    for index in range(70):
        replacement = stored_path.with_name(f"cache-entry-{index}.apk")
        replacement.write_bytes(expected)
        stored_path.chmod(0o644)
        os.replace(replacement, stored_path)
        response = client.get("/api/v1/mobile/releases/current")
        assert response.status_code == 200

    state = app.extensions["mobile_release_integrity"]
    assert len(state.verified) == public_module._MAX_INTEGRITY_CACHE_ENTRIES


def test_mobile_hash_failure_closes_verified_file_object(
    app, client, tmp_path, monkeypatch
):
    reserved, _source, stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    stored_path.chmod(0o644)
    stored_path.write_bytes(b"")
    _change_current_android(
        app,
        "UPDATE platform_releases SET size_bytes = 0, sha256 = ? "
        "WHERE release_id = ?",
        ("a" * 64, reserved.android_release_id),
    )
    captured = []
    real_fdopen = public_module.os.fdopen

    def capture_fdopen(*args, **kwargs):
        opened = real_fdopen(*args, **kwargs)
        captured.append(opened)
        return opened

    monkeypatch.setattr(public_module.os, "fdopen", capture_fdopen)

    response = client.get("/api/v1/mobile/releases/current")

    assert response.status_code == 503
    assert len(captured) == 1
    assert captured[0].closed


def test_direct_mobile_download_sends_and_closes_same_verified_file_object(
    app, client, tmp_path, monkeypatch
):
    _reserved, _source, stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    expected = stored_path.read_bytes()
    replacement = stored_path.with_name("toctou-replacement.apk")
    replacement.write_bytes(_same_size_corruption(expected))
    captured = []
    replacement_blocked = []
    real_send_file = public_module.send_file

    def replace_path_before_send(file_or_path, *args, **kwargs):
        captured.append(file_or_path)
        stored_path.chmod(0o644)
        try:
            os.replace(replacement, stored_path)
        except PermissionError:
            replacement_blocked.append(True)
        return real_send_file(file_or_path, *args, **kwargs)

    monkeypatch.setattr(public_module, "send_file", replace_path_before_send)

    response = client.get("/downloads/mobile")

    assert response.status_code == 200
    assert response.data == expected
    assert len(captured) == 1
    assert hasattr(captured[0], "read")
    assert replacement_blocked or stored_path.read_bytes() != expected
    response.close()
    assert captured[0].closed


def test_mobile_open_rejects_inode_replaced_between_lstat_and_open(
    app, client, tmp_path, monkeypatch
):
    _reserved, _source, stored_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    expected = stored_path.read_bytes()
    replacement = stored_path.with_name("open-race-replacement.apk")
    replacement.write_bytes(expected)
    real_open = public_module.os.open
    replaced = False

    def replace_before_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if not replaced and Path(path) == stored_path:
            replaced = True
            stored_path.chmod(0o644)
            os.replace(replacement, stored_path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(public_module.os, "open", replace_before_open)

    response = client.get("/downloads/mobile")

    assert replaced is True
    assert response.status_code == 503
    assert response.is_json


def test_versioned_mobile_download_remains_available_after_new_release(
    app, client, tmp_path, monkeypatch
):
    first, _source, first_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    first_bytes = first_path.read_bytes()
    first_url = f"/downloads/mobile/{first.android_release_id}.apk"

    second, _source, second_path = _publish_android(
        app, tmp_path, monkeypatch
    )
    second_bytes = second_path.read_bytes()

    first_response = client.get(first_url)
    fixed_response = client.get("/downloads/mobile")
    metadata_response = client.get("/api/v1/mobile/releases/current")
    payload = metadata_response.get_json()

    assert first_response.status_code == 200
    assert first_response.data == first_bytes
    assert hashlib.sha256(first_response.data).hexdigest() == hashlib.sha256(
        first_bytes
    ).hexdigest()
    assert first_response.headers["Cache-Control"] == (
        "public, max-age=31536000, immutable"
    )
    assert fixed_response.data == second_bytes
    assert fixed_response.headers["Cache-Control"] == (
        "no-store, no-cache, must-revalidate"
    )
    assert payload["release_id"] == second.android_release_id
    assert payload["download_url"].endswith(
        f"/downloads/mobile/{second.android_release_id}.apk"
    )


@pytest.mark.parametrize(
    "release_id",
    ["not-a-release", "A" * 32 + "-android", "a" * 32 + "-desktop"],
)
def test_versioned_mobile_download_rejects_invalid_release_id(client, release_id):
    response = client.get(f"/downloads/mobile/{release_id}.apk")

    assert response.status_code == 404
    assert response.is_json
    assert response.get_json()["code"] == "not_found"


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://downloads.example.test",
        "https:///missing-authority",
        "https://user:pass@downloads.example.test",
        "https://downloads.example.test/root?query=1",
        "https://downloads.example.test/root?",
        "https://downloads.example.test/root#fragment",
        "https://downloads.example.test/root#",
        "https://downloads.example.test/white space",
        "https://downloads.example.test/root\r\nX-Evil: yes",
    ],
)
def test_mobile_metadata_rejects_invalid_public_base_url(
    app, client, tmp_path, monkeypatch, base_url
):
    _publish_android(app, tmp_path, monkeypatch)
    app.config["PUBLIC_BASE_URL"] = base_url

    response = client.get("/api/v1/mobile/releases/current")

    assert response.status_code == 503
    assert response.is_json
    assert response.get_json() == {
        "code": "invalid_public_base_url",
        "message": "mobile release configuration is invalid",
    }


def test_default_config_loads_public_base_url_from_environment(monkeypatch, tmp_path):
    from water_server.config import default_config

    monkeypatch.setenv(
        "WATER_PUBLIC_BASE_URL", "https://cdn.example.test/releases"
    )

    config = default_config(str(tmp_path / "instance"))

    assert config["PUBLIC_BASE_URL"] == "https://cdn.example.test/releases"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("TRUE", True), ("1", True), ("false", False), ("0", False)],
)
def test_default_config_loads_strict_trusted_release_boolean(
    monkeypatch, tmp_path, raw, expected
):
    from water_server.config import default_config

    monkeypatch.setenv("WATER_TRUSTED_IMMUTABLE_RELEASES", raw)

    assert (
        default_config(str(tmp_path / "instance"))[
            "TRUSTED_IMMUTABLE_RELEASES"
        ]
        is expected
    )


def test_default_config_rejects_invalid_trusted_release_boolean(
    monkeypatch, tmp_path
):
    from water_server.config import default_config

    monkeypatch.setenv("WATER_TRUSTED_IMMUTABLE_RELEASES", "yes")

    with pytest.raises(ValueError, match="WATER_TRUSTED_IMMUTABLE_RELEASES"):
        default_config(str(tmp_path / "instance"))
