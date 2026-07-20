from dataclasses import dataclass
import json

import requests

from client_config import ReleaseConfig
from upload_queue import UploadQueue, UploadTask
from upload_worker import ApiResponse, ClientApi, next_retry_delay, process_one


def enqueue(tmp_path):
    archive = tmp_path / "u-1.zip"
    archive.write_bytes(b"zip")
    queue = UploadQueue(tmp_path / "queue.db")
    queue.enqueue(
        UploadTask(
            upload_id="u-1",
            archive_path=archive,
            dataset_generation=1,
            app_release_id="initial",
            model_generation=1,
        )
    )
    return queue


@dataclass
class FakeApi:
    response: ApiResponse | None = None
    error: Exception | None = None

    def upload(self, task):
        if self.error:
            raise self.error
        return self.response


def test_generation_rejection_is_not_retried(tmp_path):
    queue = enqueue(tmp_path)
    api = FakeApi(ApiResponse(409, {"code": "generation_expired"}))

    process_one(queue, api, now=100)

    task = queue.get("u-1")
    assert task.status == "rejected"
    assert task.last_error == "generation_expired"


def test_server_error_is_scheduled_for_retry(tmp_path):
    queue = enqueue(tmp_path)
    api = FakeApi(ApiResponse(503, {"code": "temporarily_unavailable"}))

    process_one(queue, api, now=100, rng=lambda: 0.5)

    task = queue.get("u-1")
    assert task.status == "retry_wait"
    assert task.next_attempt_at == 110


def test_connection_error_is_scheduled_for_retry(tmp_path):
    queue = enqueue(tmp_path)
    api = FakeApi(error=requests.ConnectionError("offline"))

    process_one(queue, api, now=100, rng=lambda: 0.5)

    assert queue.get("u-1").status == "retry_wait"


def test_success_marks_uploaded(tmp_path):
    queue = enqueue(tmp_path)
    api = FakeApi(ApiResponse(201, {"status": "created"}))

    process_one(queue, api, now=100)

    assert queue.get("u-1").status == "uploaded"


def test_retry_delay_is_capped():
    assert next_retry_delay(20, rng=lambda: 0.5) == 3600


def test_client_api_reregisters_once_when_saved_credentials_are_invalid(tmp_path):
    archive = tmp_path / "sample.zip"
    archive.write_bytes(b"zip")
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(
        json.dumps({"installation_id": "old", "token": "old-token"}),
        encoding="utf-8",
    )

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

    class FakeSession:
        def __init__(self):
            self.responses = [
                FakeResponse(401, {"code": "invalid_credentials"}),
                FakeResponse(
                    201,
                    {"installation_id": "new", "token": "new-token"},
                ),
                FakeResponse(201, {"status": "created"}),
            ]
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return self.responses.pop(0)

    release = ReleaseConfig(
        schema_version=1,
        app_release_id="release-3",
        model_generation=3,
        dataset_generation=2,
        api_base_url="https://example.test",
        bootstrap_token="bootstrap",
    )
    session = FakeSession()
    api = ClientApi(
        release,
        credentials_path=credentials_path,
        session=session,
    )
    task = UploadTask(
        upload_id="upload-1",
        archive_path=archive,
        dataset_generation=2,
        app_release_id="release-3",
        model_generation=3,
    )

    response = api.upload(task)

    assert response.status_code == 201
    assert len(session.calls) == 3
    assert session.calls[1][1]["json"]["client_platform"] == "desktop"
    assert json.loads(credentials_path.read_text(encoding="utf-8")) == {
        "installation_id": "new",
        "token": "new-token",
    }
