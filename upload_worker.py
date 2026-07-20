from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from client_config import ReleaseConfig, app_data_root
from upload_queue import UploadQueue, UploadTask


RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ApiResponse:
    status_code: int
    payload: dict

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def code(self) -> str:
        return str(self.payload.get("code") or self.payload.get("status") or self.status_code)


def next_retry_delay(attempts: int, rng: Callable[[], float] = random.random) -> float:
    base = min(3600.0, 5.0 * (2 ** min(max(attempts, 0), 10)))
    return min(3600.0, base * (0.8 + rng() * 0.4))


class ClientApi:
    def __init__(
        self,
        release: ReleaseConfig,
        credentials_path: Path | None = None,
        session: requests.Session | None = None,
    ):
        self.release = release
        self.credentials_path = credentials_path or app_data_root() / "credentials.json"
        self.session = session or requests.Session()

    def _load_credentials(self) -> dict | None:
        try:
            return json.loads(self.credentials_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _register(self) -> dict:
        response = self.session.post(
            f"{self.release.api_base_url}/api/v1/client/register",
            json={
                "bootstrap_token": self.release.bootstrap_token,
                "client_platform": "desktop",
                "app_release_id": self.release.app_release_id,
                "model_generation": self.release.model_generation,
            },
            timeout=(8, 20),
        )
        response.raise_for_status()
        credentials = response.json()
        self.credentials_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.credentials_path.with_suffix(".tmp")
        temp.write_text(json.dumps(credentials), encoding="utf-8")
        temp.replace(self.credentials_path)
        return credentials

    def _upload_once(
        self,
        task: UploadTask,
        credentials: dict,
    ) -> ApiResponse:
        headers = {
            "Authorization": f"Bearer {credentials['token']}",
            "X-Installation-ID": credentials["installation_id"],
        }
        with task.archive_path.open("rb") as stream:
            response = self.session.post(
                f"{self.release.api_base_url}/api/v1/results",
                headers=headers,
                files={"file": (task.archive_path.name, stream, "application/zip")},
                timeout=(10, 90),
            )
        try:
            payload = response.json()
        except ValueError:
            payload = {"code": f"http_{response.status_code}"}
        return ApiResponse(response.status_code, payload)

    def upload(self, task: UploadTask) -> ApiResponse:
        credentials = self._load_credentials() or self._register()
        response = self._upload_once(task, credentials)
        if response.status_code == 401 and response.code in {
            "authentication_required",
            "invalid_credentials",
        }:
            self.credentials_path.unlink(missing_ok=True)
            response = self._upload_once(task, self._register())
        return response


def process_one(
    queue: UploadQueue,
    api,
    now: float | None = None,
    rng: Callable[[], float] = random.random,
) -> UploadTask | None:
    current = time.time() if now is None else now
    task = queue.claim_next(current)
    if task is None:
        return None
    try:
        response = api.upload(task)
    except requests.RequestException as exc:
        queue.schedule_retry(
            task.upload_id,
            next_retry_delay(task.attempts, rng),
            exc.__class__.__name__,
            current,
        )
        return task
    except OSError as exc:
        queue.mark_rejected(task.upload_id, f"local_file_error:{exc}")
        return task

    if response.ok or response.status_code == 208:
        queue.mark_uploaded(task.upload_id)
    elif response.status_code in RETRYABLE_STATUS:
        queue.schedule_retry(
            task.upload_id,
            next_retry_delay(task.attempts, rng),
            response.code,
            current,
        )
    else:
        queue.mark_rejected(task.upload_id, response.code)
    return task


class UploadWorker(threading.Thread):
    def __init__(
        self,
        queue: UploadQueue,
        api: ClientApi,
        status_callback: Callable[[str, int], None] | None = None,
        poll_seconds: float = 5,
    ):
        super().__init__(name="result-upload-worker", daemon=True)
        self.queue = queue
        self.api = api
        self.status_callback = status_callback
        self.poll_seconds = poll_seconds
        self._wake = threading.Event()
        self._stop_event = threading.Event()

    def wake(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            processed = process_one(self.queue, self.api)
            pending = self.queue.pending_count()
            if self.status_callback:
                status = "同步完成" if pending == 0 else "等待上传"
                self.status_callback(status, pending)
            if processed is None:
                self._wake.wait(self.poll_seconds)
                self._wake.clear()
