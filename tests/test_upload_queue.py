from pathlib import Path

from upload_queue import UploadQueue, UploadTask


def make_task(tmp_path, upload_id="u-1"):
    archive = tmp_path / f"{upload_id}.zip"
    archive.write_bytes(b"zip")
    return UploadTask(
        upload_id=upload_id,
        archive_path=archive,
        dataset_generation=1,
        app_release_id="initial",
        model_generation=1,
    )


def test_enqueue_survives_reopen(tmp_path):
    path = tmp_path / "queue.db"
    first = UploadQueue(path)
    first.enqueue(make_task(tmp_path))
    first.close()

    reopened = UploadQueue(path)

    assert reopened.get("u-1").status == "pending"
    assert reopened.pending_count() == 1


def test_duplicate_enqueue_is_idempotent(tmp_path):
    queue = UploadQueue(tmp_path / "queue.db")
    task = make_task(tmp_path)

    queue.enqueue(task)
    queue.enqueue(task)

    assert queue.total_count() == 1


def test_stale_uploading_task_returns_to_pending_on_reopen(tmp_path):
    path = tmp_path / "queue.db"
    queue = UploadQueue(path)
    queue.enqueue(make_task(tmp_path))
    claimed = queue.claim_next(now=100)
    assert claimed.status == "uploading"
    queue.close()

    reopened = UploadQueue(path)

    assert reopened.get("u-1").status == "pending"


def test_mark_uploaded_deletes_archive(tmp_path):
    queue = UploadQueue(tmp_path / "queue.db")
    task = make_task(tmp_path)
    queue.enqueue(task)

    queue.mark_uploaded(task.upload_id)

    assert queue.get(task.upload_id).status == "uploaded"
    assert not task.archive_path.exists()
