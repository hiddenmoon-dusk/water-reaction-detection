import os
import threading

import numpy as np
import pytest
from PyQt5.QtCore import QObject, QThread, pyqtSlot
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QMessageBox

from client_config import ReleaseConfig
from reaction_app import DetectScreen, ORBITAL_STYLE, UploadStatusBridge


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class FakeQueue:
    def __init__(self):
        self.tasks = []

    def enqueue(self, task):
        self.tasks.append(task)

    def pending_count(self):
        return len(self.tasks)


def release():
    return ReleaseConfig(
        schema_version=1,
        app_release_id="initial",
        model_generation=1,
        dataset_generation=1,
        api_base_url="https://water.example.test",
        bootstrap_token="test",
    )


def test_save_enqueues_upload(qapp, monkeypatch, tmp_path):
    queue = FakeQueue()
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    screen = DetectScreen(
        "污水",
        release=release(),
        upload_queue=queue,
        result_root=tmp_path / "结果",
        archive_root=tmp_path / "queue",
    )
    screen.img_rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    screen.image_view.annotations = [
        (3, 4, 30, 40, "已反应", 0.95, QColor(183, 227, 41))
    ]

    screen.saveResult()

    assert queue.pending_count() == 1
    assert (tmp_path / "结果" / "污水" / "001" / "result.json").is_file()


def test_stylesheet_contains_archive_tokens():
    assert "#d9ff3f" in ORBITAL_STYLE
    assert "#00a7b5" in ORBITAL_STYLE
    assert "#111111" in ORBITAL_STYLE
    assert "border-radius:0" in ORBITAL_STYLE


def test_image_view_motion_entry_points(qapp):
    from reaction_app import ImageView

    view = ImageView()
    view.startScanMotion()
    view.startResultMotion()
    view.stopMotion()


def test_upload_status_bridge_delivers_on_qt_thread(qapp):
    bridge = UploadStatusBridge()

    class Receiver(QObject):
        def __init__(self):
            super().__init__()
            self.received = None

        @pyqtSlot(str, int)
        def receive(self, status, pending):
            self.received = (status, pending, QThread.currentThread())

    receiver = Receiver()
    bridge.statusChanged.connect(receiver.receive)

    thread = threading.Thread(target=lambda: bridge.publish("等待上传", 2))
    thread.start()
    thread.join()
    qapp.processEvents()

    assert receiver.received == ("等待上传", 2, qapp.thread())
