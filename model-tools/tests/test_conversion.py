from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from dataclasses import FrozenInstanceError
import hashlib
import multiprocessing
import os
from pathlib import Path
import threading
import time
from types import MappingProxyType
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from water_models import conversion
from water_models.conversion import (
    ModelConversionError,
    atomic_write_bytes,
    convert_classifier,
    convert_detector,
    inspect_tflite,
)


@pytest.fixture(scope="session")
def tiny_detector_tflite_bytes() -> bytes:
    """Build the valid detector fixture once for the whole test session."""

    inputs = tf.keras.Input(batch_shape=(1, 640, 640, 3), name="images")
    pooled = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    raw = tf.keras.layers.Dense(50)(pooled)
    outputs = tf.keras.layers.Reshape((5, 10))(raw)
    model = tf.keras.Model(inputs, outputs)
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def _save_tiny_classifier(path: Path) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input((128, 128, 3), name="pixels"),
            tf.keras.layers.Rescaling(1 / 255.0),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                kernel_initializer="ones",
                bias_initializer="zeros",
            ),
        ]
    )
    model.save(path)
    return model


def _run_tflite(path: Path, image: np.ndarray) -> float:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_detail["index"], image)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_detail["index"])[0, 0])


def _multiprocess_atomic_writer(
    target_text: str,
    payload: bytes,
    workers_ready: object,
    active_writers: object,
    maximum_active_writers: object,
    counter_guard: object,
) -> None:
    target = Path(target_text)
    workers_ready.wait(timeout=30)
    with conversion._serialized_target(target):
        with counter_guard:
            active_writers.value += 1
            maximum_active_writers.value = max(
                maximum_active_writers.value,
                active_writers.value,
            )
        time.sleep(0.1)
        with counter_guard:
            active_writers.value -= 1
    for _ in range(2):
        atomic_write_bytes(target, payload)


def test_classifier_conversion_preserves_probability(tmp_path: Path) -> None:
    source = tmp_path / "classifier.h5"
    target = tmp_path / "classifier.tflite"
    model = _save_tiny_classifier(source)
    image = np.full((1, 128, 128, 3), 64, dtype=np.float32)

    convert_classifier(source, target, precision="fp16")

    inspected = inspect_tflite(target)
    expected = float(model(image, training=False).numpy()[0, 0])
    assert inspected.input.shape == [1, 128, 128, 3]
    assert inspected.output.shape == [1, 1]
    assert _run_tflite(target, image) == pytest.approx(expected, abs=1e-3)
    assert target.stat().st_size > 0


def test_classifier_conversion_supports_fp32(tmp_path: Path) -> None:
    source = tmp_path / "classifier.h5"
    target = tmp_path / "classifier.tflite"
    _save_tiny_classifier(source)

    convert_classifier(source, target, precision="fp32")

    inspected = inspect_tflite(target)
    assert inspected.input.dtype == "float32"
    assert inspected.output.dtype == "float32"


def test_classifier_conversion_rejects_source_as_target_without_modifying_h5(
    tmp_path: Path,
) -> None:
    source = tmp_path / "classifier.h5"
    _save_tiny_classifier(source)
    original = source.read_bytes()

    with pytest.raises(ModelConversionError, match="same file"):
        convert_classifier(source, source, precision="fp16")

    assert source.read_bytes() == original
    reloaded = tf.keras.models.load_model(source, compile=False)
    assert reloaded.input_shape == (None, 128, 128, 3)
    assert reloaded.output_shape == (None, 1)


def test_classifier_conversion_rejects_normalized_path_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "classifier.h5"
    source.write_bytes(b"source")
    (tmp_path / "alias").mkdir()
    target = tmp_path / "alias" / ".." / source.name
    monkeypatch.setattr(
        conversion.tf.keras.models,
        "load_model",
        lambda *args, **kwargs: pytest.fail("aliased source must not be loaded"),
    )

    with pytest.raises(ModelConversionError, match="same file"):
        convert_classifier(source, target, precision="fp16")

    assert source.read_bytes() == b"source"


@pytest.mark.skipif(os.name != "nt", reason="Windows path case normalization")
def test_classifier_conversion_rejects_case_only_path_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "classifier.h5"
    source.write_bytes(b"source")
    target = tmp_path / "CLASSIFIER.H5"
    monkeypatch.setattr(
        conversion.tf.keras.models,
        "load_model",
        lambda *args, **kwargs: pytest.fail("aliased source must not be loaded"),
    )

    with pytest.raises(ModelConversionError, match="same file"):
        convert_classifier(source, target, precision="fp16")

    assert source.read_bytes() == b"source"


def test_classifier_conversion_rejects_hardlink_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "classifier.h5"
    target = tmp_path / "classifier-alias.h5"
    source.write_bytes(b"source")
    os.link(source, target)
    monkeypatch.setattr(
        conversion.tf.keras.models,
        "load_model",
        lambda *args, **kwargs: pytest.fail("aliased source must not be loaded"),
    )

    with pytest.raises(ModelConversionError, match="same file"):
        convert_classifier(source, target, precision="fp16")

    assert source.read_bytes() == b"source"
    assert target.read_bytes() == b"source"


def test_classifier_conversion_rejects_symlink_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "classifier.h5"
    target = tmp_path / "classifier-alias.h5"
    source.write_bytes(b"source")
    try:
        target.symlink_to(source)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    monkeypatch.setattr(
        conversion.tf.keras.models,
        "load_model",
        lambda *args, **kwargs: pytest.fail("aliased source must not be loaded"),
    )

    with pytest.raises(ModelConversionError, match="same file"):
        convert_classifier(source, target, precision="fp16")

    assert source.read_bytes() == b"source"
    assert target.read_bytes() == b"source"


def test_classifier_conversion_tolerates_samefile_probe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "classifier.h5"
    target = tmp_path / "classifier.tflite"
    source.write_bytes(b"source")
    target.write_bytes(b"existing")
    model = SimpleNamespace(
        input_shape=(None, 128, 128, 3), output_shape=(None, 1)
    )
    converter = SimpleNamespace(
        optimizations=[],
        target_spec=SimpleNamespace(supported_types=[]),
        convert=lambda: b"converted",
    )
    monkeypatch.setattr(
        conversion.os.path,
        "samefile",
        lambda first, second: (_ for _ in ()).throw(OSError("probe unavailable")),
    )
    monkeypatch.setattr(
        conversion.tf.keras.models,
        "load_model",
        lambda path, compile: model,
    )
    monkeypatch.setattr(
        conversion.tf.lite.TFLiteConverter,
        "from_keras_model",
        lambda loaded: converter,
    )

    convert_classifier(source, target, precision="fp32")

    assert target.read_bytes() == b"converted"


@pytest.mark.parametrize("precision", ("fp16", "fp32"))
def test_classifier_conversion_configures_requested_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
) -> None:
    source = tmp_path / "classifier.h5"
    target = tmp_path / "classifier.tflite"
    model = SimpleNamespace(
        input_shape=(None, 128, 128, 3), output_shape=(None, 1)
    )
    untouched_supported_types = [object()]
    converter = SimpleNamespace(
        optimizations=[],
        target_spec=SimpleNamespace(supported_types=untouched_supported_types),
        convert=lambda: b"converted",
    )
    load_calls: list[tuple[Path, bool]] = []

    def load_model(path: Path, *, compile: bool) -> object:
        load_calls.append((path, compile))
        return model

    monkeypatch.setattr(conversion.tf.keras.models, "load_model", load_model)
    monkeypatch.setattr(
        conversion.tf.lite.TFLiteConverter,
        "from_keras_model",
        lambda loaded: converter,
    )

    convert_classifier(source, target, precision)

    assert load_calls == [(source, False)]
    assert converter.optimizations == [tf.lite.Optimize.DEFAULT]
    if precision == "fp16":
        assert converter.target_spec.supported_types == [tf.float16]
    else:
        assert converter.target_spec.supported_types is untouched_supported_types


def test_classifier_conversion_rejects_unsupported_precision(tmp_path: Path) -> None:
    target = tmp_path / "classifier.tflite"
    target.write_bytes(b"existing")

    with pytest.raises(ModelConversionError, match="unsupported precision: int8"):
        convert_classifier(tmp_path / "missing.h5", target, precision="int8")

    assert target.read_bytes() == b"existing"


@pytest.mark.parametrize(
    ("input_shape", "output_shape"),
    (
        ((None, 64, 128, 3), (None, 1)),
        ((None, 128, 128, 3), (None, 2)),
    ),
)
def test_classifier_conversion_rejects_tensor_contract_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_shape: tuple[int | None, ...],
    output_shape: tuple[int | None, ...],
) -> None:
    model = SimpleNamespace(input_shape=input_shape, output_shape=output_shape)
    monkeypatch.setattr(
        conversion.tf.keras.models,
        "load_model",
        lambda source, compile: model,
    )

    with pytest.raises(ModelConversionError, match="tensor contract mismatch"):
        convert_classifier(tmp_path / "classifier.h5", tmp_path / "out.tflite", "fp16")


def test_converter_failure_preserves_target_and_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "classifier.h5"
    source.write_bytes(b"source")
    target = tmp_path / "classifier.tflite"
    target.write_bytes(b"existing")
    model = SimpleNamespace(
        input_shape=(None, 128, 128, 3), output_shape=(None, 1)
    )
    failure = RuntimeError("conversion failed")

    class FailingConverter:
        optimizations: list[object] = []
        target_spec = SimpleNamespace(supported_types=[])

        def convert(self) -> bytes:
            raise failure

    monkeypatch.setattr(
        conversion.tf.keras.models,
        "load_model",
        lambda path, compile: model,
    )
    monkeypatch.setattr(
        conversion.tf.lite.TFLiteConverter,
        "from_keras_model",
        lambda loaded: FailingConverter(),
    )

    with pytest.raises(ModelConversionError, match="convert classifier") as caught:
        convert_classifier(source, target, "fp16")

    assert caught.value.__cause__ is failure
    assert target.read_bytes() == b"existing"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "classifier.h5",
        "classifier.tflite",
    ]


def test_atomic_write_creates_parent_and_replaces_target(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "model.tflite"

    atomic_write_bytes(target, b"converted")

    assert target.read_bytes() == b"converted"
    assert list(target.parent.iterdir()) == [target]


class _FakeKernel32:
    def __init__(
        self,
        *,
        handle: int = 101,
        wait_result: int = 0,
        release_result: bool = True,
        close_result: bool = True,
        release_exception: BaseException | None = None,
        close_exception: BaseException | None = None,
    ) -> None:
        self.handle = handle
        self.wait_result = wait_result
        self.release_result = release_result
        self.close_result = close_result
        self.release_exception = release_exception
        self.close_exception = close_exception
        self.calls: list[tuple[object, ...]] = []

    def CreateMutexW(
        self, security: object, initial_owner: bool, name: str
    ) -> int:
        self.calls.append(("create", security, initial_owner, name))
        return self.handle

    def WaitForSingleObject(self, handle: int, timeout_ms: int) -> int:
        self.calls.append(("wait", handle, timeout_ms))
        return self.wait_result

    def ReleaseMutex(self, handle: int) -> bool:
        self.calls.append(("release", handle))
        if self.release_exception is not None:
            raise self.release_exception
        return self.release_result

    def CloseHandle(self, handle: int) -> bool:
        self.calls.append(("close", handle))
        if self.close_exception is not None:
            raise self.close_exception
        return self.close_result


class _FakeWinFunction:
    argtypes: list[object] | None = None
    restype: object = None


def test_kernel32_signatures_use_win32_abi_types() -> None:
    kernel32 = SimpleNamespace(
        CreateMutexW=_FakeWinFunction(),
        WaitForSingleObject=_FakeWinFunction(),
        ReleaseMutex=_FakeWinFunction(),
        CloseHandle=_FakeWinFunction(),
    )

    conversion._configure_kernel32(kernel32)

    assert kernel32.CreateMutexW.argtypes == [
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    assert kernel32.CreateMutexW.restype is wintypes.HANDLE
    assert kernel32.WaitForSingleObject.argtypes == [
        wintypes.HANDLE,
        wintypes.DWORD,
    ]
    assert kernel32.WaitForSingleObject.restype is wintypes.DWORD
    assert kernel32.ReleaseMutex.argtypes == [wintypes.HANDLE]
    assert kernel32.ReleaseMutex.restype is wintypes.BOOL
    assert kernel32.CloseHandle.argtypes == [wintypes.HANDLE]
    assert kernel32.CloseHandle.restype is wintypes.BOOL


def test_windows_mutex_name_is_stable_and_path_specific() -> None:
    first_key = r"c:\models\classifier.tflite"
    second_key = r"c:\models\other.tflite"

    first = conversion._windows_mutex_name(first_key)

    assert first == conversion._windows_mutex_name(first_key)
    assert first != conversion._windows_mutex_name(second_key)
    assert first == (
        "Local\\water-models-"
        + hashlib.sha256(first_key.encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize("wait_result", (0x00000000, 0x00000080))
def test_windows_mutex_accepts_acquired_and_abandoned_waits(
    monkeypatch: pytest.MonkeyPatch, wait_result: int
) -> None:
    kernel32 = _FakeKernel32(wait_result=wait_result)
    monkeypatch.setattr(conversion, "_last_windows_error", lambda: 5, raising=False)

    with conversion._windows_named_mutex("target-key", kernel32=kernel32):
        kernel32.calls.append(("body",))

    assert kernel32.calls == [
        (
            "create",
            None,
            False,
            conversion._windows_mutex_name("target-key"),
        ),
        ("wait", 101, 30_000),
        ("body",),
        ("release", 101),
        ("close", 101),
    ]


def test_windows_mutex_timeout_closes_without_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = _FakeKernel32(wait_result=0x00000102)
    monkeypatch.setattr(conversion, "_last_windows_error", lambda: 5, raising=False)

    with pytest.raises(TimeoutError, match="30.0 seconds"):
        with conversion._windows_named_mutex("target-key", kernel32=kernel32):
            pytest.fail("timed-out mutex must not enter")

    assert [call[0] for call in kernel32.calls] == ["create", "wait", "close"]


def test_windows_mutex_create_failure_reports_windows_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = _FakeKernel32(handle=0)
    monkeypatch.setattr(conversion, "_last_windows_error", lambda: 6)

    with pytest.raises(OSError, match="CreateMutexW.*6"):
        with conversion._windows_named_mutex("target-key", kernel32=kernel32):
            pytest.fail("failed mutex creation must not enter")

    assert [call[0] for call in kernel32.calls] == ["create"]


def test_windows_mutex_wait_failure_reports_windows_error_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = _FakeKernel32(wait_result=0xFFFFFFFF)
    monkeypatch.setattr(conversion, "_last_windows_error", lambda: 123, raising=False)

    with pytest.raises(OSError, match="WaitForSingleObject.*123"):
        with conversion._windows_named_mutex("target-key", kernel32=kernel32):
            pytest.fail("failed mutex wait must not enter")

    assert [call[0] for call in kernel32.calls] == ["create", "wait", "close"]


def test_windows_mutex_cleanup_errors_do_not_replace_body_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = _FakeKernel32(release_result=False, close_result=False)
    monkeypatch.setattr(conversion, "_last_windows_error", lambda: 5, raising=False)
    primary = RuntimeError("body failed")

    with pytest.raises(RuntimeError, match="body failed") as caught:
        with conversion._windows_named_mutex("target-key", kernel32=kernel32):
            raise primary

    assert caught.value is primary
    assert [call[0] for call in kernel32.calls] == [
        "create",
        "wait",
        "release",
        "close",
    ]
    notes = getattr(caught.value, "__notes__", [])
    assert any("ReleaseMutex" in note for note in notes)
    assert any("CloseHandle" in note for note in notes)


def test_windows_mutex_cleanup_call_exceptions_preserve_body_and_close_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_failure = RuntimeError("release call raised")
    close_failure = RuntimeError("close call raised")
    kernel32 = _FakeKernel32(
        release_exception=release_failure,
        close_exception=close_failure,
    )
    monkeypatch.setattr(conversion, "_last_windows_error", lambda: 5)
    primary = ValueError("body failed")

    with pytest.raises(ValueError, match="body failed") as caught:
        with conversion._windows_named_mutex("target-key", kernel32=kernel32):
            raise primary

    assert caught.value is primary
    assert [call[0] for call in kernel32.calls] == [
        "create",
        "wait",
        "release",
        "close",
    ]
    notes = getattr(caught.value, "__notes__", [])
    assert any("ReleaseMutex" in note and "release call raised" in note for note in notes)
    assert any("CloseHandle" in note and "close call raised" in note for note in notes)


def test_windows_mutex_cleanup_raises_release_error_after_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = _FakeKernel32(release_result=False, close_result=False)
    monkeypatch.setattr(conversion, "_last_windows_error", lambda: 5, raising=False)

    with pytest.raises(OSError, match="ReleaseMutex.*5") as caught:
        with conversion._windows_named_mutex("target-key", kernel32=kernel32):
            pass

    assert [call[0] for call in kernel32.calls] == [
        "create",
        "wait",
        "release",
        "close",
    ]
    assert any(
        "CloseHandle" in note for note in getattr(caught.value, "__notes__", [])
    )


def test_parent_directory_sync_uses_open_fsync_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[tuple[Path, int]] = []
    synced: list[int] = []
    closed: list[int] = []
    directory_descriptor = 12345

    monkeypatch.setattr(
        conversion,
        "_supports_parent_directory_sync",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        conversion.os,
        "open",
        lambda path, flags: opened.append((path, flags)) or directory_descriptor,
    )
    monkeypatch.setattr(conversion.os, "fsync", synced.append)
    monkeypatch.setattr(conversion.os, "close", closed.append)

    conversion._sync_parent_directory(tmp_path)

    assert opened == [(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))]
    assert synced == [directory_descriptor]
    assert closed == [directory_descriptor]


def test_parent_directory_sync_preserves_fsync_error_when_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = 12345
    primary = OSError("directory fsync failed")
    close_failure = OSError("directory close failed")
    monkeypatch.setattr(
        conversion,
        "_supports_parent_directory_sync",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(conversion.os, "open", lambda path, flags: descriptor)
    monkeypatch.setattr(
        conversion.os,
        "fsync",
        lambda received: (_ for _ in ()).throw(primary),
    )
    monkeypatch.setattr(
        conversion.os,
        "close",
        lambda received: (_ for _ in ()).throw(close_failure),
    )

    with pytest.raises(OSError, match="directory fsync failed") as caught:
        conversion._sync_parent_directory(tmp_path)

    assert caught.value is primary
    assert any(
        "directory close failed" in note
        for note in getattr(caught.value, "__notes__", [])
    )


def test_atomic_write_exposes_parent_sync_failure_after_complete_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.tflite"
    target.write_bytes(b"existing")
    failure = OSError("directory fsync failed")
    monkeypatch.setattr(
        conversion,
        "_sync_parent_directory",
        lambda parent: (_ for _ in ()).throw(failure),
        raising=False,
    )

    with pytest.raises(OSError, match="directory fsync failed") as caught:
        atomic_write_bytes(target, b"converted")

    assert caught.value is failure
    assert target.read_bytes() == b"converted"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_retries_transient_windows_replace_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.tflite"
    target.write_bytes(b"existing")
    original_replace = Path.replace
    replace_attempts = 0
    sleep_delays: list[float] = []

    def flaky_replace(path: Path, destination: Path) -> Path:
        nonlocal replace_attempts
        replace_attempts += 1
        if replace_attempts <= 2:
            failure = PermissionError("target is temporarily busy")
            failure.winerror = 5
            raise failure
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(time, "sleep", sleep_delays.append)
    monkeypatch.setattr(conversion, "_is_windows", lambda: True, raising=False)

    atomic_write_bytes(target, b"converted")

    assert replace_attempts == 3
    assert sleep_delays == [0.01, 0.02]
    assert target.read_bytes() == b"converted"
    assert list(tmp_path.iterdir()) == [target]
    assert conversion._TARGET_LOCKS == {}


def test_atomic_write_bounds_windows_replace_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.tflite"
    target.write_bytes(b"existing")
    failure = PermissionError("target stays busy")
    failure.winerror = 5
    replace_attempts = 0
    sleep_delays: list[float] = []

    def fail_replace(path: Path, destination: Path) -> Path:
        nonlocal replace_attempts
        replace_attempts += 1
        raise failure

    monkeypatch.setattr(Path, "replace", fail_replace)
    monkeypatch.setattr(time, "sleep", sleep_delays.append)
    monkeypatch.setattr(conversion, "_is_windows", lambda: True, raising=False)

    with pytest.raises(PermissionError, match="target stays busy") as caught:
        atomic_write_bytes(target, b"converted")

    assert caught.value is failure
    assert replace_attempts == 3
    assert sleep_delays == [0.01, 0.02]
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]
    assert conversion._TARGET_LOCKS == {}


def test_atomic_write_does_not_retry_other_replace_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.tflite"
    target.write_bytes(b"existing")
    failure = OSError("invalid replacement")
    replace_attempts = 0
    sleep_delays: list[float] = []

    def fail_replace(path: Path, destination: Path) -> Path:
        nonlocal replace_attempts
        replace_attempts += 1
        raise failure

    monkeypatch.setattr(Path, "replace", fail_replace)
    monkeypatch.setattr(time, "sleep", sleep_delays.append)

    with pytest.raises(OSError, match="invalid replacement") as caught:
        atomic_write_bytes(target, b"converted")

    assert caught.value is failure
    assert replace_attempts == 1
    assert sleep_delays == []
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]
    assert conversion._TARGET_LOCKS == {}


def test_target_lock_rolls_back_registry_when_acquire_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = SimpleNamespace(
        lock=SimpleNamespace(
            acquire=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
            release=lambda: pytest.fail("an unacquired lock must not be released"),
        ),
        users=0,
    )
    monkeypatch.setattr(conversion, "_TargetLockEntry", lambda: entry)

    try:
        with pytest.raises(KeyboardInterrupt):
            with conversion._serialized_target(tmp_path / "model.tflite"):
                pytest.fail("interrupted lock must not enter its critical section")

        assert entry.users == 0
        assert conversion._TARGET_LOCKS == {}
    finally:
        with conversion._TARGET_LOCKS_GUARD:
            conversion._TARGET_LOCKS.clear()


def test_target_lock_nests_windows_mutex_inside_process_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class ObservedLock:
        def acquire(self) -> None:
            events.append("process-acquire")

        def release(self) -> None:
            events.append("process-release")

    entry = SimpleNamespace(lock=ObservedLock(), users=0)

    @contextmanager
    def observed_windows_mutex(key: str):
        events.append("windows-acquire")
        try:
            yield
        finally:
            events.append("windows-release")

    monkeypatch.setattr(conversion, "_TargetLockEntry", lambda: entry)
    monkeypatch.setattr(conversion, "_is_windows", lambda: True)
    monkeypatch.setattr(conversion, "_windows_named_mutex", observed_windows_mutex)

    with conversion._serialized_target(tmp_path / "model.tflite"):
        events.append("body")

    assert events == [
        "process-acquire",
        "windows-acquire",
        "body",
        "windows-release",
        "process-release",
    ]
    assert conversion._TARGET_LOCKS == {}


def test_target_lock_cleans_registry_when_windows_mutex_wait_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel32 = _FakeKernel32(wait_result=0xFFFFFFFF)
    monkeypatch.setattr(conversion, "_is_windows", lambda: True)
    monkeypatch.setattr(conversion, "_load_kernel32", lambda: kernel32)
    monkeypatch.setattr(conversion, "_last_windows_error", lambda: 123)

    with pytest.raises(OSError, match="WaitForSingleObject.*123"):
        with conversion._serialized_target(tmp_path / "model.tflite"):
            pytest.fail("failed mutex wait must not enter")

    assert [call[0] for call in kernel32.calls] == ["create", "wait", "close"]
    assert conversion._TARGET_LOCKS == {}


def test_target_lock_preserves_body_error_when_mutex_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel32 = _FakeKernel32(release_result=False, close_result=False)
    monkeypatch.setattr(conversion, "_is_windows", lambda: True)
    monkeypatch.setattr(conversion, "_load_kernel32", lambda: kernel32)
    monkeypatch.setattr(conversion, "_last_windows_error", lambda: 5)
    primary = RuntimeError("body failed")

    with pytest.raises(RuntimeError, match="body failed") as caught:
        with conversion._serialized_target(tmp_path / "model.tflite"):
            raise primary

    assert caught.value is primary
    notes = getattr(caught.value, "__notes__", [])
    assert any("ReleaseMutex" in note for note in notes)
    assert any("CloseHandle" in note for note in notes)
    assert conversion._TARGET_LOCKS == {}


def test_atomic_write_serializes_concurrent_writers_to_same_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.tflite"
    payloads = [b"a" * 131_072, b"b" * 131_072]
    workers_ready = threading.Barrier(len(payloads))
    attempt_guard = threading.Lock()
    second_lock_attempted = threading.Event()
    underlying_lock = threading.Lock()
    lock_attempts = 0
    replace_calls = 0

    class ObservedLock:
        def acquire(self) -> None:
            nonlocal lock_attempts
            with attempt_guard:
                lock_attempts += 1
                if lock_attempts == 2:
                    second_lock_attempted.set()
            underlying_lock.acquire()

        def release(self) -> None:
            underlying_lock.release()

    entry = SimpleNamespace(lock=ObservedLock(), users=0)
    original_replace_with_retry = conversion._replace_with_retry

    def controlled_replace(path: Path, destination: Path) -> None:
        nonlocal replace_calls
        with attempt_guard:
            replace_calls += 1
            call_number = replace_calls
        if call_number == 1:
            assert second_lock_attempted.wait(timeout=5)
        original_replace_with_retry(path, destination)

    @contextmanager
    def no_cross_process_lock(key: str):
        yield

    monkeypatch.setattr(conversion, "_TargetLockEntry", lambda: entry)
    monkeypatch.setattr(conversion, "_cross_process_target_lock", no_cross_process_lock)
    monkeypatch.setattr(conversion, "_replace_with_retry", controlled_replace)

    def write(payload: bytes) -> None:
        workers_ready.wait(timeout=2)
        atomic_write_bytes(target, payload)

    with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        futures = [executor.submit(write, payload) for payload in payloads]
        for future in futures:
            future.result(timeout=10)

    assert second_lock_attempted.is_set()
    assert lock_attempts == 2
    assert replace_calls == 2
    assert target.read_bytes() in payloads
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
    assert conversion._TARGET_LOCKS == {}


@pytest.mark.skipif(os.name != "nt", reason="requires Windows named mutexes")
def test_atomic_write_serializes_real_windows_processes(tmp_path: Path) -> None:
    target = tmp_path / "model.tflite"
    payloads = [b"a" * 262_144, b"b" * 262_144]
    context = multiprocessing.get_context("spawn")
    workers_ready = context.Barrier(len(payloads))
    active_writers = context.Value("i", 0, lock=False)
    maximum_active_writers = context.Value("i", 0, lock=False)
    counter_guard = context.Lock()
    processes = [
        context.Process(
            target=_multiprocess_atomic_writer,
            args=(
                str(target),
                payload,
                workers_ready,
                active_writers,
                maximum_active_writers,
                counter_guard,
            ),
        )
        for payload in payloads
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert [process.exitcode for process in processes] == [0, 0]
    assert maximum_active_writers.value == 1
    assert active_writers.value == 0
    assert target.read_bytes() in payloads
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_atomic_write_replace_failure_preserves_target_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.tflite"
    target.write_bytes(b"existing")
    failure = PermissionError("replace denied")
    original_replace = Path.replace

    def fail_temp_replace(path: Path, destination: Path) -> Path:
        if path != target:
            raise failure
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_temp_replace)

    with pytest.raises(PermissionError, match="replace denied"):
        atomic_write_bytes(target, b"converted")

    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_preserves_primary_error_when_all_cleanup_steps_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.tflite"
    target.write_bytes(b"existing")
    primary = OSError("write setup failed")
    close_failure = OSError("close cleanup failed")
    unlink_failure = OSError("unlink cleanup failed")
    close_calls: list[int] = []
    unlink_calls: list[Path] = []
    original_close = os.close
    original_unlink = Path.unlink

    def fail_fdopen(descriptor: int, mode: str) -> object:
        raise primary

    def close_then_fail(descriptor: int) -> None:
        close_calls.append(descriptor)
        original_close(descriptor)
        raise close_failure

    def unlink_then_fail(path: Path, *, missing_ok: bool = False) -> None:
        unlink_calls.append(path)
        original_unlink(path, missing_ok=missing_ok)
        raise unlink_failure

    monkeypatch.setattr(conversion.os, "fdopen", fail_fdopen)
    monkeypatch.setattr(conversion.os, "close", close_then_fail)
    monkeypatch.setattr(Path, "unlink", unlink_then_fail)

    with pytest.raises(OSError, match="write setup failed") as caught:
        atomic_write_bytes(target, b"converted")

    assert caught.value is primary
    assert len(close_calls) == 1
    assert len(unlink_calls) == 1
    notes = getattr(caught.value, "__notes__", [])
    assert any("close cleanup failed" in note for note in notes)
    assert any("unlink cleanup failed" in note for note in notes)
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_preserves_stream_write_error_when_stream_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.tflite"
    target.write_bytes(b"existing")
    primary = OSError("stream write failed")
    close_failure = OSError("stream close failed")
    original_close = os.close

    class FailingStream:
        def __init__(self, descriptor: int) -> None:
            self.descriptor = descriptor

        def __enter__(self) -> FailingStream:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

        def write(self, data: bytes) -> None:
            raise primary

        def close(self) -> None:
            original_close(self.descriptor)
            raise close_failure

    monkeypatch.setattr(
        conversion.os,
        "fdopen",
        lambda descriptor, mode: FailingStream(descriptor),
    )

    with pytest.raises(OSError, match="stream write failed") as caught:
        atomic_write_bytes(target, b"converted")

    assert caught.value is primary
    notes = getattr(caught.value, "__notes__", [])
    assert any("stream close failed" in note for note in notes)
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]


class _FakeInterpreter:
    def __init__(
        self,
        inputs: list[dict[str, object]],
        outputs: list[dict[str, object]],
    ) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self.allocated = False

    def allocate_tensors(self) -> None:
        self.allocated = True

    def get_input_details(self) -> list[dict[str, object]]:
        return self._inputs

    def get_output_details(self) -> list[dict[str, object]]:
        return self._outputs


def _detail(name: str, shape: list[int], dtype: object) -> dict[str, object]:
    return {
        "name": name,
        "shape": np.asarray(shape, dtype=np.int32),
        "dtype": dtype,
        "index": 0,
        "shape_signature": np.asarray(shape, dtype=np.int32),
        "quantization": (0.0, 0),
        "quantization_parameters": {
            "scales": np.asarray([], dtype=np.float32),
            "zero_points": np.asarray([], dtype=np.int32),
            "quantized_dimension": 0,
        },
        "sparsity_parameters": {},
    }


def test_inspect_tflite_returns_faithful_immutable_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_detail = _detail(
        "serving_default_pixels:0", [1, 128, 128, 3], np.float32
    )
    input_detail["shape_signature"] = np.asarray(
        [1, -1, -1, 3], dtype=np.int32
    )
    fake = _FakeInterpreter(
        [input_detail],
        [_detail("StatefulPartitionedCall:0", [1, 1], np.float16)],
    )
    received: dict[str, object] = {}

    def make_interpreter(*, model_path: str) -> _FakeInterpreter:
        received["model_path"] = model_path
        return fake

    monkeypatch.setattr(conversion.tf.lite, "Interpreter", make_interpreter)
    path = tmp_path / "model.tflite"

    inspected = inspect_tflite(path)

    assert received == {"model_path": str(path)}
    assert fake.allocated is True
    assert inspected.input.name == "serving_default_pixels:0"
    assert inspected.input.shape == [1, 128, 128, 3]
    assert inspected.input.shape_signature == [1, -1, -1, 3]
    assert inspected.input.dtype == "float32"
    assert inspected.output.name == "StatefulPartitionedCall:0"
    assert inspected.output.shape == [1, 1]
    assert inspected.output.dtype == "float16"
    with pytest.raises(FrozenInstanceError):
        inspected.input.name = "mutated"
    returned_shape = inspected.input.shape
    returned_shape.append(99)
    assert inspected.input.shape == [1, 128, 128, 3]
    returned_signature = inspected.input.shape_signature
    returned_signature.append(-1)
    assert inspected.input.shape_signature == [1, -1, -1, 3]


def test_inspect_tflite_falls_back_to_shape_when_signature_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_detail = _detail("input", [1, 128, 128, 3], np.float32)
    output_detail = _detail("output", [1, 1], np.float32)
    del input_detail["shape_signature"]
    del output_detail["shape_signature"]
    fake = _FakeInterpreter([input_detail], [output_detail])
    monkeypatch.setattr(
        conversion.tf.lite,
        "Interpreter",
        lambda *, model_path: fake,
    )

    inspected = inspect_tflite(tmp_path / "model.tflite")

    assert inspected.input.shape_signature == [1, 128, 128, 3]
    assert inspected.output.shape_signature == [1, 1]


def test_inspected_tensor_copies_mutable_shape_inputs() -> None:
    shape = [1, 5, 10]
    signature = [1, 5, 10]

    inspected = conversion.InspectedTensor(
        "raw", "float32", shape, signature  # type: ignore[arg-type]
    )
    shape.append(99)
    signature.append(-1)

    assert inspected.shape == [1, 5, 10]
    assert inspected.shape_signature == [1, 5, 10]


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    (
        ([], [_detail("output", [1, 1], np.float32)]),
        (
            [_detail("input", [1, 128, 128, 3], np.float32)],
            [
                _detail("output_a", [1, 1], np.float32),
                _detail("output_b", [1, 1], np.float32),
            ],
        ),
    ),
)
def test_inspect_tflite_rejects_missing_or_multiple_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inputs: list[dict[str, object]],
    outputs: list[dict[str, object]],
) -> None:
    fake = _FakeInterpreter(inputs, outputs)
    monkeypatch.setattr(
        conversion.tf.lite,
        "Interpreter",
        lambda *, model_path: fake,
    )

    with pytest.raises(ModelConversionError, match="exactly one input and one output"):
        inspect_tflite(tmp_path / "model.tflite")


def test_inspect_tflite_wraps_bad_model_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.tflite"
    path.write_bytes(b"not a TensorFlow Lite model")

    with pytest.raises(ModelConversionError, match="inspect TFLite model") as caught:
        inspect_tflite(path)

    assert caught.value.__cause__ is not None


class _FakeYolo:
    def __init__(
        self,
        exported: object,
        *,
        task: object = "detect",
        names: object = None,
        export_error: BaseException | None = None,
    ) -> None:
        self.task = task
        self.names = {0: "lib"} if names is None else names
        self.exported = exported
        self.export_error = export_error
        self.export_calls: list[dict[str, object]] = []

    def export(self, **kwargs: object) -> object:
        self.export_calls.append(kwargs)
        if self.export_error is not None:
            raise self.export_error
        return self.exported


class _DefaultFakeYolo:
    def __init__(self, original_pt_path: str, exported_bytes: bytes) -> None:
        self.task = "detect"
        self.names = {0: "lib"}
        self.model = SimpleNamespace(pt_path=original_pt_path)
        self.exported_bytes = exported_bytes
        self.export_calls: list[dict[str, object]] = []
        self.export_anchors: list[Path] = []

    def export(self, **kwargs: object) -> str:
        self.export_calls.append(kwargs)
        anchor = Path(self.model.pt_path)
        self.export_anchors.append(anchor)
        export_directory = anchor.parent / f"{anchor.stem}_saved_model"
        export_directory.mkdir()
        artifact = export_directory / f"{anchor.stem}_float16.tflite"
        artifact.write_bytes(self.exported_bytes)
        return str(export_directory)


@pytest.mark.parametrize(
    ("precision", "half"),
    (("fp16", True), ("fp32", False)),
)
def test_detector_conversion_exports_exact_precision_contract(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    precision: str,
    half: bool,
) -> None:
    exported = tmp_path / "exported.tflite"
    exported.write_bytes(tiny_detector_tflite_bytes)
    source = Path("models") / "detector.pt"
    received_sources: list[object] = []
    export_result = str(exported) if precision == "fp16" else exported
    fake = _FakeYolo(export_result, names=MappingProxyType({0: "lib"}))

    def loader(received: object) -> _FakeYolo:
        received_sources.append(received)
        return fake

    target = tmp_path / "detector.tflite"
    spec = convert_detector(source, target, precision, loader=loader)

    assert received_sources == [str(source)]
    assert fake.export_calls == [
        {
            "format": "tflite",
            "imgsz": 640,
            "nms": False,
            "half": half,
            "int8": False,
        }
    ]
    assert spec.class_names == ["lib"]
    assert spec.input.shape == [1, 640, 640, 3]
    assert spec.output.shape == [1, 5, 10]
    assert spec.input.dtype == "float32"
    assert spec.output.dtype == "float32"
    assert target.read_bytes() == tiny_detector_tflite_bytes
    published = inspect_tflite(target)
    assert spec.input == published.input
    assert spec.output == published.output

    returned_names = spec.class_names
    returned_names.append("mutated")
    assert spec.class_names == ["lib"]
    with pytest.raises(FrozenInstanceError):
        spec.output = spec.input
    with pytest.raises(TypeError):
        type(spec)(spec.input, spec.output, _class_names=("other",))


def test_detector_conversion_default_loader_uses_yolo_with_string_path(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    source = tmp_path / "detector.pt"
    source.write_bytes(b"checkpoint")
    original_pt_path = str(source)
    fake = _DefaultFakeYolo(original_pt_path, tiny_detector_tflite_bytes)
    received: list[object] = []
    monkeypatch.setattr(
        ultralytics,
        "YOLO",
        lambda source: received.append(source) or fake,
    )

    convert_detector(source, tmp_path / "target.tflite", "fp16")

    assert received == [str(source)]
    assert fake.model.pt_path == original_pt_path
    assert len(fake.export_anchors) == 1
    anchor = fake.export_anchors[0]
    assert anchor.parent != source.parent
    assert not anchor.parent.exists()
    assert source.read_bytes() == b"checkpoint"
    assert fake.export_calls == [
        {
            "format": "tflite",
            "imgsz": 640,
            "nms": False,
            "half": True,
            "int8": False,
        }
    ]


def test_detector_conversion_restores_default_anchor_before_artifact_resolution(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    source = tmp_path / "detector.pt"
    source.write_bytes(b"checkpoint")
    original_pt_path = str(source)
    fake = _DefaultFakeYolo(original_pt_path, tiny_detector_tflite_bytes)
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    original_resolve = conversion._resolve_exported_tflite

    def observe_resolve(exported: object) -> Path:
        anchor = fake.export_anchors[-1]
        assert fake.model.pt_path == original_pt_path
        assert anchor.parent.is_dir()
        return original_resolve(exported)

    monkeypatch.setattr(conversion, "_resolve_exported_tflite", observe_resolve)

    convert_detector(source, tmp_path / "target.tflite", "fp16")

    assert fake.model.pt_path == original_pt_path


def test_detector_conversion_isolates_default_export_working_directory(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    source = tmp_path / "detector.pt"
    source.write_bytes(b"checkpoint")
    fake = _DefaultFakeYolo(str(source), tiny_detector_tflite_bytes)
    observed_cwds: list[Path] = []
    original_export = fake.export

    def export_with_side_effect(**kwargs: object) -> object:
        observed_cwds.append(Path.cwd())
        Path("calibration_image_sample_data_20x128x128x3_float32.npy").write_bytes(
            b"calibration"
        )
        return original_export(**kwargs)

    fake.export = export_with_side_effect  # type: ignore[method-assign]
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    target = tmp_path / "target.tflite"

    convert_detector(source, target, "fp16")

    assert Path.cwd() == caller.resolve()
    assert len(observed_cwds) == 1
    assert observed_cwds[0] != caller.resolve()
    assert not observed_cwds[0].exists()
    assert list(caller.iterdir()) == []
    assert target.read_bytes() == tiny_detector_tflite_bytes


def test_detector_conversion_restores_cwd_after_default_export_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    source = tmp_path / "detector.pt"
    failure = RuntimeError("export failed")
    fake = _FakeYolo(tmp_path / "unused.tflite")
    fake.model = SimpleNamespace(pt_path=str(source))
    observed_cwds: list[Path] = []

    def fail_export(**kwargs: object) -> object:
        observed_cwds.append(Path.cwd())
        Path("calibration.npy").write_bytes(b"calibration")
        raise failure

    fake.export = fail_export  # type: ignore[method-assign]
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    target = tmp_path / "target.tflite"

    with pytest.raises(ModelConversionError, match="export detector") as caught:
        convert_detector(source, target, "fp16")

    assert caught.value.__cause__ is failure
    assert Path.cwd() == caller.resolve()
    assert len(observed_cwds) == 1
    assert not observed_cwds[0].exists()
    assert list(caller.iterdir()) == []
    assert not target.exists()


@pytest.mark.parametrize("export_fails", (False, True))
def test_detector_conversion_restores_exact_export_environment(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
    export_fails: bool,
) -> None:
    import ultralytics

    monkeypatch.setenv("WATER_EXISTING", "original")
    monkeypatch.setenv("WATER_DELETED", "restore-me")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "caller-choice")
    monkeypatch.setenv("PATH", "caller-path")
    monkeypatch.delenv("TF_USE_LEGACY_KERAS", raising=False)
    monkeypatch.delenv("WATER_EXPORT_NEW", raising=False)
    snapshot = dict(os.environ)
    source = tmp_path / "detector.pt"
    fake = _DefaultFakeYolo(str(source), tiny_detector_tflite_bytes)
    original_export = fake.export
    failure = RuntimeError("export failed")

    def mutate_environment(**kwargs: object) -> object:
        os.environ["WATER_EXISTING"] = "changed"
        del os.environ["WATER_DELETED"]
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        os.environ["PATH"] = "export-path"
        os.environ["WATER_EXPORT_NEW"] = "new"
        if export_fails:
            raise failure
        return original_export(**kwargs)

    fake.export = mutate_environment  # type: ignore[method-assign]
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    target = tmp_path / "target.tflite"

    if export_fails:
        with pytest.raises(ModelConversionError, match="export detector") as caught:
            convert_detector(source, target, "fp16")
        assert caught.value.__cause__ is failure
        assert not target.exists()
    else:
        convert_detector(source, target, "fp16")
        assert target.read_bytes() == tiny_detector_tflite_bytes

    assert dict(os.environ) == snapshot


def test_detector_environment_restore_is_best_effort() -> None:
    delete_failure = OSError("delete failed")
    set_failure = OSError("set failed")

    class FailingEnvironment(dict[str, str]):
        def __delitem__(self, key: str) -> None:
            if key == "a-bad-extra":
                raise delete_failure
            super().__delitem__(key)

        def __setitem__(self, key: str, value: str) -> None:
            if key == "z-bad-set":
                raise set_failure
            super().__setitem__(key, value)

    snapshot = {
        "deleted": "restored",
        "keep": "original",
        "z-bad-set": "original",
    }
    environment = FailingEnvironment(
        {
            "a-bad-extra": "extra",
            "keep": "changed",
            "new-extra": "extra",
            "z-bad-set": "changed",
        }
    )

    with pytest.raises(OSError, match="delete failed") as caught:
        conversion._restore_process_environment(environment, snapshot)

    assert caught.value is delete_failure
    assert environment["a-bad-extra"] == "extra"
    assert "new-extra" not in environment
    assert environment["deleted"] == "restored"
    assert environment["keep"] == "original"
    assert environment["z-bad-set"] == "changed"
    assert any(
        "set failed" in note for note in getattr(caught.value, "__notes__", [])
    )


def test_detector_environment_restore_failure_prevents_publish_and_restores_cwd(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    source = tmp_path / "detector.pt"
    fake = _DefaultFakeYolo(str(source), tiny_detector_tflite_bytes)
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    restore_failure = OSError("environment restore failed")

    def fail_after_restore(environment: object, snapshot: dict[str, str]) -> None:
        current = dict(os.environ)
        for key in current.keys() - snapshot.keys():
            del os.environ[key]
        for key, value in snapshot.items():
            os.environ[key] = value
        raise restore_failure

    monkeypatch.setattr(
        conversion,
        "_restore_process_environment",
        fail_after_restore,
        raising=False,
    )
    target = tmp_path / "target.tflite"
    target.write_bytes(b"existing")

    with pytest.raises(ModelConversionError, match="restore detector export environment") as caught:
        convert_detector(source, target, "fp16")

    assert caught.value.__cause__ is restore_failure
    assert Path.cwd() == caller.resolve()
    assert target.read_bytes() == b"existing"
    assert len(fake.export_anchors) == 1
    assert not fake.export_anchors[0].parent.exists()


def test_detector_export_error_survives_environment_restore_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    source = tmp_path / "detector.pt"
    export_failure = RuntimeError("export failed")
    restore_failure = OSError("environment restore failed")
    fake = _FakeYolo(tmp_path / "unused.tflite", export_error=export_failure)
    fake.model = SimpleNamespace(pt_path=str(source))
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    original_restore = conversion._restore_process_environment

    def restore_then_fail(environment: object, snapshot: dict[str, str]) -> None:
        original_restore(environment, snapshot)
        raise restore_failure

    monkeypatch.setattr(
        conversion, "_restore_process_environment", restore_then_fail
    )

    with pytest.raises(ModelConversionError, match="export detector") as caught:
        convert_detector(source, tmp_path / "target.tflite", "fp16")

    assert caught.value.__cause__ is export_failure
    assert any(
        "environment restore failed" in note
        for note in getattr(caught.value, "__notes__", [])
    )
    assert Path.cwd() == caller.resolve()
    assert not (tmp_path / "target.tflite").exists()


def test_detector_export_environment_does_not_select_legacy_keras_loader(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    monkeypatch.delenv("TF_USE_LEGACY_KERAS", raising=False)
    source = tmp_path / "detector.pt"
    fake = _DefaultFakeYolo(str(source), tiny_detector_tflite_bytes)
    original_export = fake.export

    def enable_legacy_keras(**kwargs: object) -> object:
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        return original_export(**kwargs)

    fake.export = enable_legacy_keras  # type: ignore[method-assign]
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)

    convert_detector(source, tmp_path / "target.tflite", "fp16")

    lazy_loader_choice = (
        "legacy" if os.environ.get("TF_USE_LEGACY_KERAS") == "1" else "keras3"
    )
    assert lazy_loader_choice == "keras3"


def test_detector_conversion_default_loader_requires_safe_export_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    source = tmp_path / "detector.pt"
    source.write_bytes(b"checkpoint")
    fake = _FakeYolo(tmp_path / "must-not-export.tflite")
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)

    with pytest.raises(ModelConversionError, match="redirect detector export"):
        convert_detector(source, tmp_path / "target.tflite", "fp16")

    assert fake.export_calls == []
    assert source.read_bytes() == b"checkpoint"


def test_detector_conversion_wraps_broken_default_export_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    class BrokenInnerModel:
        @property
        def pt_path(self) -> str:
            raise RuntimeError("broken pt_path getter")

    fake = _FakeYolo(tmp_path / "must-not-export.tflite")
    fake.model = BrokenInnerModel()
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)

    with pytest.raises(ModelConversionError, match="redirect detector export") as caught:
        convert_detector(
            tmp_path / "detector.pt",
            tmp_path / "target.tflite",
            "fp16",
        )

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert fake.export_calls == []


def test_detector_conversion_restores_anchor_when_redirect_setter_mutates_then_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    source = tmp_path / "detector.pt"
    original_pt_path = str(source)

    class MutatingFailingSetter:
        def __init__(self) -> None:
            self.value = original_pt_path

        @property
        def pt_path(self) -> str:
            return self.value

        @pt_path.setter
        def pt_path(self, value: str) -> None:
            self.value = value
            if value != original_pt_path:
                raise RuntimeError("redirect setter failed after mutation")

    inner = MutatingFailingSetter()
    fake = _FakeYolo(tmp_path / "must-not-export.tflite")
    fake.model = inner
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)

    with pytest.raises(ModelConversionError, match="redirect detector export"):
        convert_detector(source, tmp_path / "target.tflite", "fp16")

    assert inner.pt_path == original_pt_path
    assert fake.export_calls == []


def test_detector_conversion_restore_failure_prevents_publish(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    source = tmp_path / "detector.pt"
    source.write_bytes(b"checkpoint")
    original_pt_path = str(source)
    restore_failure = OSError("restore failed")

    class RestoreFailingInner:
        def __init__(self) -> None:
            self.value = original_pt_path
            self.redirected = False

        @property
        def pt_path(self) -> str:
            return self.value

        @pt_path.setter
        def pt_path(self, value: str) -> None:
            if self.redirected and value == original_pt_path:
                raise restore_failure
            self.value = value
            self.redirected = value != original_pt_path

    inner = RestoreFailingInner()
    fake = _DefaultFakeYolo(original_pt_path, tiny_detector_tflite_bytes)
    fake.model = inner
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    target = tmp_path / "target.tflite"
    target.write_bytes(b"existing")

    with pytest.raises(ModelConversionError, match="restore detector model.pt_path") as caught:
        convert_detector(source, target, "fp16")

    assert caught.value.__cause__ is restore_failure
    assert target.read_bytes() == b"existing"
    assert len(fake.export_anchors) == 1
    assert not fake.export_anchors[0].parent.exists()


def test_detector_conversion_preserves_pt_restore_error_when_cwd_restore_fails(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    original_cwd = Path.cwd()
    source = tmp_path / "detector.pt"
    original_pt_path = str(source)
    pt_failure = OSError("pt_path restore failed")
    cwd_failure = OSError("cwd restore failed")

    class RestoreFailingInner:
        def __init__(self) -> None:
            self.value = original_pt_path
            self.redirected = False

        @property
        def pt_path(self) -> str:
            return self.value

        @pt_path.setter
        def pt_path(self, value: str) -> None:
            if self.redirected and value == original_pt_path:
                raise pt_failure
            self.value = value
            self.redirected = value != original_pt_path

    fake = _DefaultFakeYolo(original_pt_path, tiny_detector_tflite_bytes)
    fake.model = RestoreFailingInner()
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    original_chdir = os.chdir

    def fail_after_restore(path: object) -> None:
        original_chdir(path)
        if Path(path) == original_cwd:
            raise cwd_failure

    monkeypatch.setattr(conversion.os, "chdir", fail_after_restore)
    target = tmp_path / "target.tflite"
    target.write_bytes(b"existing")

    with pytest.raises(ModelConversionError, match="restore detector model.pt_path") as caught:
        convert_detector(source, target, "fp16")

    assert caught.value.__cause__ is pt_failure
    assert any(
        "cwd restore failed" in note
        for note in getattr(caught.value, "__notes__", [])
    )
    assert Path.cwd() == original_cwd
    assert target.read_bytes() == b"existing"


def test_detector_conversion_default_loader_restores_anchor_after_export_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    source = tmp_path / "detector.pt"
    source.write_bytes(b"checkpoint")
    original_pt_path = str(source)
    fake = _FakeYolo(
        tmp_path / "unused.tflite", export_error=RuntimeError("export failed")
    )
    fake.model = SimpleNamespace(pt_path=original_pt_path)
    observed_anchors: list[Path] = []
    original_export = fake.export

    def observe_export(**kwargs: object) -> object:
        observed_anchors.append(Path(fake.model.pt_path))
        return original_export(**kwargs)

    fake.export = observe_export  # type: ignore[method-assign]
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)

    with pytest.raises(ModelConversionError, match="export detector"):
        convert_detector(source, tmp_path / "target.tflite", "fp16")

    assert fake.model.pt_path == original_pt_path
    assert len(observed_anchors) == 1
    assert observed_anchors[0].parent != source.parent
    assert not observed_anchors[0].parent.exists()


def test_detector_conversion_default_loader_rejects_artifact_outside_workspace(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    source = tmp_path / "detector.pt"
    source.write_bytes(b"checkpoint")
    original_pt_path = str(source)
    external = tmp_path / "external.tflite"
    external.write_bytes(tiny_detector_tflite_bytes)
    fake = _FakeYolo(external)
    fake.model = SimpleNamespace(pt_path=original_pt_path)
    observed_anchors: list[Path] = []
    original_export = fake.export

    def observe_export(**kwargs: object) -> object:
        observed_anchors.append(Path(fake.model.pt_path))
        return original_export(**kwargs)

    fake.export = observe_export  # type: ignore[method-assign]
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)

    with pytest.raises(ModelConversionError, match="owned export workspace"):
        convert_detector(source, tmp_path / "target.tflite", "fp16")

    assert fake.model.pt_path == original_pt_path
    assert external.read_bytes() == tiny_detector_tflite_bytes
    assert len(observed_anchors) == 1
    assert not observed_anchors[0].parent.exists()


def test_detector_conversion_default_loader_uses_distinct_concurrent_workspaces(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    sources = [tmp_path / f"detector-{index}.pt" for index in range(2)]
    for source in sources:
        source.write_bytes(b"checkpoint")
    fakes = {
        str(source): _DefaultFakeYolo(str(source), tiny_detector_tflite_bytes)
        for source in sources
    }
    first_entered = threading.Event()
    allow_first_exit = threading.Event()
    second_loaded = threading.Event()
    second_entered = threading.Event()
    events: list[str] = []
    first_fake = fakes[str(sources[0])]
    second_fake = fakes[str(sources[1])]
    first_export = first_fake.export
    second_export = second_fake.export
    monkeypatch.delenv("WATER_CONCURRENT_EXPORT", raising=False)
    environment_snapshot = dict(os.environ)

    def controlled_first(**kwargs: object) -> object:
        events.append("first-start")
        os.environ["WATER_CONCURRENT_EXPORT"] = "first"
        first_entered.set()
        assert allow_first_exit.wait(timeout=5)
        result = first_export(**kwargs)
        events.append("first-end")
        return result

    def observed_second(**kwargs: object) -> object:
        events.append("second-start")
        assert "WATER_CONCURRENT_EXPORT" not in os.environ
        os.environ["WATER_CONCURRENT_EXPORT"] = "second"
        second_entered.set()
        result = second_export(**kwargs)
        events.append("second-end")
        return result

    first_fake.export = controlled_first  # type: ignore[method-assign]
    second_fake.export = observed_second  # type: ignore[method-assign]

    def load(source: str) -> _DefaultFakeYolo:
        if source == str(sources[1]):
            second_loaded.set()
        return fakes[source]

    monkeypatch.setattr(ultralytics, "YOLO", load)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            convert_detector, sources[0], tmp_path / "target-0.tflite", "fp16"
        )
        assert first_entered.wait(timeout=5)
        second_future = executor.submit(
            convert_detector, sources[1], tmp_path / "target-1.tflite", "fp16"
        )
        assert second_loaded.wait(timeout=5)
        assert not second_entered.is_set()
        allow_first_exit.set()
        futures = [first_future, second_future]
        for future in futures:
            future.result(timeout=15)

    workspace_parents = {
        fake.export_anchors[0].parent for fake in fakes.values()
    }
    assert len(workspace_parents) == 2
    assert all(not workspace.exists() for workspace in workspace_parents)
    assert [fake.model.pt_path for fake in fakes.values()] == [
        str(source) for source in sources
    ]
    assert events == ["first-start", "first-end", "second-start", "second-end"]
    assert dict(os.environ) == environment_snapshot


def test_detector_conversion_publishes_before_default_workspace_cleanup(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    source = tmp_path / "detector.pt"
    source.write_bytes(b"checkpoint")
    fake = _DefaultFakeYolo(str(source), tiny_detector_tflite_bytes)
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    target = tmp_path / "target.tflite"
    observed_workspaces: list[Path] = []

    def observe_publish(received_target: Path, data: bytes) -> None:
        anchor = fake.export_anchors[-1]
        workspace = anchor.parent
        artifact = (
            workspace
            / f"{anchor.stem}_saved_model"
            / f"{anchor.stem}_float16.tflite"
        )
        assert workspace.is_dir()
        assert artifact.read_bytes() == tiny_detector_tflite_bytes
        assert data == tiny_detector_tflite_bytes
        observed_workspaces.append(workspace)
        Path(received_target).write_bytes(data)

    monkeypatch.setattr(conversion, "atomic_write_bytes", observe_publish)

    spec = convert_detector(source, target, "fp16")

    assert spec.class_names == ["lib"]
    assert target.read_bytes() == tiny_detector_tflite_bytes
    assert len(observed_workspaces) == 1
    assert not observed_workspaces[0].exists()


def test_detector_conversion_cleans_default_workspace_after_publish_failure(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    source = tmp_path / "detector.pt"
    source.write_bytes(b"checkpoint")
    fake = _DefaultFakeYolo(str(source), tiny_detector_tflite_bytes)
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    target = tmp_path / "target.tflite"
    target.write_bytes(b"existing")
    failure = OSError("publish failed")
    observed_workspaces: list[Path] = []

    def fail_publish(received_target: Path, data: bytes) -> None:
        anchor = fake.export_anchors[-1]
        workspace = anchor.parent
        artifact = (
            workspace
            / f"{anchor.stem}_saved_model"
            / f"{anchor.stem}_float16.tflite"
        )
        assert workspace.is_dir()
        assert artifact.read_bytes() == tiny_detector_tflite_bytes
        observed_workspaces.append(workspace)
        raise failure

    monkeypatch.setattr(conversion, "atomic_write_bytes", fail_publish)

    with pytest.raises(ModelConversionError, match="publish converted detector") as caught:
        convert_detector(source, target, "fp16")

    assert caught.value.__cause__ is failure
    assert target.read_bytes() == b"existing"
    assert len(observed_workspaces) == 1
    assert not observed_workspaces[0].exists()


def test_detector_conversion_warns_but_returns_after_committed_cleanup_failure(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    cleanup_failure = OSError("cleanup failed after commit")
    owned_path = tmp_path / "owned-export"

    class FailingCleanupWorkspace:
        def __init__(self) -> None:
            self.path = owned_path
            self.path.mkdir()

        def cleanup(self) -> None:
            raise cleanup_failure

    source = tmp_path / "detector.pt"
    source.write_bytes(b"checkpoint")
    fake = _DefaultFakeYolo(str(source), tiny_detector_tflite_bytes)
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    monkeypatch.setattr(
        conversion,
        "_OwnedExportWorkspace",
        FailingCleanupWorkspace,
        raising=False,
    )
    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        conversion,
        "LOGGER",
        SimpleNamespace(warning=lambda *args: warnings.append(args)),
        raising=False,
    )
    target = tmp_path / "target.tflite"

    spec = convert_detector(source, target, "fp16")

    assert spec.class_names == ["lib"]
    assert target.read_bytes() == tiny_detector_tflite_bytes
    assert len(warnings) == 1
    assert warnings[0][1] == owned_path
    assert warnings[0][2] is cleanup_failure


def test_detector_conversion_preserves_primary_error_when_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    cleanup_failure = OSError("cleanup also failed")

    class FailingCleanupWorkspace:
        def __init__(self) -> None:
            self.path = tmp_path / "owned-export"
            self.path.mkdir()

        def cleanup(self) -> None:
            raise cleanup_failure

    source = tmp_path / "detector.pt"
    export_failure = RuntimeError("export failed")
    fake = _FakeYolo(tmp_path / "unused.tflite", export_error=export_failure)
    fake.model = SimpleNamespace(pt_path=str(source))
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    monkeypatch.setattr(
        conversion,
        "_OwnedExportWorkspace",
        FailingCleanupWorkspace,
        raising=False,
    )

    with pytest.raises(ModelConversionError, match="export detector") as caught:
        convert_detector(source, tmp_path / "target.tflite", "fp16")

    assert caught.value.__cause__ is export_failure
    assert any(
        "cleanup also failed" in note
        for note in getattr(caught.value, "__notes__", [])
    )


def test_uncommitted_workspace_cleanup_error_is_exposed(tmp_path: Path) -> None:
    failure = OSError("cleanup failed without primary")
    workspace = SimpleNamespace(
        path=tmp_path / "owned-export",
        cleanup=lambda: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(OSError, match="cleanup failed without primary") as caught:
        conversion._cleanup_owned_export_workspace(
            workspace,
            primary_error=None,
            committed=False,
        )

    assert caught.value is failure


def test_detector_cwd_lock_interruption_does_not_release_unacquired_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptedLock:
        def acquire(self) -> None:
            raise KeyboardInterrupt()

        def release(self) -> None:
            pytest.fail("unacquired CWD lock must not be released")

    monkeypatch.setattr(
        conversion, "_DETECTOR_CWD_LOCK", InterruptedLock(), raising=False
    )

    with pytest.raises(KeyboardInterrupt):
        with conversion._detector_export_working_directory(tmp_path):
            pytest.fail("interrupted lock must not enter")


def test_detector_cwd_restore_and_release_failures_preserve_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_cwd = Path.cwd()
    primary = RuntimeError("body failed")
    restore_failure = OSError("cwd restore failed")
    release_failure = OSError("cwd lock release failed")
    original_chdir = os.chdir

    class FailingReleaseLock:
        def acquire(self) -> None:
            return None

        def release(self) -> None:
            raise release_failure

    def fail_after_restore(path: object) -> None:
        original_chdir(path)
        if Path(path) == original_cwd:
            raise restore_failure

    monkeypatch.setattr(
        conversion, "_DETECTOR_CWD_LOCK", FailingReleaseLock(), raising=False
    )
    monkeypatch.setattr(conversion.os, "chdir", fail_after_restore)

    with pytest.raises(RuntimeError, match="body failed") as caught:
        with conversion._detector_export_working_directory(tmp_path):
            raise primary

    assert caught.value is primary
    notes = getattr(caught.value, "__notes__", [])
    assert any("cwd restore failed" in note for note in notes)
    assert any("cwd lock release failed" in note for note in notes)
    assert Path.cwd() == original_cwd


def test_detector_conversion_wraps_cwd_lookup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    failure = OSError("cannot read cwd")
    source = tmp_path / "detector.pt"
    fake = _DefaultFakeYolo(str(source), b"unused")
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    monkeypatch.setattr(
        conversion,
        "_resolved_current_directory",
        lambda: (_ for _ in ()).throw(failure),
        raising=False,
    )

    with pytest.raises(ModelConversionError, match="read current working directory") as caught:
        convert_detector(source, tmp_path / "target.tflite", "fp16")

    assert caught.value.__cause__ is failure
    assert fake.export_calls == []


def test_detector_conversion_wraps_chdir_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ultralytics

    failure = OSError("cannot change cwd")
    source = tmp_path / "detector.pt"
    fake = _DefaultFakeYolo(str(source), b"unused")
    monkeypatch.setattr(ultralytics, "YOLO", lambda received: fake)
    monkeypatch.setattr(
        conversion.os,
        "chdir",
        lambda path: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(ModelConversionError, match="change detector export working directory") as caught:
        convert_detector(source, tmp_path / "target.tflite", "fp16")

    assert caught.value.__cause__ is failure
    assert fake.export_calls == []


def test_detector_conversion_rejects_precision_before_loading(
    tmp_path: Path,
) -> None:
    target = tmp_path / "detector.tflite"
    target.write_bytes(b"existing")

    with pytest.raises(ModelConversionError, match="unsupported precision: int8"):
        convert_detector(
            Path("detector.pt"),
            target,
            "int8",
            loader=lambda source: pytest.fail("invalid precision must not load"),
        )

    assert target.read_bytes() == b"existing"


def test_detector_conversion_rejects_target_with_wrong_suffix_before_loading(
    tmp_path: Path,
) -> None:
    target = tmp_path / "detector.bin"

    with pytest.raises(ModelConversionError, match="target.*.tflite"):
        convert_detector(
            Path("detector.pt"),
            target,
            "fp16",
            loader=lambda source: pytest.fail("invalid target must not load"),
        )

    assert not target.exists()


def test_detector_conversion_rejects_source_as_target_before_loading(
    tmp_path: Path,
) -> None:
    source = tmp_path / "detector.tflite"
    source.write_bytes(b"source")

    with pytest.raises(ModelConversionError, match="same file"):
        convert_detector(
            source,
            source,
            "fp16",
            loader=lambda received: pytest.fail("alias must not load"),
        )

    assert source.read_bytes() == b"source"


def test_detector_conversion_rejects_normalized_source_target_alias(
    tmp_path: Path,
) -> None:
    source = tmp_path / "detector.tflite"
    source.write_bytes(b"source")
    (tmp_path / "alias").mkdir()
    target = tmp_path / "alias" / ".." / source.name

    with pytest.raises(ModelConversionError, match="same file"):
        convert_detector(
            source,
            target,
            "fp16",
            loader=lambda received: pytest.fail("alias must not load"),
        )

    assert source.read_bytes() == b"source"


@pytest.mark.skipif(os.name != "nt", reason="Windows path case normalization")
def test_detector_conversion_rejects_case_only_source_target_alias(
    tmp_path: Path,
) -> None:
    source = tmp_path / "detector.tflite"
    source.write_bytes(b"source")
    target = tmp_path / "DETECTOR.TFLITE"

    with pytest.raises(ModelConversionError, match="same file"):
        convert_detector(
            source,
            target,
            "fp16",
            loader=lambda received: pytest.fail("alias must not load"),
        )

    assert source.read_bytes() == b"source"


def test_detector_conversion_rejects_hardlink_source_target_alias(
    tmp_path: Path,
) -> None:
    source = tmp_path / "detector.pt"
    target = tmp_path / "detector.tflite"
    source.write_bytes(b"source")
    os.link(source, target)

    with pytest.raises(ModelConversionError, match="same file"):
        convert_detector(
            source,
            target,
            "fp16",
            loader=lambda received: pytest.fail("alias must not load"),
        )

    assert source.read_bytes() == b"source"
    assert target.read_bytes() == b"source"


def test_detector_conversion_rejects_symlink_source_target_alias(
    tmp_path: Path,
) -> None:
    source = tmp_path / "detector.pt"
    target = tmp_path / "detector.tflite"
    source.write_bytes(b"source")
    os.symlink(source, target)

    with pytest.raises(ModelConversionError, match="same file"):
        convert_detector(
            source,
            target,
            "fp16",
            loader=lambda received: pytest.fail("alias must not load"),
        )

    assert source.read_bytes() == b"source"


@pytest.mark.parametrize(
    ("task", "names", "message"),
    (
        ("segment", {0: "lib"}, "detection task"),
        ("detect", {0: "other"}, "class names"),
        ("detect", {1: "lib"}, "class names"),
        ("detect", {0: "lib", 1: "other"}, "class names"),
        ("detect", ["lib"], "class names"),
        ("detect", {False: "lib"}, "class names"),
    ),
)
def test_detector_conversion_rejects_wrong_model_contract_before_export(
    tmp_path: Path,
    task: object,
    names: object,
    message: str,
) -> None:
    target = tmp_path / "detector.tflite"
    target.write_bytes(b"existing")
    fake = _FakeYolo(tmp_path / "unused.tflite", task=task, names=names)

    with pytest.raises(ModelConversionError, match=message):
        convert_detector(
            Path("detector.pt"), target, "fp16", loader=lambda source: fake
        )

    assert fake.export_calls == []
    assert target.read_bytes() == b"existing"


def test_detector_conversion_wraps_loader_failure_with_cause(
    tmp_path: Path,
) -> None:
    failure = OSError("cannot open checkpoint")

    with pytest.raises(ModelConversionError, match="load detector") as caught:
        convert_detector(
            Path("detector.pt"),
            tmp_path / "detector.tflite",
            "fp16",
            loader=lambda source: (_ for _ in ()).throw(failure),
        )

    assert caught.value.__cause__ is failure


def test_detector_conversion_wraps_export_failure_and_preserves_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "detector.tflite"
    target.write_bytes(b"existing")
    failure = RuntimeError("export failed")
    fake = _FakeYolo(tmp_path / "unused.tflite", export_error=failure)

    with pytest.raises(ModelConversionError, match="export detector") as caught:
        convert_detector(
            Path("detector.pt"), target, "fp16", loader=lambda source: fake
        )

    assert caught.value.__cause__ is failure
    assert target.read_bytes() == b"existing"


@pytest.mark.parametrize("returned", ("missing.tflite", 123))
def test_detector_conversion_rejects_missing_or_invalid_export_result(
    tmp_path: Path,
    returned: object,
) -> None:
    target = tmp_path / "detector.tflite"
    target.write_bytes(b"existing")
    if isinstance(returned, str):
        returned = tmp_path / returned

    with pytest.raises(ModelConversionError, match="exported TFLite"):
        convert_detector(
            Path("detector.pt"),
            target,
            "fp16",
            loader=lambda source: _FakeYolo(returned),
        )

    assert target.read_bytes() == b"existing"


def test_detector_conversion_rejects_exported_file_with_wrong_suffix(
    tmp_path: Path,
) -> None:
    exported = tmp_path / "detector.bin"
    exported.write_bytes(b"not a TFLite path")
    target = tmp_path / "detector.tflite"
    target.write_bytes(b"existing")

    with pytest.raises(ModelConversionError, match="not a .tflite file"):
        convert_detector(
            Path("detector.pt"),
            target,
            "fp16",
            loader=lambda source: _FakeYolo(exported),
        )

    assert exported.read_bytes() == b"not a TFLite path"
    assert target.read_bytes() == b"existing"


@pytest.mark.parametrize("candidate_count", (0, 2))
def test_detector_conversion_rejects_directory_without_unique_tflite(
    tmp_path: Path,
    candidate_count: int,
) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    for index in range(candidate_count):
        (export_directory / f"detector-{index}.tflite").write_bytes(b"candidate")
    target = tmp_path / "detector.tflite"
    target.write_bytes(b"existing")

    with pytest.raises(ModelConversionError, match="exactly one.*TFLite"):
        convert_detector(
            Path("detector.pt"),
            target,
            "fp16",
            loader=lambda source: _FakeYolo(export_directory),
        )

    assert export_directory.is_dir()
    assert len(list(export_directory.iterdir())) == candidate_count
    assert target.read_bytes() == b"existing"


def test_detector_conversion_rejects_nested_only_tflite_directory(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
) -> None:
    export_directory = tmp_path / "export"
    nested = export_directory / "saved_model"
    nested.mkdir(parents=True)
    exported = nested / "detector.tflite"
    exported.write_bytes(tiny_detector_tflite_bytes)

    target = tmp_path / "detector.tflite"
    target.write_bytes(b"existing")

    with pytest.raises(ModelConversionError, match="exactly one.*TFLite"):
        convert_detector(
            Path("detector.pt"),
            target,
            "fp16",
            loader=lambda source: _FakeYolo(export_directory),
        )

    assert exported.read_bytes() == tiny_detector_tflite_bytes
    assert target.read_bytes() == b"existing"


def test_detector_conversion_bounds_directory_candidate_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    candidates = [
        export_directory / f"detector-{index}.tflite" for index in range(3)
    ]
    for candidate in candidates:
        candidate.write_bytes(b"candidate")
    yielded: list[Path] = []
    original_iterdir = Path.iterdir

    def bounded_iterdir(path: Path):
        if path != export_directory:
            yield from original_iterdir(path)
            return
        for index, candidate in enumerate(candidates):
            if index == 2:
                raise AssertionError("directory scan consumed a third candidate")
            yielded.append(candidate)
            yield candidate

    monkeypatch.setattr(Path, "iterdir", bounded_iterdir)

    with pytest.raises(ModelConversionError, match="exactly one.*TFLite"):
        convert_detector(
            Path("detector.pt"),
            tmp_path / "detector.tflite",
            "fp16",
            loader=lambda source: _FakeYolo(export_directory),
        )

    assert yielded == candidates[:2]


@pytest.mark.parametrize("candidate_first", (False, True))
def test_detector_conversion_bounds_total_export_directory_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_first: bool,
) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    candidate = export_directory / "detector.tflite"
    candidate.write_bytes(b"candidate")
    limit = 3
    monkeypatch.setattr(
        conversion,
        "_MAX_EXPORT_DIRECTORY_ENTRIES",
        limit,
        raising=False,
    )
    yielded: list[Path] = []

    def many_entries(path: Path):
        if candidate_first:
            yielded.append(candidate)
            yield candidate
        for index in range(5000):
            if len(yielded) == limit + 1:
                pytest.fail("directory scan consumed beyond limit + 1 entries")
            entry = export_directory / f"metadata-{index}.txt"
            yielded.append(entry)
            yield entry

    monkeypatch.setattr(Path, "iterdir", many_entries)

    with pytest.raises(ModelConversionError, match="export directory has too many entries"):
        convert_detector(
            Path("detector.pt"),
            tmp_path / "target.tflite",
            "fp16",
            loader=lambda source: _FakeYolo(export_directory),
        )

    assert len(yielded) == limit + 1


def test_detector_conversion_accepts_unique_candidate_within_directory_limit(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    (export_directory / "metadata.txt").write_text("metadata", encoding="utf-8")
    candidate = export_directory / "detector.tflite"
    candidate.write_bytes(tiny_detector_tflite_bytes)
    monkeypatch.setattr(conversion, "_MAX_EXPORT_DIRECTORY_ENTRIES", 2, raising=False)
    target = tmp_path / "target.tflite"

    convert_detector(
        Path("detector.pt"),
        target,
        "fp16",
        loader=lambda source: _FakeYolo(export_directory),
    )

    assert target.read_bytes() == tiny_detector_tflite_bytes


@pytest.mark.parametrize("marked", ("directory", "candidate"))
def test_detector_conversion_rejects_reparse_export_paths(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
    marked: str,
) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    candidate = export_directory / "detector.tflite"
    candidate.write_bytes(tiny_detector_tflite_bytes)
    marked_path = export_directory if marked == "directory" else candidate
    monkeypatch.setattr(
        conversion,
        "_is_link_or_reparse",
        lambda path: Path(path) == marked_path,
        raising=False,
    )
    target = tmp_path / "target.tflite"
    target.write_bytes(b"existing")

    with pytest.raises(ModelConversionError, match="link or reparse"):
        convert_detector(
            Path("detector.pt"),
            target,
            "fp16",
            loader=lambda source: _FakeYolo(export_directory),
        )

    assert target.read_bytes() == b"existing"


def test_detector_conversion_rejects_direct_symlink_artifact(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
) -> None:
    actual = tmp_path / "actual.tflite"
    actual.write_bytes(tiny_detector_tflite_bytes)
    exported = tmp_path / "exported.tflite"
    os.symlink(actual, exported)
    target = tmp_path / "target.tflite"
    target.write_bytes(b"existing")

    with pytest.raises(ModelConversionError, match="link or reparse"):
        convert_detector(
            Path("detector.pt"),
            target,
            "fp16",
            loader=lambda source: _FakeYolo(exported),
        )

    assert actual.read_bytes() == tiny_detector_tflite_bytes
    assert exported.is_symlink()
    assert target.read_bytes() == b"existing"


def test_detector_conversion_rejects_candidate_outside_returned_directory(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_directory = tmp_path / "export"
    export_directory.mkdir()
    outside = tmp_path / "outside.tflite"
    outside.write_bytes(tiny_detector_tflite_bytes)
    original_iterdir = Path.iterdir

    def escaped_iterdir(path: Path):
        if path == export_directory:
            return iter((outside,))
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", escaped_iterdir)

    with pytest.raises(ModelConversionError, match="outside export directory"):
        convert_detector(
            Path("detector.pt"),
            tmp_path / "target.tflite",
            "fp16",
            loader=lambda source: _FakeYolo(export_directory),
        )


def test_detector_conversion_rejects_artifact_aliasing_existing_target(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
) -> None:
    target = tmp_path / "detector.tflite"
    target.write_bytes(tiny_detector_tflite_bytes)
    original = target.read_bytes()

    with pytest.raises(ModelConversionError, match="artifact.*target.*same file"):
        convert_detector(
            Path("detector.pt"),
            target,
            "fp16",
            loader=lambda source: _FakeYolo(target),
        )

    assert target.read_bytes() == original


def test_detector_conversion_rejects_artifact_hardlinked_to_source(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
) -> None:
    source = tmp_path / "detector.pt"
    source.write_bytes(tiny_detector_tflite_bytes)
    exported = tmp_path / "exported.tflite"
    os.link(source, exported)
    target = tmp_path / "target.tflite"

    with pytest.raises(ModelConversionError, match="artifact.*source.*same file"):
        convert_detector(
            source,
            target,
            "fp16",
            loader=lambda received: _FakeYolo(exported),
        )

    assert source.read_bytes() == tiny_detector_tflite_bytes
    assert not target.exists()


def test_detector_conversion_rejects_bad_tflite_before_replacing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "detector.pt"
    source.write_bytes(b"checkpoint")
    exported = tmp_path / "exported.tflite"
    exported.write_bytes(b"not a TFLite model")
    target = tmp_path / "detector.tflite"
    target.write_bytes(b"existing")

    with pytest.raises(ModelConversionError, match="inspect TFLite model") as caught:
        convert_detector(
            source, target, "fp16", loader=lambda received: _FakeYolo(exported)
        )

    assert caught.value.__cause__ is not None
    assert source.read_bytes() == b"checkpoint"
    assert exported.read_bytes() == b"not a TFLite model"
    assert target.read_bytes() == b"existing"


def test_detector_conversion_publishes_the_exact_bytes_that_were_inspected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = tmp_path / "exported.tflite"
    exported.write_bytes(b"validated bytes")
    target = tmp_path / "detector.tflite"
    target.write_bytes(b"existing")
    inspection = conversion.TFLiteInspection(
        input=conversion.InspectedTensor(
            "images", "float32", (1, 640, 640, 3)
        ),
        output=conversion.InspectedTensor("raw", "float32", (1, 5, 10)),
    )
    inspected_paths: list[Path] = []

    def inspect_then_change(path: Path) -> conversion.TFLiteInspection:
        inspected_path = Path(path)
        inspected_paths.append(inspected_path)
        assert inspected_path.read_bytes() == b"validated bytes"
        inspected_path.write_bytes(b"changed after inspection")
        return inspection

    monkeypatch.setattr(conversion, "inspect_tflite", inspect_then_change)

    convert_detector(
        Path("detector.pt"),
        target,
        "fp16",
        loader=lambda source: _FakeYolo(exported),
    )

    assert target.read_bytes() == b"validated bytes"
    assert exported.read_bytes() == b"validated bytes"
    assert len(inspected_paths) == 1
    assert not inspected_paths[0].exists()


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "message"),
    (
        ([1, 320, 320, 3], [1, 5, 10], "detector input"),
        ([1, 640, 640, 3], [1, 6, 10], "detector output"),
        ([1, 640, 640, 3], [1, 5, 0], "detector output"),
    ),
)
def test_detector_conversion_rejects_bad_tensor_contract_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_shape: list[int],
    output_shape: list[int],
    message: str,
) -> None:
    exported = tmp_path / "exported.tflite"
    exported.write_bytes(b"opaque exporter result")
    target = tmp_path / "detector.tflite"
    target.write_bytes(b"existing")
    inspection = conversion.TFLiteInspection(
        input=conversion.InspectedTensor("images", "float32", tuple(input_shape)),
        output=conversion.InspectedTensor("raw", "float32", tuple(output_shape)),
    )
    monkeypatch.setattr(conversion, "inspect_tflite", lambda path: inspection)

    with pytest.raises(ModelConversionError, match=message):
        convert_detector(
            Path("detector.pt"),
            target,
            "fp16",
            loader=lambda source: _FakeYolo(exported),
        )

    assert exported.read_bytes() == b"opaque exporter result"
    assert target.read_bytes() == b"existing"


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype"),
    (
        ("int64", "float32"),
        ("uint8", "float32"),
        ("float16", "float32"),
        ("float32", "int64"),
        ("float32", "uint8"),
        ("float32", "float16"),
    ),
)
def test_detector_conversion_rejects_non_float32_io_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_dtype: str,
    output_dtype: str,
) -> None:
    exported = tmp_path / "exported.tflite"
    exported.write_bytes(b"opaque exporter result")
    target = tmp_path / "detector.tflite"
    target.write_bytes(b"existing")
    inspection = conversion.TFLiteInspection(
        input=conversion.InspectedTensor(
            "images", input_dtype, (1, 640, 640, 3)
        ),
        output=conversion.InspectedTensor("raw", output_dtype, (1, 5, 10)),
    )
    monkeypatch.setattr(conversion, "inspect_tflite", lambda path: inspection)

    with pytest.raises(ModelConversionError, match="float32"):
        convert_detector(
            Path("detector.pt"),
            target,
            "fp16",
            loader=lambda source: _FakeYolo(exported),
        )

    assert target.read_bytes() == b"existing"


@pytest.mark.parametrize(
    ("input_signature", "output_signature", "message"),
    (
        ((1, -1, 640, 3), (1, 5, 10), "detector input"),
        ((1, 640, 640, 3), (1, 5, -1), "detector output"),
    ),
)
def test_detector_conversion_rejects_dynamic_shape_signatures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_signature: tuple[int, ...],
    output_signature: tuple[int, ...],
    message: str,
) -> None:
    exported = tmp_path / "exported.tflite"
    exported.write_bytes(b"opaque exporter result")
    target = tmp_path / "detector.tflite"
    target.write_bytes(b"existing")
    inspection = conversion.TFLiteInspection(
        input=conversion.InspectedTensor(
            "images", "float32", (1, 640, 640, 3), input_signature
        ),
        output=conversion.InspectedTensor(
            "raw", "float32", (1, 5, 10), output_signature
        ),
    )
    monkeypatch.setattr(conversion, "inspect_tflite", lambda path: inspection)

    with pytest.raises(ModelConversionError, match=message):
        convert_detector(
            Path("detector.pt"),
            target,
            "fp16",
            loader=lambda source: _FakeYolo(exported),
        )

    assert target.read_bytes() == b"existing"


def test_detector_conversion_never_deletes_source_or_exported_artifact(
    tmp_path: Path,
    tiny_detector_tflite_bytes: bytes,
) -> None:
    source_directory = tmp_path / "models"
    source_directory.mkdir()
    source = source_directory / "detector.pt"
    source.write_bytes(b"checkpoint")
    exported = source_directory / "detector-fp16.tflite"
    exported.write_bytes(tiny_detector_tflite_bytes)

    convert_detector(
        source,
        tmp_path / "published" / "detector.tflite",
        "fp16",
        loader=lambda received: _FakeYolo(exported),
    )

    assert source_directory.is_dir()
    assert source.read_bytes() == b"checkpoint"
    assert exported.read_bytes() == tiny_detector_tflite_bytes
