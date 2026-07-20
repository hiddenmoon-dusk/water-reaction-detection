"""TensorFlow Lite conversion and inspection helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
import hashlib
import logging
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from typing import Iterator

import numpy as np
import tensorflow as tf


LOGGER = logging.getLogger(__name__)
_MAX_EXPORT_DIRECTORY_ENTRIES = 256


class ModelConversionError(RuntimeError):
    """Raised when a model cannot be converted or inspected safely."""


def _normalized_path_key(path: Path) -> str:
    return os.path.normcase(os.fspath(path.resolve(strict=False)))


def _is_windows() -> bool:
    return os.name == "nt"


def _supports_parent_directory_sync() -> bool:
    return not _is_windows()


class _TargetLockEntry:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0


_TARGET_LOCKS: dict[str, _TargetLockEntry] = {}
_TARGET_LOCKS_GUARD = threading.Lock()
_DETECTOR_CWD_LOCK = threading.Lock()
_REPLACE_RETRY_DELAYS = (0.01, 0.02)
_MUTEX_WAIT_TIMEOUT_MS = 30_000
_WAIT_OBJECT_0 = 0x00000000
_WAIT_ABANDONED = 0x00000080
_WAIT_TIMEOUT = 0x00000102
_WAIT_FAILED = 0xFFFFFFFF


def _windows_mutex_name(normalized_key: str) -> str:
    digest = hashlib.sha256(normalized_key.encode("utf-8")).hexdigest()
    return f"Local\\water-models-{digest}"


def _last_windows_error() -> int:
    return ctypes.get_last_error()


def _windows_api_error(operation: str) -> OSError:
    error_code = _last_windows_error()
    return OSError(
        error_code,
        f"{operation} failed with Windows error {error_code}",
    )


def _load_kernel32() -> object:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _configure_kernel32(kernel32)
    return kernel32


def _configure_kernel32(kernel32: object) -> None:
    kernel32.CreateMutexW.argtypes = [
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL


def _cleanup_windows_mutex(
    kernel32: object,
    handle: object,
    *,
    acquired: bool,
    primary_error: BaseException | None,
) -> None:
    cleanup_errors: list[tuple[str, BaseException]] = []
    if acquired:
        try:
            released = kernel32.ReleaseMutex(handle)
        except BaseException as error:
            cleanup_errors.append(("ReleaseMutex", error))
        else:
            if not released:
                cleanup_errors.append(
                    ("ReleaseMutex", _windows_api_error("ReleaseMutex"))
                )
    try:
        closed = kernel32.CloseHandle(handle)
    except BaseException as error:
        cleanup_errors.append(("CloseHandle", error))
    else:
        if not closed:
            cleanup_errors.append(("CloseHandle", _windows_api_error("CloseHandle")))

    if not cleanup_errors:
        return
    if primary_error is not None:
        for operation, error in cleanup_errors:
            primary_error.add_note(f"{operation} cleanup failed: {error!r}")
        return
    operation, cleanup_error = cleanup_errors[0]
    cleanup_error.add_note(f"during {operation} cleanup")
    for later_operation, later_error in cleanup_errors[1:]:
        cleanup_error.add_note(
            f"{later_operation} cleanup also failed: {later_error!r}"
        )
    raise cleanup_error


@contextmanager
def _windows_named_mutex(
    normalized_key: str, *, kernel32: object | None = None
) -> Iterator[None]:
    if kernel32 is None:
        kernel32 = _load_kernel32()
    handle = kernel32.CreateMutexW(
        None,
        False,
        _windows_mutex_name(normalized_key),
    )
    if not handle:
        raise _windows_api_error("CreateMutexW")

    acquired = False
    try:
        wait_result = kernel32.WaitForSingleObject(handle, _MUTEX_WAIT_TIMEOUT_MS)
        if wait_result in (_WAIT_OBJECT_0, _WAIT_ABANDONED):
            acquired = True
        elif wait_result == _WAIT_TIMEOUT:
            raise TimeoutError(
                "timed out after 30.0 seconds waiting for Windows model write mutex"
            )
        elif wait_result == _WAIT_FAILED:
            raise _windows_api_error("WaitForSingleObject")
        else:
            raise OSError(f"unexpected Windows mutex wait result: {wait_result}")
        yield
    except BaseException as primary_error:
        _cleanup_windows_mutex(
            kernel32,
            handle,
            acquired=acquired,
            primary_error=primary_error,
        )
        raise
    else:
        _cleanup_windows_mutex(
            kernel32,
            handle,
            acquired=acquired,
            primary_error=None,
        )


@contextmanager
def _cross_process_target_lock(normalized_key: str) -> Iterator[None]:
    if _is_windows():
        with _windows_named_mutex(normalized_key):
            yield
    else:
        yield


@contextmanager
def _serialized_target(target: Path) -> Iterator[None]:
    key = _normalized_path_key(target)
    with _TARGET_LOCKS_GUARD:
        entry = _TARGET_LOCKS.get(key)
        if entry is None:
            entry = _TargetLockEntry()
            _TARGET_LOCKS[key] = entry
        entry.users += 1

    acquired = False
    try:
        entry.lock.acquire()
        acquired = True
        with _cross_process_target_lock(key):
            yield
    finally:
        try:
            if acquired:
                entry.lock.release()
        finally:
            with _TARGET_LOCKS_GUARD:
                entry.users -= 1
                if entry.users == 0 and _TARGET_LOCKS.get(key) is entry:
                    del _TARGET_LOCKS[key]


def _ensure_distinct_files(source: Path, target: Path) -> None:
    if _normalized_path_key(source) == _normalized_path_key(target):
        raise ModelConversionError(
            f"model source and target refer to the same file: {source}"
        )

    try:
        aliases_existing_file = (
            source.exists() and target.exists() and os.path.samefile(source, target)
        )
    except OSError:
        aliases_existing_file = False
    if aliases_existing_file:
        raise ModelConversionError(
            "model source and target refer to the same file: "
            f"{source} and {target}"
        )


@dataclass(frozen=True)
class InspectedTensor:
    """Immutable metadata for one TensorFlow Lite tensor."""

    name: str
    dtype: str
    _shape: tuple[int, ...] = field(repr=False)
    _shape_signature: tuple[int, ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        shape = tuple(int(dimension) for dimension in self._shape)
        signature = (
            shape
            if self._shape_signature is None
            else tuple(int(dimension) for dimension in self._shape_signature)
        )
        object.__setattr__(self, "_shape", shape)
        object.__setattr__(self, "_shape_signature", signature)

    @property
    def shape(self) -> list[int]:
        """Return the tensor dimensions without exposing mutable internal state."""

        return list(self._shape)

    @property
    def shape_signature(self) -> list[int]:
        """Return static/dynamic dimension metadata without exposing state."""

        return list(self._shape_signature or ())


@dataclass(frozen=True)
class TFLiteInspection:
    """The single input and output exposed by a TensorFlow Lite model."""

    input: InspectedTensor
    output: InspectedTensor


@dataclass(frozen=True)
class DetectorSpec:
    """Validated immutable metadata for an exported single-class detector."""

    input: InspectedTensor
    output: InspectedTensor
    _class_names: tuple[str, ...] = field(
        default=("lib",), init=False, repr=False
    )

    @property
    def class_names(self) -> list[str]:
        """Return compatible list-shaped class metadata without exposing state."""

        return list(self._class_names)


def _replace_with_retry(temporary: Path, target: Path) -> None:
    for attempt in range(len(_REPLACE_RETRY_DELAYS) + 1):
        try:
            temporary.replace(target)
            return
        except OSError as exc:
            retryable = _is_windows() and (
                isinstance(exc, PermissionError) or getattr(exc, "winerror", None) == 5
            )
            if not retryable or attempt == len(_REPLACE_RETRY_DELAYS):
                raise
            time.sleep(_REPLACE_RETRY_DELAYS[attempt])


def _sync_parent_directory(parent: Path) -> None:
    if not _supports_parent_directory_sync():
        return
    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except BaseException as sync_error:
        try:
            os.close(descriptor)
        except BaseException as cleanup_error:
            sync_error.add_note(
                f"failed to close parent directory: {cleanup_error!r}"
            )
        raise
    else:
        os.close(descriptor)


def atomic_write_bytes(target: Path, data: bytes) -> None:
    """Perform a process-level atomic replace, not power-loss-durable storage.

    The temporary file is synced before replacement. POSIX also syncs the parent
    directory, while Windows ``Path.replace`` cannot promise rename durability
    across sudden power loss; the release layer must provide that stronger policy.
    """

    target = Path(target)
    with _serialized_target(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        descriptor: int | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            stream = os.fdopen(descriptor, "wb")
            descriptor = None
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException as write_error:
                try:
                    stream.close()
                except BaseException as cleanup_error:
                    write_error.add_note(
                        f"failed to close temporary stream: {cleanup_error!r}"
                    )
                raise
            else:
                stream.close()
            _replace_with_retry(temporary, target)
            _sync_parent_directory(target.parent)
        except BaseException as primary_error:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException as cleanup_error:
                    primary_error.add_note(
                        f"failed to close temporary descriptor: {cleanup_error!r}"
                    )
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except BaseException as cleanup_error:
                    primary_error.add_note(
                        f"failed to remove temporary file: {cleanup_error!r}"
                    )
            raise


def convert_classifier(source: Path, target: Path, precision: str) -> None:
    """Convert a classifier with the Android tensor contract to TensorFlow Lite."""

    source = Path(source)
    target = Path(target)
    if precision not in {"fp16", "fp32"}:
        raise ModelConversionError(f"unsupported precision: {precision}")

    try:
        _ensure_distinct_files(source, target)
        model = tf.keras.models.load_model(source, compile=False)
        if model.input_shape != (None, 128, 128, 3) or model.output_shape != (
            None,
            1,
        ):
            raise ModelConversionError("classifier tensor contract mismatch")

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if precision == "fp16":
            converter.target_spec.supported_types = [tf.float16]
        converted = converter.convert()
        atomic_write_bytes(target, converted)
    except ModelConversionError:
        raise
    except Exception as exc:
        raise ModelConversionError(f"failed to convert classifier: {source}") from exc


def _default_detector_loader(source: str) -> object:
    from ultralytics import YOLO

    return YOLO(source)


def _validate_detector_model(model: object) -> None:
    try:
        task = getattr(model, "task")
        names = getattr(model, "names")
    except Exception as exc:
        raise ModelConversionError("failed to inspect detector model contract") from exc

    if task != "detect":
        raise ModelConversionError("detector must use the detection task")
    if not isinstance(names, Mapping) or len(names) != 1:
        raise ModelConversionError("detector class names must be exactly {0: 'lib'}")
    key, name = next(iter(names.items()))
    if type(key) is not int or key != 0 or name != "lib":
        raise ModelConversionError("detector class names must be exactly {0: 'lib'}")


def _is_link_or_reparse(path: Path) -> bool:
    """Check link-like filesystem metadata without following the path."""

    metadata = path.lstat()
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _resolved_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(directory.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _reject_link_or_reparse(path: Path, description: str) -> None:
    try:
        rejected = _is_link_or_reparse(path)
    except OSError as exc:
        raise ModelConversionError(f"failed to inspect {description}: {path}") from exc
    if rejected:
        raise ModelConversionError(f"{description} must not be a link or reparse point")


def _resolve_exported_tflite(exported: object) -> Path:
    if not isinstance(exported, (str, os.PathLike)):
        raise ModelConversionError("exported TFLite result must be a path")

    path = Path(exported)
    _reject_link_or_reparse(path, "exported TFLite path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ModelConversionError(
            f"exported TFLite result does not exist: {path}"
        ) from exc
    if stat.S_ISREG(metadata.st_mode):
        if path.suffix.lower() != ".tflite":
            raise ModelConversionError(
                f"exported TFLite result is not a .tflite file: {path}"
            )
        return path
    if stat.S_ISDIR(metadata.st_mode):
        candidates: list[Path] = []
        try:
            children = path.iterdir()
            for entry_count, candidate in enumerate(children, start=1):
                if entry_count > _MAX_EXPORT_DIRECTORY_ENTRIES:
                    raise ModelConversionError(
                        "export directory has too many entries"
                    )
                if candidate.suffix.lower() != ".tflite":
                    continue
                _reject_link_or_reparse(candidate, "exported TFLite candidate")
                candidate_metadata = candidate.lstat()
                if not stat.S_ISREG(candidate_metadata.st_mode):
                    continue
                if not _resolved_within(candidate, path):
                    raise ModelConversionError(
                        "exported TFLite candidate is outside export directory: "
                        f"{candidate}"
                    )
                candidates.append(candidate)
                if len(candidates) == 2:
                    break
        except ModelConversionError:
            raise
        except OSError as exc:
            raise ModelConversionError(
                f"failed to inspect exported TFLite directory: {path}"
            ) from exc
        if len(candidates) != 1:
            raise ModelConversionError(
                "export directory must contain exactly one exported TFLite file"
            )
        return candidates[0]
    raise ModelConversionError(
        f"exported TFLite result is not a file or directory: {path}"
    )


def _ensure_detector_artifact_is_distinct(
    artifact: Path, source: Path, target: Path
) -> None:
    try:
        _ensure_distinct_files(artifact, source)
    except ModelConversionError as exc:
        raise ModelConversionError(
            "detector artifact and source refer to the same file"
        ) from exc
    try:
        _ensure_distinct_files(artifact, target)
    except ModelConversionError as exc:
        raise ModelConversionError(
            "detector artifact and target refer to the same file"
        ) from exc


def _validate_detector_tensors(inspection: TFLiteInspection) -> None:
    if inspection.input.dtype != "float32" or inspection.output.dtype != "float32":
        raise ModelConversionError("detector input and output dtypes must be float32")
    if inspection.input.shape != [1, 640, 640, 3]:
        raise ModelConversionError("detector input tensor contract mismatch")
    if inspection.input.shape_signature != [1, 640, 640, 3]:
        raise ModelConversionError("detector input shape signature must be static")
    output_shape = inspection.output.shape
    if (
        len(output_shape) != 3
        or output_shape[0] != 1
        or output_shape[1] != 5
        or output_shape[2] <= 0
    ):
        raise ModelConversionError("detector output tensor contract mismatch")
    if inspection.output.shape_signature != output_shape:
        raise ModelConversionError("detector output shape signature must be static")


class _OwnedExportWorkspace:
    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="water-model-export-")
        self.path = Path(self._temporary.name)

    def cleanup(self) -> None:
        self._temporary.cleanup()


def _cleanup_owned_export_workspace(
    workspace: object,
    *,
    primary_error: BaseException | None,
    committed: bool,
) -> None:
    try:
        workspace.cleanup()
    except BaseException as cleanup_error:
        if primary_error is not None:
            primary_error.add_note(
                f"failed to clean detector export workspace: {cleanup_error!r}"
            )
        elif committed:
            LOGGER.warning(
                "failed to clean committed detector export workspace %s: %r",
                workspace.path,
                cleanup_error,
            )
        else:
            raise


def _resolved_current_directory() -> Path:
    return Path.cwd().resolve()


def _restore_process_environment(
    environment: MutableMapping[str, str], snapshot: Mapping[str, str]
) -> None:
    """Restore an environment by per-key diff while attempting every change."""

    errors: list[tuple[str, BaseException]] = []
    current_keys = set(environment)
    snapshot_keys = set(snapshot)
    for key in sorted(current_keys - snapshot_keys):
        try:
            del environment[key]
        except BaseException as exc:
            errors.append((f"delete {key!r}", exc))
    for key in sorted(snapshot_keys):
        if environment.get(key) == snapshot[key]:
            continue
        try:
            environment[key] = snapshot[key]
        except BaseException as exc:
            errors.append((f"set {key!r}", exc))

    if errors:
        operation, primary_error = errors[0]
        primary_error.add_note(f"during environment restore: {operation}")
        for later_operation, later_error in errors[1:]:
            primary_error.add_note(
                f"environment restore also failed during {later_operation}: "
                f"{later_error!r}"
            )
        raise primary_error


@contextmanager
def _detector_export_working_directory(workspace: Path) -> Iterator[None]:
    """Exclusively isolate process-wide CWD/environment during default export.

    Callers must not mutate ``os.environ`` concurrently with this critical section.
    """

    acquired = False
    changed = False
    original_cwd: Path | None = None
    environment_snapshot: dict[str, str] | None = None
    primary_error: BaseException | None = None
    try:
        _DETECTOR_CWD_LOCK.acquire()
        acquired = True
    except BaseException:
        raise

    try:
        try:
            original_cwd = _resolved_current_directory()
        except Exception as exc:
            primary_error = ModelConversionError(
                "failed to read current working directory"
            )
            primary_error.__cause__ = exc
        if primary_error is None:
            try:
                environment_snapshot = dict(os.environ)
            except Exception as exc:
                primary_error = ModelConversionError(
                    "failed to snapshot detector export environment"
                )
                primary_error.__cause__ = exc
        if primary_error is None:
            try:
                os.chdir(workspace)
                changed = True
            except Exception as exc:
                primary_error = ModelConversionError(
                    "failed to change detector export working directory"
                )
                primary_error.__cause__ = exc
        if primary_error is None:
            try:
                yield
            except BaseException as exc:
                primary_error = exc
    finally:
        if environment_snapshot is not None:
            try:
                _restore_process_environment(os.environ, environment_snapshot)
            except BaseException as environment_error:
                if primary_error is not None:
                    primary_error.add_note(
                        "failed to restore detector export environment: "
                        f"{environment_error!r}"
                    )
                    for note in getattr(environment_error, "__notes__", []):
                        primary_error.add_note(note)
                else:
                    primary_error = ModelConversionError(
                        "failed to restore detector export environment"
                    )
                    primary_error.__cause__ = environment_error
        if changed and original_cwd is not None:
            try:
                os.chdir(original_cwd)
            except BaseException as restore_error:
                if primary_error is not None:
                    primary_error.add_note(
                        f"failed to restore working directory: {restore_error!r}"
                    )
                else:
                    primary_error = ModelConversionError(
                        "failed to restore detector export working directory"
                    )
                    primary_error.__cause__ = restore_error
        if acquired:
            try:
                _DETECTOR_CWD_LOCK.release()
            except BaseException as release_error:
                if primary_error is not None:
                    primary_error.add_note(
                        f"failed to release detector CWD lock: {release_error!r}"
                    )
                else:
                    primary_error = release_error

    if primary_error is not None:
        raise primary_error


@contextmanager
def _redirect_default_detector_export(
    model: object, workspace: Path
) -> Iterator[None]:
    try:
        inner_model = getattr(model, "model")
        original_pt_path = getattr(inner_model, "pt_path")
    except Exception as exc:
        raise ModelConversionError(
            "cannot safely redirect detector export: missing model.pt_path"
        ) from exc

    export_anchor = workspace / "detector.pt"
    try:
        setattr(inner_model, "pt_path", str(export_anchor))
    except Exception as exc:
        failure = ModelConversionError(
            "cannot safely redirect detector export: model.pt_path is not writable"
        )
        try:
            setattr(inner_model, "pt_path", original_pt_path)
        except BaseException as cleanup_error:
            failure.add_note(
                f"failed to restore detector model.pt_path: {cleanup_error!r}"
            )
        raise failure from exc

    try:
        yield
    except BaseException as primary_error:
        try:
            setattr(inner_model, "pt_path", original_pt_path)
        except BaseException as cleanup_error:
            primary_error.add_note(
                f"failed to restore detector model.pt_path: {cleanup_error!r}"
            )
        raise
    else:
        try:
            setattr(inner_model, "pt_path", original_pt_path)
        except Exception as exc:
            raise ModelConversionError(
                "failed to restore detector model.pt_path"
            ) from exc


def _export_detector(model: object, source: Path, precision: str) -> object:
    try:
        return model.export(
            format="tflite",
            imgsz=640,
            nms=False,
            half=precision == "fp16",
            int8=False,
        )
    except Exception as exc:
        raise ModelConversionError(f"failed to export detector: {source}") from exc


def _read_validated_detector_export(
    exported: object,
    source: Path,
    target: Path,
    *,
    owned_workspace: Path | None,
) -> tuple[bytes, TFLiteInspection]:
    artifact = _resolve_exported_tflite(exported)
    _ensure_detector_artifact_is_distinct(artifact, source, target)
    if owned_workspace is not None and not _resolved_within(
        artifact, owned_workspace
    ):
        raise ModelConversionError(
            "default detector artifact must remain inside the owned export workspace"
        )
    try:
        converted = artifact.read_bytes()
    except OSError as exc:
        raise ModelConversionError(
            f"failed to read exported detector TFLite: {artifact}"
        ) from exc
    try:
        with tempfile.TemporaryDirectory(prefix="water-model-detector-") as staging:
            staged_artifact = Path(staging) / "detector.tflite"
            staged_artifact.write_bytes(converted)
            inspection = inspect_tflite(staged_artifact)
            _validate_detector_tensors(inspection)
    except ModelConversionError:
        raise
    except Exception as exc:
        raise ModelConversionError(
            f"failed to stage exported detector TFLite: {artifact}"
        ) from exc
    return converted, inspection


def _publish_detector(target: Path, converted: bytes) -> None:
    try:
        atomic_write_bytes(target, converted)
    except Exception as exc:
        raise ModelConversionError(
            f"failed to publish converted detector: {target}"
        ) from exc


def convert_detector(
    source: Path,
    target: Path,
    precision: str,
    loader: Callable[[str], object] | None = None,
) -> DetectorSpec:
    """Export and publish a validated raw single-class YOLO TFLite model."""

    source = Path(source)
    target = Path(target)
    if precision not in {"fp16", "fp32"}:
        raise ModelConversionError(f"unsupported precision: {precision}")
    _ensure_distinct_files(source, target)
    if target.suffix.lower() != ".tflite":
        raise ModelConversionError("detector target must use the .tflite suffix")

    uses_default_loader = loader is None
    effective_loader = _default_detector_loader if uses_default_loader else loader
    try:
        model = effective_loader(str(source))
    except Exception as exc:
        raise ModelConversionError(f"failed to load detector: {source}") from exc

    _validate_detector_model(model)
    if uses_default_loader:
        try:
            owned_workspace = _OwnedExportWorkspace()
        except Exception as exc:
            raise ModelConversionError(
                "failed to create detector export workspace"
            ) from exc
        committed = False
        try:
            with _detector_export_working_directory(owned_workspace.path):
                with _redirect_default_detector_export(model, owned_workspace.path):
                    exported = _export_detector(model, source, precision)
            converted, inspection = _read_validated_detector_export(
                exported,
                source,
                target,
                owned_workspace=owned_workspace.path,
            )
            _publish_detector(target, converted)
            committed = True
        except BaseException as primary_error:
            _cleanup_owned_export_workspace(
                owned_workspace,
                primary_error=primary_error,
                committed=False,
            )
            raise
        else:
            _cleanup_owned_export_workspace(
                owned_workspace,
                primary_error=None,
                committed=committed,
            )
    else:
        exported = _export_detector(model, source, precision)
        converted, inspection = _read_validated_detector_export(
            exported,
            source,
            target,
            owned_workspace=None,
        )
        _publish_detector(target, converted)
    return DetectorSpec(input=inspection.input, output=inspection.output)


def _inspect_tensor(detail: dict[str, object]) -> InspectedTensor:
    shape = tuple(int(dimension) for dimension in detail["shape"])
    signature_value = detail.get("shape_signature", detail["shape"])
    shape_signature = tuple(int(dimension) for dimension in signature_value)
    return InspectedTensor(
        name=str(detail["name"]),
        dtype=np.dtype(detail["dtype"]).name,
        _shape=shape,
        _shape_signature=shape_signature,
    )


def inspect_tflite(model_path: Path) -> TFLiteInspection:
    """Allocate and inspect a TensorFlow Lite model with one input and output."""

    model_path = Path(model_path)
    try:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ModelConversionError(
                "TFLite model must expose exactly one input and one output"
            )
        return TFLiteInspection(
            input=_inspect_tensor(inputs[0]),
            output=_inspect_tensor(outputs[0]),
        )
    except ModelConversionError:
        raise
    except Exception as exc:
        raise ModelConversionError(
            f"failed to inspect TFLite model: {model_path}"
        ) from exc
