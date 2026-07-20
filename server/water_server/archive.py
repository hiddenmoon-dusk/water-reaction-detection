from __future__ import annotations

import io
import json
import stat
import zipfile
from pathlib import PurePosixPath

from PIL import Image, UnidentifiedImageError


ALLOWED_RESULT_FILES = {"original.jpg", "annotated.png", "result.json"}


class InvalidArchive(ValueError):
    pass


def read_result_archive(data: bytes, max_uncompressed: int) -> dict[str, bytes]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise InvalidArchive("文件不是有效 ZIP") from exc

    with archive:
        infos = archive.infolist()
        names = {info.filename for info in infos}
        if names != ALLOWED_RESULT_FILES or len(infos) != 3:
            raise InvalidArchive("ZIP 必须且只能包含三件套")

        total_size = 0
        for info in infos:
            path = PurePosixPath(info.filename)
            unix_mode = info.external_attr >> 16
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(path.parts) != 1
                or info.is_dir()
                or stat.S_ISLNK(unix_mode)
                or info.flag_bits & 0x1
            ):
                raise InvalidArchive("ZIP 包含不安全路径或文件")
            total_size += info.file_size
            if total_size > max_uncompressed:
                raise InvalidArchive("ZIP 解压后尺寸超限")

        files = {name: archive.read(name) for name in ALLOWED_RESULT_FILES}

    _verify_image(files["original.jpg"], "JPEG")
    _verify_image(files["annotated.png"], "PNG")
    return files


def _verify_image(data: bytes, expected_format: str) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != expected_format:
                raise InvalidArchive(f"图片格式必须是 {expected_format}")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidArchive("图片无法解码") from exc
