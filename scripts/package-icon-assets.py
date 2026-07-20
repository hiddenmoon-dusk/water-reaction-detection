from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def package_icon(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"找不到 PNG 图标：{source}")

    with Image.open(source) as image:
        if image.width != image.height:
            raise ValueError("PNG 图标必须是正方形")
        if image.width < 256:
            raise ValueError("PNG 图标至少需要 256×256 像素")
        rgba = image.convert("RGBA")

    target.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(target, format="ICO", sizes=[(size, size) for size in ICON_SIZES])


def main() -> None:
    parser = argparse.ArgumentParser(description="将正方形 PNG 打包为多尺寸 Windows ICO")
    parser.add_argument("source_png", type=Path)
    parser.add_argument("target_ico", type=Path)
    args = parser.parse_args()
    package_icon(args.source_png, args.target_ico)
    print(f"已生成 Windows 图标：{args.target_ico}")


if __name__ == "__main__":
    main()
