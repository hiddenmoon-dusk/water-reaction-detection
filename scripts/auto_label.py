"""
自动预标注工具 — 用已训练的 YOLO 模型对新图像进行推理并保存 YOLO 格式标签。
将待标注的图片放入 yolo_dataset/images/new/ 目录下，运行此脚本即可自动生成标签。
"""
from ultralytics import YOLO
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
DETECTOR_PATH = ROOT / "models" / "detector.pt"

IMG_DIR = ROOT / "yolo_dataset" / "images" / "new"
LABEL_DIR = ROOT / "yolo_dataset" / "labels" / "new"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"加载模型: {DETECTOR_PATH}")
model = YOLO(str(DETECTOR_PATH))

images = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.jpeg")) + list(IMG_DIR.glob("*.png"))
print(f"找到 {len(images)} 张图片")

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"跳过无法读取的图片: {img_path}")
        continue

    h, w = img.shape[:2]
    results = model(img)

    label_lines = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0])
            if conf < 0.3:
                continue
            cls = int(box.cls[0])
            xywh = box.xywh[0].cpu().numpy()
            x_center, y_center, bw, bh = xywh
            x_center /= w
            y_center /= h
            bw /= w
            bh /= h
            label_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    label_path = LABEL_DIR / f"{img_path.stem}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))

    print(f"{img_path.name}: 检测到 {len(label_lines)} 个目标，标签 -> {label_path}")

print("\n自动标注完成！请手动检查标签是否正确，然后移动到 train 或 val 目录。")
