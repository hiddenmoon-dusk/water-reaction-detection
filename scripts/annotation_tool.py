"""
简易 OpenCV 标注工具 — 替代 LabelImg。
在当前环境直接运行，无需切换到 Python 3.9。

操作说明:
  - 鼠标左键拖拽画框
  - N 键: 新建空白标签
  - S 键: 保存当前图片的 YOLO 格式标签
  - D 键: 删除上一个框
  - ← → 键: 切换上一张/下一张图片
  - Q 键: 退出
"""
import cv2
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

IMAGES_DIR = ROOT / "yolo_dataset" / "images" / "new"
LABELS_DIR = ROOT / "yolo_dataset" / "labels" / "new"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["tube"]  # 类别名，可根据需要修改
RECT_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

images = sorted(
    list(IMAGES_DIR.glob("*.jpg"))
    + list(IMAGES_DIR.glob("*.jpeg"))
    + list(IMAGES_DIR.glob("*.png"))
)

if not images:
    print(f"请将待标注图片放入: {IMAGES_DIR}")
    sys.exit(1)

idx = 0
drawing = False
start_x = start_y = -1
boxes = []  # [(x1, y1, x2, y2), ...]

img = cv2.imread(str(images[idx]))
if img is None:
    print(f"无法读取: {images[idx]}")
    sys.exit(1)

clone = img.copy()
WIN = "Annotation Tool"

cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)


def redraw():
    global clone
    clone = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(clone, (x1, y1), (x2, y2), RECT_COLOR, 2)
        cv2.putText(clone, f"{i + 1}", (x1, y1 - 5), FONT, 0.6, RECT_COLOR, 2)
    h, w = clone.shape[:2]
    cv2.putText(
        clone,
        f"{images[idx].name} | 框数: {len(boxes)} | N:新建 S:保存 D:删除 Q:退出",
        (10, h - 10),
        FONT,
        0.5,
        (255, 255, 255),
        1,
    )


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        tmp = clone.copy()
        cv2.rectangle(tmp, (start_x, start_y), (x, y), (0, 0, 255), 1)
        cv2.imshow(WIN, tmp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2 = max(0, min(x, img.shape[1]))
        y2 = max(0, min(y, img.shape[0]))
        x1 = max(0, min(start_x, img.shape[1]))
        y1_ = max(0, min(start_y, img.shape[0]))
        if abs(x2 - x1) > 5 and abs(y2 - y1_) > 5:
            boxes.append((x1, y1_, x2, y2))
        redraw()
        cv2.imshow(WIN, clone)


def save_labels():
    h, w = img.shape[:2]
    lines = []
    for x1, y1, x2, y2 in boxes:
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    label_path = LABELS_DIR / f"{images[idx].stem}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(lines))
    print(f"已保存: {label_path} ({len(lines)} 个框)")


def load_labels():
    global boxes
    label_path = LABELS_DIR / f"{images[idx].stem}.txt"
    boxes = []
    if label_path.exists():
        h, w = img.shape[:2]
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, xc, yc, bw, bh = map(float, parts)
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                boxes.append((x1, y1, x2, y2))


def load_image():
    global img
    img_path = images[idx]
    img = cv2.imread(str(img_path))
    if img is None:
        return
    load_labels()
    redraw()


cv2.setMouseCallback(WIN, mouse_callback)
load_labels()
redraw()
cv2.imshow(WIN, clone)

while True:
    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("n"):
        boxes.clear()
        redraw()
        cv2.imshow(WIN, clone)
    elif key == ord("s"):
        save_labels()
    elif key == ord("d"):
        if boxes:
            boxes.pop()
        redraw()
        cv2.imshow(WIN, clone)
    elif key == 81:  # 左箭头
        idx = max(0, idx - 1)
        load_image()
        cv2.imshow(WIN, clone)
    elif key == 83:  # 右箭头
        idx = min(len(images) - 1, idx + 1)
        load_image()
        cv2.imshow(WIN, clone)

cv2.destroyAllWindows()
