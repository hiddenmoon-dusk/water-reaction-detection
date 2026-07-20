"""命令行预测脚本 — 用于批量处理或集成到其他流程

用法:
  python scripts/predict.py test1.jpg
  python scripts/predict.py test1.jpg --scan          # 启用精细扫描
  python scripts/predict.py test1.jpg --conf 0.5 --cls 0.6
  python scripts/predict.py *.jpg                     # 批量处理
"""
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent.parent

DETECTOR_PATH = ROOT / "models" / "detector.pt"
CLASSIFIER_PATH = ROOT / "models" / "classifier.h5"

CLASS_NAMES = {0: "Negative (未反应)", 1: "Positive (已反应)"}


def load_models():
    detector = YOLO(str(DETECTOR_PATH))
    classifier = tf.keras.models.load_model(str(CLASSIFIER_PATH))
    input_size = classifier.input_shape[1:3][::-1]  # (H,W) -> (W,H) for cv2.resize
    return detector, classifier, tuple(input_size)


def tiled_detect(detector, image, tile_size=640, overlap=0.2, conf=0.3):
    """
    滑动窗口分块检测，解决大图中小目标漏检问题。
    返回: list of [x1, y1, x2, y2, conf, cls] (全局坐标)
    """
    h, w = image.shape[:2]

    if h <= tile_size and w <= tile_size:
        results = detector(image, conf=conf, verbose=False)
        all_boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                all_boxes.append([x1, y1, x2, y2, float(box.conf[0]), int(box.cls[0])])
        return all_boxes

    stride_h = int(tile_size * (1 - overlap))
    stride_w = int(tile_size * (1 - overlap))

    cols = max(1, (w - tile_size) // stride_w + 2)
    rows = max(1, (h - tile_size) // stride_h + 2)
    total = rows * cols
    done = 0

    all_boxes = []

    for row in range(rows):
        for col in range(cols):
            x = col * stride_w
            y = row * stride_h
            x = min(x, w - tile_size)
            y = min(y, h - tile_size)
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x = max(0, x2 - tile_size)
            y = max(0, y2 - tile_size)

            tile = image[y:y2, x:x2]
            results = detector(tile, conf=conf, verbose=False)

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    all_boxes.append([
                        bx1 + x, by1 + y, bx2 + x, by2 + y,
                        float(box.conf[0]), int(box.cls[0]),
                    ])

            done += 1
            print(f"\r  扫描进度: {done}/{total} 块", end="", flush=True)

    print()  # 换行

    # NMS 去重
    if len(all_boxes) < 2:
        return all_boxes

    all_boxes = np.array(all_boxes)
    boxes_xyxy = all_boxes[:, :4]
    scores = all_boxes[:, 4]

    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        keep.append(order[0])
        if len(order) == 1:
            break
        x1 = np.maximum(boxes_xyxy[order[0], 0], boxes_xyxy[order[1:], 0])
        y1 = np.maximum(boxes_xyxy[order[0], 1], boxes_xyxy[order[1:], 1])
        x2 = np.minimum(boxes_xyxy[order[0], 2], boxes_xyxy[order[1:], 2])
        y2 = np.minimum(boxes_xyxy[order[0], 3], boxes_xyxy[order[1:], 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes_xyxy[order[0], 2] - boxes_xyxy[order[0], 0]) * \
                (boxes_xyxy[order[0], 3] - boxes_xyxy[order[0], 1])
        area2 = (boxes_xyxy[order[1:], 2] - boxes_xyxy[order[1:], 0]) * \
                (boxes_xyxy[order[1:], 3] - boxes_xyxy[order[1:], 1])
        iou = inter / (area1 + area2 - inter + 1e-6)
        order = order[1:][iou < 0.5]

    return all_boxes[keep].tolist()


def process_image(image_path, detector, classifier, input_size,
                  conf_thresh=0.3, cls_thresh=0.5, save=True, scan=False):
    image_path = Path(image_path).resolve()  # 转为绝对路径
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # ===== 检测 =====
    if scan and (h > 640 or w > 640):
        print(f"  🔍 精细扫描模式 (图片尺寸: {w}x{h})")
        all_boxes = tiled_detect(detector, img_rgb, conf=conf_thresh)
    else:
        results = detector(img_rgb, conf=conf_thresh, verbose=False)
        all_boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                all_boxes.append([x1, y1, x2, y2, float(box.conf[0]), int(box.cls[0])])

    found = 0
    for box_info in all_boxes:
        x1, y1, x2, y2, det_conf, det_cls = box_info
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)

        if y2 - y1 <= 0 or x2 - x1 <= 0:
            continue

        cropped = img_rgb[y1:y2, x1:x2]
        resized = cv2.resize(cropped, input_size)
        input_tensor = np.expand_dims(resized, axis=0)
        prediction = classifier.predict(input_tensor, verbose=0)[0][0]

        predicted_class = 1 if prediction > cls_thresh else 0
        label = CLASS_NAMES[predicted_class]
        confidence = prediction if predicted_class == 1 else 1 - prediction

        found += 1
        print(f"  反应管 #{found}: {label} (置信度: {confidence:.2%})")

        color = (0, 255, 0) if predicted_class == 1 else (255, 0, 0)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        text = f"{label.split(' ')[0]}: {confidence:.1%}"
        cv2.putText(img_rgb, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if found == 0:
        print("  未检测到反应管")
        if scan:
            print("  已使用精细扫描模式，请确认反应管特征是否与训练数据一致。")
        else:
            print("  提示: 可添加 --scan 参数启用精细扫描以检测图片中的小目标。")
    else:
        print(f"  共检测到 {found} 个反应管")

    if save:
        output_path = image_path.parent / f"result_{image_path.name}"
        cv2.imwrite(str(output_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"  结果已保存: {output_path}")

    return img_rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="反应管检测与分类")
    parser.add_argument("images", nargs="+", help="图片路径（支持通配符批量处理）")
    parser.add_argument("--conf", type=float, default=0.3, help="目标检测置信度阈值")
    parser.add_argument("--cls", type=float, default=0.5, help="分类阈值")
    parser.add_argument("--no-save", action="store_true", help="不保存结果图")
    parser.add_argument("--scan", action="store_true",
                        help="启用精细扫描模式（滑动窗口分块检测，适合反应管占比小的大图）")
    args = parser.parse_args()

    print("加载模型...")
    detector, classifier, input_size = load_models()

    for img_path in args.images:
        print(f"处理: {img_path}")
        process_image(img_path, detector, classifier, input_size,
                      args.conf, args.cls, not args.no_save, args.scan)
