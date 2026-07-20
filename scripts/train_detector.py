from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 使用 yolo11n (最新版) 或 yolo8n
model = YOLO("yolov8n.pt")

print("开始训练目标检测模型...")
results = model.train(
    data=str(ROOT / "dataset.yaml"),
    epochs=50,
    imgsz=640,
    project=str(ROOT / "runs"),
    name="detect",
)

# 复制最佳模型到 models/
import shutil
trained = ROOT / "runs" / "detect" / "weights" / "best.pt"
if trained.exists():
    shutil.copy(str(trained), str(MODEL_DIR / "detector.pt"))
    print(f"最佳模型已复制到: {MODEL_DIR / 'detector.pt'}")

print("训练结束")
