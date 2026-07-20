from ultralytics import YOLO
model = YOLO("yolov8n.pt")
print("开始训练")
results = model.train(data="C:/Users/Muelsyse/Desktop/work/dataset.yaml", epochs=50, imgsz=640)
print("训练结束")