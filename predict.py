import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

DETECTOR_MODEL_PATH = "C:/Users/Muelsyse/Desktop/work/runs/detect/train/weights/best.pt"

CLASSIFIER_MODEL_PATH = "C:/Users/Muelsyse/Desktop/work/reaction_classifier.h5"

CLASS_NAMES_CNN = ['Negative (未反应)', 'Positive (已反应)']

TEST_IMAGE_PATH = "C:/Users/Muelsyse/Desktop/work/test2.jpg"

IMG_HEIGHT = 128
IMG_WIDTH = 128


print("正在加载目标检测模型...")
try:
    detector = YOLO(DETECTOR_MODEL_PATH)
    print("目标检测模型加载成功！")
except Exception as e:
    print(f"加载目标检测模型失败：{e}")
    print("请检查 DETECTOR_MODEL_PATH 是否正确。")
    exit()

print("正在加载分类模型...")
try:
    classifier = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
    print("分类模型加载成功！")
except Exception as e:
    print(f"加载分类模型失败：{e}")
    print("请检查 CLASSIFIER_MODEL_PATH 是否正确。")
    exit()

def process_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误：无法读取图片 {image_path}")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"读取或转换图片时出错: {e}")
        return

    print(f"正在处理图片: {image_path}")

    detection_results = detector(img_rgb)

    for result in detection_results:
        boxes = result.boxes
        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            y1, y2 = max(0, y1), min(img_rgb.shape[0], y2)
            x1, x2 = max(0, x1), min(img_rgb.shape[1], x2)

            if y2 - y1 <= 0 or x2 - x1 <= 0:
                print("警告：检测到的框无效，跳过此框。")
                continue

            cropped_tube = img_rgb[y1:y2, x1:x2]

            if cropped_tube.size == 0:
                print("警告：裁剪出的区域是空的，跳过CNN分类。")
                continue

            resized_tube = cv2.resize(cropped_tube, (IMG_WIDTH, IMG_HEIGHT))

            input_tensor = np.expand_dims(resized_tube, axis=0)


            try:
                prediction = classifier.predict(input_tensor, verbose=0)[0][0]

                predicted_class_idx = 1 if prediction > 0.5 else 0
                label = CLASS_NAMES_CNN[predicted_class_idx]

                confidence = prediction if predicted_class_idx == 1 else 1 - prediction

                print(f"检测到反应管，CNN 预测为: {label} (置信度: {confidence:.2f})")

                text = f"{label} ({confidence:.2f})"

                color = (0, 255, 0) if predicted_class_idx == 1 else (255, 0, 0)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception as e:
                print(f"CNN 分类预测或绘制结果时出错: {e}")

    if len(detection_results[0].boxes) == 0:
        print("未在图片中检测到反应管。")

    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_display)
    plt.title("Reaction Tube Detection and Classification Result")
    plt.axis('off')
    plt.show()

    # 如果你想保存结果图片，可以取消下面两行的注释
    # output_path = "output_" + os.path.basename(image_path)
    # cv2.imwrite(output_path, img)
    # print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    print("正在打开文件选择器...")

    selected_file_path = filedialog.askopenfilename(
        title="请选择要测试的反应管图片",
        filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")]
    )

    if selected_file_path:
        print(f"成功载入图片: {selected_file_path}")
        process_image(selected_file_path)
    else:
        print("您取消了选择，程序结束。")