import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from pathlib import Path
import numpy as np

# ===== 项目根目录 (scripts/ 的上级) =====
ROOT = Path(__file__).resolve().parent.parent

IMG_HEIGHT, IMG_WIDTH = 224, 224  # MobileNetV2 需要 224x224
BATCH_SIZE = 16
EPOCHS = 50
DATA_DIR = ROOT / "CNN_Dataset"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ===== 数据增强 =====
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
])

# ===== 加载数据集 =====
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATA_DIR),
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATA_DIR),
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

# ===== 计算类别权重（处理数据不均衡） =====
class_names = train_ds.class_names
total_samples = sum(len(files) for _, dirs, files in tf.io.gfile.walk(str(DATA_DIR)) if files)
n_classes = len(class_names)
class_counts = np.zeros(n_classes)
for images, labels in train_ds.unbatch():
    class_counts[int(labels.numpy())] += 1
for images, labels in val_ds.unbatch():
    class_counts[int(labels.numpy())] += 1

class_weight = {}
for i in range(n_classes):
    class_weight[i] = total_samples / (n_classes * class_counts[i])

print(f"类别: {class_names}")
print(f"类别样本数: {dict(zip(class_names, class_counts))}")
print(f"类别权重: {class_weight}")

# ===== 构建模型 (MobileNetV2 迁移学习) =====
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # 冻结预训练权重

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

# ===== 训练（第一阶段：仅训练头部） =====
print("\n=== 第一阶段：训练分类头 ===")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weight,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ],
)

# ===== 解冻微调（第二阶段） =====
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False  # 仅解冻后半部分

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

print("\n=== 第二阶段：微调整个模型 ===")
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ],
)

# ===== 保存模型 =====
model_path = MODEL_DIR / "classifier.h5"
model.save(str(model_path))
print(f"\n分类模型已保存到: {model_path}")

# ===== 评估 =====
loss, acc = model.evaluate(val_ds)
print(f"验证集准确率: {acc:.4f}")
