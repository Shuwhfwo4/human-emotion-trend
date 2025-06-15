import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from emotion_model import build_emotion_model

# 1. 路径与超参数配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FER2013 数据集目录
DATASET_DIR = os.path.join(BASE_DIR, "face_emotion_system", "dataset", "fer2013")

# 训练时保存模型权重的路径
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
EMO_MODEL_PATH   = os.path.join(MODEL_DIR, "emotion_model.h5")
LABEL_NAMES_PATH = os.path.join(MODEL_DIR, "label_names.npz")  # 保存情绪标签映射

# 图像尺寸与训练超参数
IMG_SIZE    = (48, 48)
BATCH_SIZE  = 64
EPOCHS      = 30
NUM_CLASSES = 7  # FER2013 分 7 类情绪

# 2. 数据生成器：归一化 + 验证集拆分
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # 20% 作为验证集
)

# 2.1 训练集生成器（80% 样本）
train_generator = datagen.flow_from_directory(
    directory=os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

# 2.2 验证集生成器（20% 样本）
val_generator = datagen.flow_from_directory(
    directory=os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True,
    seed=42
)

# 2.3 测试集生成器
test_generator = datagen.flow_from_directory(
    directory=os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# 保存情绪标签映射：{'Angry':0, 'Disgust':1, …}
label_map = train_generator.class_indices
# 反转映射：{0:'Angry', 1:'Disgust', …}
label_map_inv = {v: k for k, v in label_map.items()}

# 将整数键转换为字符串后保存，避免 TypeError
label_map_strkeys = {str(k): v for k, v in label_map_inv.items()}
np.savez(LABEL_NAMES_PATH, **label_map_strkeys)

print(f"检测到 {len(label_map)} 个类别：{label_map_inv}")

# 3. 构建 9 层情绪识别模型并编译
model = build_emotion_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES)
model.summary()

#4. 设置回调：保存最佳模型 提前停止
checkpoint = ModelCheckpoint(
    filepath=EMO_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
earlystop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# 5. 开始训练
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop]
)

print(f"9 层情绪模型训练完成，已保存最优权重到：{EMO_MODEL_PATH}")

# 在测试集上评估并打印准确率
loss, acc = model.evaluate(test_generator)
print(f"测试集准确率：{acc * 100:.2f}%")

