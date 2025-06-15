import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from face_model import build_face_model
from collections import Counter
BASE_DIR = Path(__file__).parent
DATASET_PATH = Path(r"D:\temp\img_align_celeba") #以下都需要按需求更改
MODEL_SAVE_PATH = BASE_DIR / "models" / "face_recognizer_celeba.h5"
LABEL_SAVE_PATH = BASE_DIR / "models" / "face_label_names_celeba.npz"
IDENTITY_FILE = Path(r"D:\temp\Anno\identity_CelebA.txt")
BBOX_FILE = Path(r"D:\temp\Anno\list_bbox_celeba.txt")
EVAL_FILE = Path(r"D:\temp\Eval\list_eval_partition.txt")

def load_celeba_dataset():
    """加载CelebA数据集并预处理"""
    # 读取身份标签
    print("加载身份标签...")
    identity_df = pd.read_csv(IDENTITY_FILE, sep=" ", header=None, names=["image", "identity"])
    identity_df = identity_df.set_index("image")
    # 读取边界框信息
    print("加载边界框信息...")
    with open(BBOX_FILE, 'r') as f:
        lines = f.readlines()
    # 从第3行开始读取数据
    bbox_data = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 5:
            bbox_data.append({
                "image": parts[0],
                "x_1": int(parts[1]),
                "y_1": int(parts[2]),
                "width": int(parts[3]),
                "height": int(parts[4])
            })
    bbox_df = pd.DataFrame(bbox_data)
    bbox_df = bbox_df.set_index("image")

    # 读取评估划分
    print("加载评估划分...")
    eval_df = pd.read_csv(EVAL_FILE, sep=" ", header=None, names=["image", "partition"])
    eval_df = eval_df.set_index("image")

    # 合并所有数据
    full_df = identity_df.join(bbox_df, how="inner").join(eval_df, how="inner")

    # 过滤出训练集数据 (partition = 0)
    train_df = full_df[full_df["partition"] == 0]

    # 统计每个身份的出现次数
    identity_counts = train_df["identity"].value_counts()

    # 提高阈值减少类别数量
    min_images = 30
    valid_identities = identity_counts[identity_counts >= min_images].index

    # 过滤数据集
    filtered_df = train_df[train_df["identity"].isin(valid_identities)]

    # 创建身份到标签ID的映射
    unique_identities = filtered_df["identity"].unique()
    identity_to_id = {identity: idx for idx, identity in enumerate(unique_identities)}
    id_to_identity = {idx: identity for identity, idx in identity_to_id.items()}
    print(f"处理后的数据集包含 {len(filtered_df)} 张图片，{len(unique_identities)} 个身份")

    faces = []
    labels = []
    skipped_count = 0  # 记录跳过的无效图像数量
    label_counts = Counter()  # 用于统计每个标签的样本数量

    # 处理每张图片
    for image_name, row in filtered_df.iterrows():
        img_path = DATASET_PATH / image_name

        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取图片: {img_path}")
            skipped_count += 1
            continue

        # 获取边界框坐标
        x, y, w, h = row["x_1"], row["y_1"], row["width"], row["height"]

        # 扩展边界框以确保包含完整人脸
        expand_factor = 0.1

        # 计算扩展后的边界框
        x1 = max(0, int(x - w * expand_factor))
        y1 = max(0, int(y - h * expand_factor))
        w1 = min(img.shape[1] - x1, int(w * (1 + 2 * expand_factor)))
        h1 = min(img.shape[0] - y1, int(h * (1 + 2 * expand_factor)))

        # 确保边界框有效（宽度和高度大于0）
        if w1 <= 0 or h1 <= 0:
            # 如果扩展后的边界框无效，使用原始边界框
            x1 = max(0, x)
            y1 = max(0, y)
            w1 = min(img.shape[1] - x1, w)
            h1 = min(img.shape[0] - y1, h)

            # 再次检查有效性
            if w1 <= 0 or h1 <= 0:
                print(f"无效的边界框: {image_name} - x={x}, y={y}, w={w}, h={h}")
                skipped_count += 1
                continue

        # 裁剪人脸区域
        face_roi = img[y1:y1 + h1, x1:x1 + w1]

        # 检查裁剪区域是否有效
        if face_roi.size == 0:
            print(f"裁剪出空区域: {image_name} - x={x1}, y={y1}, w={w1}, h={h1}")
            skipped_count += 1
            continue
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"转换灰度图失败: {image_name} - {e}")
            skipped_count += 1
            continue
        # 调整大小并添加到数据集
        face_resized = cv2.resize(gray, (64, 64))
        label_id = identity_to_id[row["identity"]]

        # 记录标签计数
        label_counts[label_id] += 1
        faces.append(face_resized)
        labels.append(label_id)

    # 转换为NumPy数组
    faces = np.array(faces, dtype="float32") / 255.0
    faces = np.expand_dims(faces, axis=-1)

    # 统计每个标签的样本数量
    print(f"跳过了 {skipped_count} 张无效图像")
    print(f"原始标签数量: {len(labels)}")

    # 检查标签分布并过滤掉样本数少于2的类别
    valid_indices = []
    for idx, label_id in enumerate(labels):
        if label_counts[label_id] >= 2:
            valid_indices.append(idx)

    # 只保留有效样本
    faces = faces[valid_indices]
    labels = [labels[i] for i in valid_indices]

    # 重新映射标签ID，使其连续
    unique_labels = sorted(set(labels))
    new_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_labels)}
    labels = [new_id_map[old_id] for old_id in labels]

    # 更新标签映射字典
    new_id_to_identity = {}
    for old_id, identity in id_to_identity.items():
        if old_id in new_id_map:
            new_id = new_id_map[old_id]
            new_id_to_identity[new_id] = identity

    # 转换为独热编码
    labels = to_categorical(np.array(labels))

    print(f"处理后数据集形状: {faces.shape}")
    print(f"有效类别数量: {labels.shape[1]}")
    print(f"过滤后样本数量: {len(labels)}")
    return faces, labels, new_id_to_identity

def train_celeba_model():
    """训练基于CelebA的人脸识别模型"""
    # 1. 加载数据
    print("=" * 50)
    print("开始加载CelebA数据集...")
    X, y, label_names = load_celeba_dataset()
    if len(X) == 0:
        print("\n错误：没有加载任何人脸数据，无法训练")
        exit(1)

    # 检查样本数量
    num_samples = X.shape[0]
    print(f"\n加载了 {num_samples} 个人脸样本")
    if num_samples < 100:
        print("\n警告：样本数量较少，模型性能可能受限")

    # 2. 划分训练集和验证集
    test_size = 0.2
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size,
        stratify=np.argmax(y, axis=1),
        random_state=42
    )
    print(f"\n数据集划分:")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  验证集: {X_val.shape[0]} 样本")
    print(f"  类别数量: {y.shape[1]}")

    # 3. 数据增强
    datagen = ImageDataGenerator(
        rotation_range=15,  # 减少旋转范围以节省内存
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1]  # 减少亮度变化范围
    )

    # 4. 构建模型 - 传递类别数量
    print("\n构建模型...")
    model = build_face_model(num_classes=y.shape[1])

    # 5. 设置回调函数
    checkpoint = ModelCheckpoint(
        str(MODEL_SAVE_PATH),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    earlystop = EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    # 6. 训练模型
    print("\n开始训练模型...")
    epochs = 30
    batch_size = 32

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[checkpoint, earlystop],
        verbose=1,
        steps_per_epoch=len(X_train) // batch_size
    )

    # 7. 保存模型
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    np.savez(LABEL_SAVE_PATH, label_names=label_names)
    print(f"\n训练完成！")
    print(f"模型保存至: {MODEL_SAVE_PATH}")
    print(f"标签映射保存至: {LABEL_SAVE_PATH}")

    # 打印最终准确率
    if 'val_accuracy' in history.history and len(history.history['val_accuracy']) > 0:
        val_acc = history.history['val_accuracy'][-1] * 100
        print(f"验证集准确率: {val_acc:.2f}%")
    else:
        print("无法获取验证集准确率")

if __name__ == "__main__":
    train_celeba_model()