import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
# 新的路径
new_path = r'D:\temp\haarcascade_frontalface_default.xml' #需要按需求更改
CASCADE_PATH = new_path

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_recognizer_celeba.h5")
LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_label_names_celeba.npz")

# 打印路径
print(f"CASCADE_PATH: {CASCADE_PATH}")

# 初始化人脸检测器
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

# 加载训练好的深度学习模型
face_model = load_model(MODEL_PATH)

# 载入标签映射
data = np.load(LABELS_PATH, allow_pickle=True)  # 添加 allow_pickle=True
label_names = data['label_names'].item()

def predict_face_name(face_roi):
    try:
        # 1. 预处理图像
        face_resized = cv2.resize(face_roi, (64, 64))
        img_array = img_to_array(face_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 2. 预测
        predictions = face_model.predict(img_array)
        confidence = np.max(predictions)
        label_index = np.argmax(predictions)

        # 3. 设置置信度阈值
        if confidence > 0.35:
            return label_names[label_index]
        else:
            return "未录入"
    except Exception as e:
        print(f"人脸识别错误: {e}")
        return "Error"
