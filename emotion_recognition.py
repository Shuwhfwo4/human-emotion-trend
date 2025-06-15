import numpy as np
import os
from keras.models import load_model
from emotion_model import build_emotion_model

# 1. 拼出权重与标签路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.h5")
LABEL_NAMES_PATH   = os.path.join(BASE_DIR, "models", "label_names.npz")

# 2. 构建与训练时完全一致的 9 层网络并加载权重
emo_model = build_emotion_model(input_shape=(48, 48, 1), num_classes=7)
emo_model.load_weights(EMOTION_MODEL_PATH)

# 加载情绪标签映射字典：{0:'Angry',1:'Disgust',…}
try:
    label_data = np.load(LABEL_NAMES_PATH, allow_pickle=True)
    # 如果label_data是一个字典，直接使用
    if isinstance(label_data, dict):
        emotion_label_map = {int(k): str(v) for k, v in label_data.items()}
    else:
        # 如果是numpy数组格式，尝试转换
        emotion_label_map = {int(k): str(v) for k, v in label_data.item().items()}
except:
    # 如果加载失败，使用默认标签
    print("Warning: Could not load emotion labels, using default labels")
    emotion_label_map = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }

def predict_emotion(face_array: np.ndarray) -> str:
    """
    输入：
      - face_array: numpy 数组，形状应为 (1,48,48,1)，值范围 [0,1]
    返回：
      - emotion_text: 预测出的情绪字符串（例如 'Happy'、'Sad' 等）
    """
    try:
        # 模型输出一个 (1,7) 的概率向量
        preds = emo_model.predict(face_array, verbose=0)  # 添加verbose=0减少输出
        idx = int(np.argmax(preds, axis=1)[0])
        emotion_result = emotion_label_map.get(idx, "Unknown")
        # 确保返回字符串
        return str(emotion_result)
    except Exception as e:
        print(f"Emotion prediction error: {e}")
        return "Unknown"
