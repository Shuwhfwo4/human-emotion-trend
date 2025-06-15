import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, Frame, Text, Scrollbar, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from datetime import datetime
import matplotlib
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入人脸识别和情绪识别模块
from face_recognition_celeba import face_detector, predict_face_name
from emotion_recognition import predict_emotion

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头，程序退出。")
    exit(1)

# 情绪映射字典（按积极程度排序）
EMOTION_MAP = {
    'Happy': 6,
    'Surprise': 5,
    'Neutral': 4,
    'Sad': 3,
    'Fear': 2,
    'Disgust': 1,
    'Angry': 0
}

# 反转映射用于显示
EMOTION_REVERSE_MAP = {v: k for k, v in EMOTION_MAP.items()}

class EmotionTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸情绪识别与跟踪")
        self.root.geometry("1000x800")

        self.last_emotion = None
        self.last_person = None
        self.emotion_history = []  # 存储情绪变化历史 [(timestamp, emotion, person)]
        self.last_record_time = datetime.now()

        self.main_frame = Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = Label(self.main_frame)
        self.video_label.pack(pady=10)

        self.control_frame = Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.chart_frame = Frame(self.main_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.history_frame = Frame(self.main_frame)
        self.history_frame.pack(fill=tk.BOTH, padx=10, pady=10)

        self.history_label = Label(self.history_frame, text="情绪变化记录:")
        self.history_label.pack(anchor=tk.W)

        self.history_text = Text(self.history_frame, height=8)
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = Scrollbar(self.history_frame, command=self.history_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text.config(yscrollcommand=self.scrollbar.set)

        self.download_btn = Button(self.control_frame, text="下载情绪图表", command=self.save_emotion_chart)
        self.download_btn.pack(side=tk.LEFT, padx=5)

        self.exit_btn = Button(self.control_frame, text="退出", command=self.exit_program)
        self.exit_btn.pack(side=tk.RIGHT, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.update_chart()

        self.update_frame()

    def record_emotion_change(self, emotion, person):
        """记录情绪变化"""
        current_time = datetime.now()
        time_diff = (current_time - self.last_record_time).total_seconds()

        # 如果情绪变化或距离上次记录超过30秒，则记录新情绪
        if emotion != self.last_emotion or person != self.last_person or time_diff > 30:
            timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            self.emotion_history.append((timestamp_str, emotion, person))
            self.last_emotion = emotion
            self.last_person = person
            self.last_record_time = current_time

            # 更新历史记录文本框
            self.history_text.insert(tk.END, f"{timestamp_str} - {person} - {emotion}\n")
            self.history_text.see(tk.END)  # 滚动到最新记录

            # 更新图表
            self.update_chart()

    def update_chart(self):
        """更新情绪变化图表"""
        self.ax.clear()

        if self.emotion_history:
            # 准备数据
            timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts, _, _ in self.emotion_history]
            emotions = [EMOTION_MAP[em] for _, em, _ in self.emotion_history]
            persons = [p for _, _, p in self.emotion_history]

            # 绘制折线图
            self.ax.plot(timestamps, emotions, 'o-', markersize=8)

            # 在每个点上方添加人名
            for i, (ts, em, person) in enumerate(self.emotion_history):
                # 将"未录入"统一显示为"Unknown"
                display_name = "Unknown" if person == "未录入" else person

                # 根据点的位置调整文本位置
                offset = 0.3 if i % 2 == 0 else 0.5  # 交替位置避免重叠
                self.ax.text(
                    datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"),
                    EMOTION_MAP[em] + offset,
                    display_name,
                    fontsize=8,
                    ha='center',
                    rotation=45
                )

            # 设置Y轴刻度和标签
            self.ax.set_yticks(list(EMOTION_MAP.values()))
            self.ax.set_yticklabels([EMOTION_REVERSE_MAP[l] for l in EMOTION_MAP.values()])

            # 设置图表标题和标签（使用中文）
            self.ax.set_title('情绪变化趋势', fontsize=12)
            self.ax.set_xlabel('时间', fontsize=10)
            self.ax.set_ylabel('情绪', fontsize=10)
            self.ax.grid(True, linestyle='--', alpha=0.7)

            # 自动调整X轴时间格式
            self.fig.autofmt_xdate()

        self.canvas.draw()

    def save_emotion_chart(self):
        """保存情绪图表到文件"""
        if not self.emotion_history:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("PDF文件", "*.pdf"), ("所有文件", "*.*")],
            title="保存情绪图表"
        )

        if file_path:
            # 创建新的图表用于保存
            fig, ax = plt.subplots(figsize=(10, 6))

            # 设置中文字体支持
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            # 准备数据
            timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts, _, _ in self.emotion_history]
            emotions = [EMOTION_MAP[em] for _, em, _ in self.emotion_history]
            persons = [p for _, _, p in self.emotion_history]

            # 绘制折线图
            ax.plot(timestamps, emotions, 'o-', markersize=8)

            # 在每个点上方添加人名
            for i, (ts, em, person) in enumerate(self.emotion_history):
                # 将"未录入"统一显示为"Unknown"
                display_name = "Unknown" if person == "未录入" else person

                # 根据点的位置调整文本位置
                offset = 0.3 if i % 2 == 0 else 0.5  # 交替位置避免重叠
                ax.text(
                    datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"),
                    EMOTION_MAP[em] + offset,
                    display_name,
                    fontsize=9,
                    ha='center',
                    rotation=45
                )

            # 设置Y轴刻度和标签
            ax.set_yticks(list(EMOTION_MAP.values()))
            ax.set_yticklabels([EMOTION_REVERSE_MAP[l] for l in EMOTION_MAP.values()])

            # 设置图表标题和标签（使用中文）
            ax.set_title('情绪变化趋势', fontsize=14)
            ax.set_xlabel('时间', fontsize=12)
            ax.set_ylabel('情绪', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)

            # 自动调整X轴时间格式
            fig.autofmt_xdate()

            # 保存图表
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    def update_frame(self):
        ret, frame = cap.read()
        if ret:
            # 转为灰度图，用于人脸检测和情绪识别预处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 人脸检测（Haar Cascade）
            faces_rects = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            current_emotion = None
            current_person = None

            for (x, y, w, h) in faces_rects:
                # 裁剪灰度人脸 ROI
                face_roi_gray = gray[y: y + h, x: x + w]

                # 人脸识别：预测姓名或 "Unknown"
                try:
                    person_name = predict_face_name(face_roi_gray)
                    # 确保是字符串
                    person_name = str(person_name) if person_name is not None else "Unknown"

                    # 如果是第一次检测到人脸，设置为当前人物
                    if current_person is None:
                        current_person = person_name
                except Exception as e:
                    print(f"Face recognition error: {e}")
                    person_name = "Unknown"

                # 情绪识别：预处理 → 预测
                try:
                    # 把灰度 ROI resize 为 48×48
                    face_resized = cv2.resize(face_roi_gray, (48, 48))
                    # 归一化到 [0,1]
                    face_array = face_resized.astype("float32") / 255.0
                    # 增加 batch 维度和通道维度，形状变为 (1,48,48,1)
                    face_array = np.expand_dims(face_array, axis=0)  # (1,48,48)
                    face_array = np.expand_dims(face_array, axis=-1)  # (1,48,48,1)
                    # 预测情绪（返回的是字符串）
                    emotion_text = predict_emotion(face_array)
                    emotion_text = str(emotion_text) if emotion_text is not None else "Unknown"

                    # 如果是第一次检测到情绪，设置为当前情绪
                    if current_emotion is None:
                        current_emotion = emotion_text
                except Exception as e:
                    print(f"Emotion recognition error: {e}")
                    emotion_text = "Unknown"

                # 在原图上绘制结果
                try:
                    # 画人脸框（绿色）
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # 在框上方写姓名
                    cv2.putText(
                        frame,
                        person_name,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    # 在框下方写情绪（蓝色）
                    cv2.putText(
                        frame,
                        emotion_text,
                        (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                except Exception as e:
                    print(f"Drawing error: {e}")
                    # 至少画一个框
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 记录情绪变化
            if current_emotion and current_emotion != "Unknown" and current_person:
                self.record_emotion_change(current_emotion, current_person)

            # 将OpenCV图像转换为Tkinter可用的图像
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)

            # 更新标签的图像
            self.video_label.config(image=img)
            self.video_label.image = img

        # 每隔10毫秒更新一次帧
        self.root.after(10, self.update_frame)

    def exit_program(self):
        cap.release()
        self.root.destroy()


# 创建主窗口并启动应用
root = tk.Tk()
app = EmotionTrackerApp(root)
root.mainloop()