from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Activation, Dropout, Flatten, Dense
)

def build_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    构建一个 9 层（可训练）的小型 CNN，用于 FER2013 情绪识别（7 类）。
    返回：已编译好的模型（可直接训练或加载权重）
    """
    model = Sequential()

    # 第一组卷积块
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 第二组卷积块
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 第三组卷积块
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 全连接块
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 输出层
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # 编译模型（训练时使用）
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

