from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Flatten, Dense, Activation
)
from keras.optimizers import Adam

def build_face_model(input_shape=(64, 64, 1), num_classes=None):
    if num_classes is None:
        raise ValueError("必须指定num_classes（类别数量）")

    model = Sequential()

    # 卷积块1
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))  # 增加卷积核数量
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 卷积块2
    model.add(Conv2D(128, (3, 3), padding='same'))  # 增加卷积核数量
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 卷积块3
    model.add(Conv2D(256, (3, 3), padding='same'))  # 增加卷积核数量
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 全连接层
    model.add(Flatten())
    model.add(Dense(512))  # 增加神经元数量
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 输出层
    model.add(Dense(num_classes, activation='softmax'))

    # 编译模型
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),  # 减小学习率
        metrics=['accuracy']
    )

    return model
