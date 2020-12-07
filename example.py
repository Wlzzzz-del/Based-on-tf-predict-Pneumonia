import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import RMSprop
import numpy as np
# 将生成器指向训练集和测试集目录
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    './smalldata/train',
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    './smalldata/test',
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
# 定义cnn模型

model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(16, (3,3), activation='relu',input_shape=(150,150,1)),
     tf.keras.layers.MaxPool2D(2,2),
     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
     tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
     tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(512, activation='relu'),
     tf.keras.layers.Dense(1, activation='sigmoid')]
)
# 编译模型
model.compile(loss = 'binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])
# fit 训练模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)
# predict预测模型结果
# predict_datagen = ImageDataGenerator(rescale=1./255)
# predict_generator = predict_datagen.flow_from_directory(
#     './chest_xray/val',
#     target_size=(150,150),
#     class_mode='binary'
# )
# predict
img = load_img('./chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg', target_size=(150, 150))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=1)
print(classes[0])