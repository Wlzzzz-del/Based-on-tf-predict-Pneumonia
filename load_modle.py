from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
# 加载已经训练好的模型
model = keras.models.load_model('./model')
print(model.summary())
# 记录工作路径
work_path = os.getcwd()
os.chdir('./chest_xray/val/NORMAL')
print("以下是正常图像的预测结果：")
for i in os.listdir():
    img = load_img(i, target_size=(120, 120), color_mode='grayscale')
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=1)
    print(classes[0])

# 返回工作路径
os.chdir(work_path)
os.chdir('./chest_xray/val/PNEUMONIA')
print("以下是不正常图像的预测结果：")
for i in os.listdir():
    img = load_img(i, target_size=(120, 120), color_mode='grayscale')
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=1)
    print(classes[0])
