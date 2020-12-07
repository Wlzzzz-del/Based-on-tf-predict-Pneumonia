import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

# 设置中文字体
font={'family':'Microsoft Yahei','weight':'bold'}
plt.rc('font',**font)

def sp_noise(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.001):
    '''
      添加高斯噪声
      mean : 均值
      var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out

def move_pic(image, distance):
    # 平移
    # distance：距离
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    mode = imgInfo[2]
    dst = np.zeros([height, width, mode], np.uint8)
    for i in range( height ):
        for j in range( width-distance):
            dst[i,j+distance] = img[i,j]
    return dst

def rotation_pic(image, angle, scale):
    # 旋转
    # angle: 角度
    # scale： 尺度
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    mode = imgInfo[2]
    dst = np.zeros([height, width, mode], np.uint8)
    matRotate = cv2.getRotationMatrix2D((height*0.5, width*0.5), angle, scale)
    dst = cv2.warpAffine(img, matRotate, (height, width))
    return dst

def mirror_pic(image):
    # 镜像
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    mode = imgInfo[2]
    dst = np.zeros([height, width, mode], np.uint8)
    for i in range( height ):
        for j in range( width):
            dst[height-i-1,j] = img[i,j]
    return dst

# 读取图片
img = cv2.imread('./chest_xray/test/NORMAL/IM-0001-0001.jpeg')

plt.imshow(mirror_pic(img))

plt.show()
# cv2.imwrite()写入图片
