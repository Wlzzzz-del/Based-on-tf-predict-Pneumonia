### TAKEN FROM https://www.kaggle.com/teyang/pneumonia-detection-resnets-pytorch

from matplotlib.image import imread
from matplotlib import pyplot as plt
import seaborn as sns
import random
import cv2
import copy

import os

# Function for plotting samples
def plot_samples(samples):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
    for i in range(len(samples)):
        image = cv2.cvtColor(imread(samples[i]), cv2.COLOR_BGR2RGB)
        ax[i//5][i%2].imshow(image)
        if i<5:
            ax[i//5][i%2].set_title("Normal", fontsize=20)
        else:
            ax[i//5][i%2].set_title("Pneumonia", fontsize=20)
        ax[i//5][i%2].axis('off')
## Plot training samples
data_dir = './chest_xray'
rand_samples = random.sample([os.path.join(data_dir+'/train/NORMAL', filename)
                              for filename in os.listdir(data_dir+'/train/NORMAL')], 5) + \
    random.sample([os.path.join(data_dir+'/train/PNEUMONIA', filename)
                   for filename in os.listdir(data_dir+'/train/PNEUMONIA')], 5)

plot_samples(rand_samples)
# plt.suptitle('Samples', fontsize=30)
plt.show()