# -*- coding: utf-8 -*-
"""
- https://github.com/john-fante
- https://www.kaggle.com/banddaniel

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2


# Creating images and resizing
main_image_path = '/content/drive/MyDrive/Colab Notebooks/new_eye_dataset/Eye Conjunctiva Segmentation Dataset/Images'

for i in os.listdir(main_image_path):
  img = cv2.imread(os.path.join(main_image_path, i ))
  dim = (400, 400)
  resized = cv2.resize(img , dim, interpolation = cv2.INTER_AREA)
  cv2.imwrite(os.path.join('/content/drive/MyDrive/Colab Notebooks/new_eye_dataset/images',i), resized)


# Creating mask images and resizing
main_mask_path = '/content/drive/MyDrive/Colab Notebooks/new_eye_dataset/Eye Conjunctiva Segmentation Dataset/Masks Annotator 1'

for i in os.listdir(main_mask_path):
  img = cv2.imread(os.path.join(main_mask_path, i ))[:,::,2]
  img = img / 255.0
  dim = (400, 400)
  resized = cv2.resize(img , dim, interpolation = cv2.INTER_AREA)
  f, bw_img = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY)
  cv2.imwrite(os.path.join('/content/drive/MyDrive/Colab Notebooks/new_eye_dataset/masks',i), bw_img)

# Convering images and masks to npy files
images_npy = np.zeros([547, 400, 400])
masks_npy = np.zeros([547, 400, 400])

images_path = '/content/drive/MyDrive/Colab Notebooks/new_eye_dataset/images'
masks_path = '/content/drive/MyDrive/Colab Notebooks/new_eye_dataset/masks'

count = 0
for i in sorted(os.listdir('/content/drive/MyDrive/Colab Notebooks/new_eye_dataset/images')):
    img = cv2.imread(os.path.join(images_path, i ))
    mask = cv2.imread(os.path.join(masks_path, i ))[:,::,1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    images_npy[count] = gray
    masks_npy[count] = mask
    count = count + 1


np.save('/content/drive/MyDrive/Colab Notebooks/new_eye_dataset/images.npy ',images_npy )
np.save('/content/drive/MyDrive/Colab Notebooks/new_eye_dataset/masks.npy ',masks_npy )
