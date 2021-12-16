# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 23:25:18 2021

@author: trina
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import PIL
import shutil
import random
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

tf.test.is_gpu_available()
tf.config.list_physical_devices(
    device_type=None
)

train_csv = pd.read_csv("train.csv")
train_csv.head()
num_classes = len(train_csv["landmark_id"].unique())
num_classes



# put .jpg into the file name
def add_txt(fn):
    return fn+'.jpg'

train_csv['id'] = train_csv['id'].apply(add_txt)

# choose those labels with more than 200 images, and choose the first 200 images of each label
# move every training files to the same folder

if not os.path.exists('training'):
    os.mkdir('training')
if not os.path.exists('validation'):
    os.mkdir('validation')
if not os.path.exists('testing'):
    os.mkdir('testing')    

label_list = train_csv['landmark_id'].unique()
cnt = 0
final_label_list = []

for label in list(label_list): # label order by random
    file_list = list(train_csv['id'][train_csv['landmark_id']==label])
    if len(file_list) >= 200:
        final_label_list.append(label)
        if not os.path.exists('training/'+str(label)):
            os.mkdir('training/'+str(label))
        if not os.path.exists('validation/'+str(label)):
            os.mkdir('validation/'+str(label))
        if not os.path.exists('testing/'+str(label)):
            os.mkdir('testing/'+str(label))
        for file in file_list[:120]:  # 120 files for training
            src = 'train/'+file[0]+'/'+file[1]+'/'+file[2]+'/'+file
            dst = 'training/'+str(label)+'/'+file
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
        for file in file_list[120:160]: # 40 files for validation
            src = 'train/'+file[0]+'/'+file[1]+'/'+file[2]+'/'+file
            dst = 'validation/'+str(label)+'/'+file
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
        for file in file_list[160:200]: # 40 files for validation
            src = 'train/'+file[0]+'/'+file[1]+'/'+file[2]+'/'+file
            dst = 'testing/'+str(label)+'/'+file
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
        cnt += 1
    if cnt == 600:# only need 500 labels
        break
