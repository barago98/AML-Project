# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 13:16:09 2021

@author: debod
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
from tqdm import tqdm_notebook as tqdm

################## Import the data ##########################

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'training'
validation_dir = 'validation'
test_dir = 'testing'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size = 1024,
    class_mode='categorical',
    seed=00)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size = 1024,
    class_mode='categorical',
    seed=00)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size = 1,
    class_mode='categorical',
    seed=00)

######################### Modelling #########################

base_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights="imagenet",
                                             input_shape = (64, 64, 3))
base_model.trainable = True
base_model.summary()

model_1 = Sequential([base_model,
                     tf.keras.layers.GlobalAveragePooling2D(),
                     tf.keras.layers.Dense(200, activation = "softmax")])
model_1.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model_1.summary()

history = model_1.fit(train_generator, epochs=30,
                      validation_data=validation_generator,
                      batch_size=4096, verbose=1)

################## Loss ##########################
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')

################## Accuracy ######################
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')



##################### Alternative  approach ########################

