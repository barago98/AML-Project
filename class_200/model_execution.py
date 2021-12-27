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
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from gp2d import *
#!nvidia-smi
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
    target_size=(128, 128),
    batch_size = 32,
    class_mode='categorical',
    seed=00)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size = 32,
    class_mode='categorical',
    seed=00)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size = 1,
    class_mode='categorical',
    seed=00)

######################### Modelling #########################

'''
1. Resnet
'''
base_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights="imagenet",
                                             input_shape = (128, 128, 3))
base_model = tf.keras.models.Model(base_model.input, base_model.layers[-3].output)
base_model.trainable = True
base_model.summary()

model_1 = Sequential([base_model,
                     GeMPooling2D(),
                     tf.keras.layers.Dense(200, activation = "softmax")])
model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model_1.summary()



file_path = "checkpoint_resnet200.h5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, 
                                                      save_best_only=True, 
                                                      monitor="val_accuracy",
                                                      mode='auto', verbose=1)

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

history = model_1.fit(train_generator, epochs=100,
                      validation_data=validation_generator,
                      batch_size=256, verbose=True, 
                      callbacks=[model_checkpoint, early_stop])
# Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc="upper right")
plt.grid()
# Accuracy 
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc="upper right")
plt.grid()

resnet200 = tf.keras.models.load_model(file_path, custom_objects={'GeMPooling2D': GeMPooling2D})

resnet200.evaluate(test_generator)
resnet200.evaluate(validation_generator)

'''
2. MobileNetV2
'''
base_model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                             input_shape = (128, 128, 3))
base_model = tf.keras.models.Model(base_model.input, base_model.layers[-3].output)
base_model.trainable = True
base_model.summary()

model_1 = Sequential([base_model,
                     GeMPooling2D(),
                     tf.keras.layers.Dense(200, activation = "softmax")])
model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model_1.summary()

file_path = "checkpoint_mobilenet.h5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, 
                                                      save_best_only=True, 
                                                      monitor="val_accuracy",
                                                      mode='auto', verbose=1)

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)

history = model_1.fit(train_generator, epochs=100,
                      validation_data=validation_generator,
                      batch_size=256, verbose=True, 
                      callbacks=[model_checkpoint,early_stop])
# Loss 
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc="upper right")
plt.grid()
# Accuracy 
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc="upper right")
plt.grid()

'''
3. VGG16
'''
base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet",
                                             input_shape = (128, 128, 3))
base_model = tf.keras.models.Model(base_model.input, base_model.layers[-3].output)
base_model.trainable = True
base_model.summary()

model_1 = Sequential([base_model,
                     GeMPooling2D(),
                     tf.keras.layers.Dense(200, activation = "softmax")])
model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model_1.summary()

file_path = "checkpoint_vgg16.h5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, 
                                                      save_best_only=True, 
                                                      monitor="val_accuracy",
                                                      mode='auto', verbose=1)

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)

history = model_1.fit(train_generator, epochs=100,
                      validation_data=validation_generator,
                      batch_size=256, verbose=True, 
                      callbacks=[model_checkpoint, early_stop])

'''
4. DenseNet121
'''

base_model = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet",
                                             input_shape = (128, 128, 3))
base_model = tf.keras.models.Model(base_model.input, base_model.layers[-3].output)
base_model.trainable = True
base_model.summary()

model_1 = Sequential([base_model,
                     GeMPooling2D(),
                     tf.keras.layers.Dense(200, activation = "softmax")])
model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model_1.summary()

file_path = "checkpoint_densenet.h5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, 
                                                      save_best_only=True, 
                                                      monitor="val_accuracy",
                                                      mode='auto', verbose=1)

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)

history = model_1.fit(train_generator, epochs=100,
                      validation_data=validation_generator,
                      batch_size=256, verbose=True, 
                      callbacks=[model_checkpoint, early_stop])