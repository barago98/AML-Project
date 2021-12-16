# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:49:21 2021

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

!nvidia-smi
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
    batch_size = 32,
    class_mode='categorical',
    seed=00)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size = 32,
    class_mode='categorical',
    seed=00)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size = 1,
    class_mode='categorical',
    seed=00)

######################### Modelling #########################

base_model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                             input_shape = (64, 64, 3))
base_model = tf.keras.models.Model(base_model.input, base_model.layers[-3].output)
base_model.trainable = True
base_model.summary()

model_1 = Sequential([base_model,
                     tf.keras.layers.GlobalAveragePooling2D(),
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

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

history = model_1.fit(train_generator, epochs=100,
                      validation_data=validation_generator,
                      batch_size=256, verbose=True, 
                      callbacks=[model_checkpoint])


################## Loss ##########################
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc="upper right")
plt.grid()

################## Accuracy ######################
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc="upper right")
plt.grid()

best_model = keras.models.load_model(file_path)

best_model.evaluate(test_generator)

################## Test labels ###################

test_label = list(test_generator.class_indices.keys())
test_label[:10]

# Test prediction
y_pred = best_model.predict(test_generator)
y_pred.shape
    
#def to_original_label(new_label): 
 #   return int(test_label[new_label])
    
def get_label(arr):
    for i in range(len(arr)):
        if arr[i] == max(arr):
            return test_label[i]

# Label from original data
true_labels=[]
for i in range(8000):
    labels=get_label(test_generator[i][1][0])
    true_labels.append(labels)
# Predicting the label 
predict_labels=[]
for i in range(8000):
    pred_labels=get_label(best_model.predict(test_generator[i][0])[0])
    predict_labels.append(pred_labels)

true_labels=[int(i) for i in true_labels]
predict_labels=[int(i) for i in predict_labels]    
##### Acccuracy #####
get_ones= [i/j for i,j in zip(true_labels,predict_labels)]  
accuracy=(get_ones.count(1)/len(get_ones))*100    
matrix = classification_report(true_labels,predict_labels)
print('Classification report : \n',matrix)

# Test evaluation for image index 20

image_index = 8 # choose a image (0-8000)
image = test_generator[image_index][0] 
image = image.reshape((64,64,3))
plt.imshow(image)
plt.show()


get_label(test_generator[image_index][1][0])
label = get_label(best_model.predict(test_generator[image_index][0])[0])
label
image_list = [] # image with the same label
for i in range(8000):
    if get_label(test_generator[i][1][0]) == label:
        image_list.append(i)
        
if len(image_list) > 10:
    for i in range(10):
        if i != image_index:
            image = test_generator[image_list[i]][0]
            image = image.reshape((64,64,3))
            plt.imshow(image)
            plt.show()
        
else:
    for i in range(len(image_list)):
        if i != image_index:
            image = test_generator[image_list[i]][0]
            image = image.reshape((64,64,3))
            plt.imshow(image)
            plt.show()