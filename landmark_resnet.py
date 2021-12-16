# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 19:54:59 2021

@author: trina
"""

import pandas as pd
import tensorflow as tf
import os
import shutil
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint


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
    seed=42)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size = 32,
    class_mode='categorical',
    seed=42)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size = 1,
    class_mode='categorical',
    seed=42)

# ResNet50 model
base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet",
                                             input_shape = (64, 64, 3))
base_model = Model(base_model.input, base_model.layers[-3].output)
base_model.trainable = True
base_model.summary()

model_1 = Sequential([base_model,
                     GlobalAveragePooling2D(),
                     keras.layers.Dense(491, activation = "softmax")])
model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
model_1.summary()

file_path = "checkpoint.h5"
model_checkpoint = ModelCheckpoint(filepath=file_path, save_best_only=True, monitor="val_accuracy",mode='auto', verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

history = model_1.fit(train_generator, epochs=100, validation_data=validation_generator, verbose=True, callbacks=[model_checkpoint])

best_model = keras.models.load_model(file_path)

best_model.evaluate(test_generator)

plt.plot(history.history['accuracy'], label = "Training Accuracy")
plt.plot(history.history['val_accuracy'], label = "Validation Accuracy")
plt.legend(loc = "upper right")

plt.plot(history.history['loss'], label = "Training Loss")
plt.plot(history.history['val_loss'], label = "Validation Loss")
plt.legend(loc = "upper right")

# Test labels

test_label = list(test_generator.class_indices.keys())
test_label[:10]

# Test prediction
y_pred = best_model.predict(test_generator)
y_pred.shape
    
def get_label(arr):
    for i in range(len(arr)):
        if arr[i] == max(arr):
            return test_label[i]

# Label from original data
true_labels=[]
for i in range(19640):
    labels=get_label(test_generator[i][1][0])
    true_labels.append(labels)
    
# Predicting the label 
predict_labels=[]
for i in range(19640):
    pred_labels=get_label(best_model.predict(test_generator[i][0])[0])
    predict_labels.append(pred_labels)

true_labels=[int(i) for i in true_labels]
predict_labels=[int(i) for i in predict_labels]    

##### Acccuracy #####
get_ones= [i/j for i,j in zip(true_labels,predict_labels)]  
accuracy=(get_ones.count(1)/len(get_ones))*100   
 
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
for i in range(19640):
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


 





















