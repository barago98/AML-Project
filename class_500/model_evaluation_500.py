# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 01:59:02 2021

@author: trina
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from gp2d import *

################### Import the dataset ################################
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
    target_size=(120, 120),
    batch_size = 32,
    class_mode='categorical',
    seed=42)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(120, 120),
    batch_size = 32,
    class_mode='categorical',
    seed=42)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(120, 120),
    batch_size = 1,
    class_mode='categorical',
    seed=42)

######################## Modelling #########################

'''
1. ResNet50

'''

base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet",
                                             input_shape = (120, 120, 3))
base_model = Model(base_model.input, base_model.layers[-3].output)
base_model.trainable = True
base_model.summary()

model_1 = Sequential([base_model,
                     GeMPooling2D(),
                     #keras.layers.Flatten(),
                     keras.layers.Dense(491, activation = "softmax")])
model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
model_1.summary()

file_path = "checkpoint_resnet_500.h5"
model_checkpoint = ModelCheckpoint(filepath=file_path, save_best_only=True, monitor="val_accuracy",mode='auto', verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30)

history = model_1.fit(train_generator, epochs=100, validation_data=validation_generator, verbose=True, callbacks=[model_checkpoint, early_stop])

# Accuracy
plt.plot(history.history['accuracy'], label = "Training Accuracy")
plt.plot(history.history['val_accuracy'], label = "Validation Accuracy")
plt.legend(loc = "upper right")
#Loss
plt.plot(history.history['loss'], label = "Training Loss")
plt.plot(history.history['val_loss'], label = "Validation Loss")
plt.legend(loc = "upper right")
# Evaluation
resnet500 = tf.keras.models.load_model(file_path, custom_objects={'GeMPooling2D': GeMPooling2D})
resnet500.evaluate(test_generator)
resnet500.evaluate(validation_generator)

'''
2. MobileNetV2

'''

base_model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                             input_shape = (120, 120, 3))
base_model = Model(base_model.input, base_model.layers[-3].output)
base_model.trainable = True
base_model.summary()

model_1 = Sequential([base_model,
                     GeMPooling2D(),
                     keras.layers.Dense(491, activation = "softmax")])
model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
model_1.summary()

file_path = "checkpoint_mobilenet.h5"
model_checkpoint = ModelCheckpoint(filepath=file_path, save_best_only=True, monitor="val_accuracy",mode='auto', verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30)

history = model_1.fit(train_generator, epochs=100, validation_data=validation_generator, verbose=True, callbacks=[model_checkpoint, early_stop])

# Accuracy
plt.plot(history.history['accuracy'], label = "Training Accuracy")
plt.plot(history.history['val_accuracy'], label = "Validation Accuracy")
plt.legend(loc = "upper right")
#Loss
plt.plot(history.history['loss'], label = "Training Loss")
plt.plot(history.history['val_loss'], label = "Validation Loss")
plt.legend(loc = "upper right")
# Evaluation
Mobilenet500 = tf.keras.models.load_model(file_path, custom_objects={'GeMPooling2D': GeMPooling2D})
Mobilenet500.evaluate(test_generator)
Mobilenet500.evaluate(validation_generator)

'''
3. DenseNet121

'''

base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet",
                                             input_shape = (120, 120, 3))
base_model = Model(base_model.input, base_model.layers[-3].output)
base_model.trainable = True
base_model.summary()

model_1 = Sequential([base_model,
                     GeMPooling2D(),
                     keras.layers.Dense(491, activation = "softmax")])
model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
model_1.summary()

file_path = "checkpoint_densenet.h5"
model_checkpoint = ModelCheckpoint(filepath=file_path, save_best_only=True, monitor="val_accuracy",mode='auto', verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30)

history = model_1.fit(train_generator, epochs=100, validation_data=validation_generator, verbose=True, callbacks=[model_checkpoint, early_stop])

# Accuracy
plt.plot(history.history['accuracy'], label = "Training Accuracy")
plt.plot(history.history['val_accuracy'], label = "Validation Accuracy")
plt.legend(loc = "upper right")
#Loss
plt.plot(history.history['loss'], label = "Training Loss")
plt.plot(history.history['val_loss'], label = "Validation Loss")
plt.legend(loc = "upper right")
# Evaluation
Densenet500 = tf.keras.models.load_model(file_path, custom_objects={'GeMPooling2D': GeMPooling2D})
Densenet500.evaluate(test_generator)
Densenet500.evaluate(validation_generator)

'''
4. VGG16

'''

base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet",
                                             input_shape = (120, 120, 3))
base_model = Model(base_model.input, base_model.layers[-3].output)
base_model.trainable = True
base_model.summary()

model_1 = Sequential([base_model,
                     GeMPooling2D(),
                     keras.layers.Dense(491, activation = "softmax")])
model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
model_1.summary()

file_path = "checkpoint_vgg16.h5"
model_checkpoint = ModelCheckpoint(filepath=file_path, save_best_only=True, monitor="val_accuracy",mode='auto', verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30)

history = model_1.fit(train_generator, epochs=100, validation_data=validation_generator, verbose=True, callbacks=[model_checkpoint, early_stop])

# Accuracy
plt.plot(history.history['accuracy'], label = "Training Accuracy")
plt.plot(history.history['val_accuracy'], label = "Validation Accuracy")
plt.legend(loc = "upper right")
#Loss
plt.plot(history.history['loss'], label = "Training Loss")
plt.plot(history.history['val_loss'], label = "Validation Loss")
plt.legend(loc = "upper right")
# Evaluation
Vgg500 = tf.keras.models.load_model(file_path, custom_objects={'GeMPooling2D': GeMPooling2D})
Vgg500.evaluate(test_generator)
Vgg500.evaluate(validation_generator)

