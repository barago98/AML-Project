# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 14:18:28 2021

@author: debod
"""
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
class GeMPooling2D(tf.keras.layers.Layer):
    def __init__(self, init_norm = 3, train_p=False,**kwargs):
        super().__init__()
        assert init_norm > 0
        self.init_norm = float(init_norm)
   
    def build(self, input_shape):
        self.p = self.add_weight(name="norms", shape=(1,),
                                 initializer=keras.initializers.constant(self.init_norm),
                                 trainable=True)
        #super(GeMPooling2D, self).build(None)
       
    def call(self, inputs: tf.Tensor, **kwargs):
        #p = self.kernel*self.init_norm
        inputs = tf.clip_by_value(inputs, clip_value_min=1e-6, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1.0/self.p)
       
        return inputs
   
    def get_config(self):
        config = super(GeMPooling2D, self).get_config()
        config.update({'init_norm':self.init_norm})
        return config