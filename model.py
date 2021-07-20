#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 14:34:59 2021

@author: cupakabrano1
"""
from data_preprocessing import target_val, img_data
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np


filter1 = 32
filter2 = 2*filter1

def createModel(n_classes, input_shape, kernel_size=(3, 3), pool_size=(2, 2), padding='same'):
    model = Sequential(name='SequentialModel')

    model.add(layers.Conv2D(filter1, kernel_size, padding=padding, activation='relu', input_shape=input_shape, name='CONV11'))
    model.add(layers.Conv2D(filter1, kernel_size, activation='relu', name='CONV12'))
    model.add(layers.MaxPooling2D(pool_size=pool_size, name='MPooling1'))

    model.add(layers.Conv2D(filter2, kernel_size, padding=padding, activation='relu', name='CONV21'))
    model.add(layers.Conv2D(filter2, kernel_size, activation='relu', name='CONV22'))
    model.add(layers.MaxPooling2D(pool_size=pool_size, name='MPooling2'))

    model.add(layers.Conv2D(filter2, kernel_size, padding=padding, activation='relu', name='CONV31'))
    model.add(layers.Conv2D(filter2, kernel_size, activation='relu', name='CONV32'))
    model.add(layers.MaxPooling2D(pool_size=pool_size, name='MPooling3'))
    model.add(layers.Dropout(0.25, name='DROP'))

    model.add(layers.Flatten(name='FLATTEN'))

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.LeakyReLU(name='LEAKY'))
    model.add(layers.Dense(n_classes, activation='softmax', name="OUTPUT"))

    return model

def prepareModel(model):
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=np.array(img_data, np.float32), y=np.array(list(map(int, target_val)), np.float32), epochs=5)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('detect.tflite', 'wb') as f:
      f.write(tflite_model)

model = prepareModel(model = createModel(n_classes = 5, input_shape =(224, 224, 3)))