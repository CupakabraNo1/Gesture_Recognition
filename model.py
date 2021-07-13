#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 14:34:59 2021

@author: cupakabrano1
"""
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, MobileNet
from data_preprocessing import target_val, img_data

import numpy as np

model = MobileNet(weights='imagenet')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# compile the model
# train the model
model.fit(x=np.array(img_data, np.float32), y=np.array(list(map(int, target_val)), np.float32), epochs=5)
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_keras_dir")


# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('detect.tflite', 'wb') as f:
  f.write(tflite_model)
