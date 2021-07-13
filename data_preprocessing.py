#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 09:52:26 2021

@author: cupakabrano1
"""

import os
import cv2
import numpy as np


IMG_HEIGHT = IMG_WIDTH = 224

def create_dataset(img_folder):

    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):

            image_path = os.path.join(img_folder, dir1,  file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),
                               interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name

img_data, class_name = create_dataset(r'./training_images/')

target_dict = {k: v for v, k in enumerate(np.unique(class_name))}

target_val = [target_dict[class_name[i]] for i in range(len(class_name))]
