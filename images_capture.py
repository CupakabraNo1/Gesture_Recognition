#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 09:52:26 2021

@author: cupakabrano1
"""

import os
import sys
import cv2 

try:
    images_count = int(sys.argv[1])
    images_label = sys.argv[2]
except Exception:
    print("\nBad syntax. Running this file needs two input parameters. Example:\n")
    print("python images_capture.py 100 mute")
    sys.exit()
    
font = cv2.FONT_HERSHEY_PLAIN
start = False

images_folder = 'training_images'
label_name = os.path.join(images_folder, images_label)
count = image_name = 0

try: 
    os.mkdir(images_folder)
except FileExistsError:
    pass
try:
    os.mkdir(label_name)
except FileExistsError:
    image_name = len(os.listdir(label_name))

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)

while True:
    ret, image = video.read()
    image = cv2.flip(image, 1)
    
    if not ret:
        continue
    
    if count == images_count:
        break
    
    cv2.rectangle(image, (200, 200), (424, 424), (255, 255, 255), 2)
    
    if start:
        region_of_interest = image[200:424, 200:424]
        save_path = os.path.join(label_name, '{}.jpg'.format(image_name + 1))
        cv2.imwrite(save_path, region_of_interest)
        image_name += 1
        count += 1
    
    cv2.putText(image, "Fit the gesture inside the white box and Press 's' key to start clicking pictures",
            (20, 30), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "Press 'q' to exit.",
            (20, 60), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "Image Count: {}".format(count),
            (20, 100), font, 1, (12, 20, 200), 2, cv2.LINE_AA)
    cv2.imshow("Get Training Images", image)
    
    k = cv2.waitKey(10)
    if k==ord('q'):
            break
    if k == ord('s'):
        start = not start
    
video.release()
# cv2.destroyAllWindows()
