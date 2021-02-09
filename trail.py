#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:08:43 2021

@author: srinivas
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Keras
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

MODEL_PATH ='pneumonia_vgg16.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    # x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    # preds=np.argmax(preds, axis=1)
    preds = int(preds[0][0])
    
    if preds==0:
        preds= "The Person is Affected By PNEUMONIA"
    else:
        preds="Result is Normal"
        
    
    
    return preds

file_path = 'Datasets/val/NORMAL/NORMAL2-IM-1437-0001.jpeg'
print(model_predict(file_path, model))
