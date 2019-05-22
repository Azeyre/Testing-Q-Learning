#import tensorflow as tf
#with tf.device('/device:GPU:0'): #Permet d'utiliser le GPU
#  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
#print(sess.run(c))

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

class CNN():        
    def __init__(self):
        #Initilisation du CNN
        classifier = Sequential()    
        #Convolution
        classifier.add(Convolution2D(filters=64, kernel_size=9, strides=3, input_shape=(100,75,1), activation="relu"))
        classifier.add(Convolution2D(filters=32, kernel_size=6, strides=2, activation="relu"))    
        #Pooling
        classifier.add(MaxPooling2D(pool_size=(2,2)))    
        #Flatten
        classifier.add(Flatten())    
        #Dense
        classifier.add(Dense(units=512, activation="relu"))
        classifier.add(Dense(units=9, activation="softmax"))    
        #Compile
        classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])       

import numpy as np
from grabscreen import grab_screen
import cv2
import time
import os

screen = grab_screen(region=(0,40,800,640)) #Récupération de l'image
screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) #Transforme l'image en noir et blanc
screen = cv2.resize(screen, (100,75)) #Redimensionner l'image a 100,75 (en partant de 800,600 8x)

CNN()