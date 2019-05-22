GAMMA = 0.99 # decay rate of past observations original 0.99
OBSERVATION = 50000. # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
ACTIONS = 9



from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os
import numpy as np
from grabscreen import grab_screen
import cv2
import time

def buildmodel():
    print("Now we build the model")
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
    print("We finish building the model")
    return classifier

def mouse_to_output(i):
    output=[0,0,0,0,0,0,0,0,0]
    output[i]=1
    return output

screen = grab_screen(region=(0,40,800,640)) #Récupération de l'image
screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) #Transforme l'image en noir et blanc
screen = cv2.resize(screen, (100,75)) #Redimensionner l'image a 100,75 (en partant de 800,600 8x)