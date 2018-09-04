#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/28 18:15
# @Author  : Hongjian Kang
# @File    : gen_data.py
import tensorflow as tf


from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, activity_l1
# from keras.layers.core import Dense

def load_model(nb_classes=100, path_to_weights=None):
    model = Sequential()
    model.add(Convolution2D(32, 3, strides=2, padding='same', input_shape=(224, 224, 3)))    # output = 112*112*32
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), (2, 2))) # output = 56*56*32

    model.add(Convolution2D(64, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), (2, 2)))  # output = 28*28*64

    model.add(Convolution2D(128, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), (2, 2)))  # output = 14*14*128

    model.add(Convolution2D(192, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), (2, 2)))  # output = 7*7*192

    model.add(Convolution2D(256, 3, strides=2, padding='same'))     # output = 4*4*256
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), (2, 2))) # output = 2*2*256


    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model