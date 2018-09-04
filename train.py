#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/30 16:26
# @Author  : Hongjian Kang
# @File    : train.py

from model import load_model
from keras.utils import plot_model
import numpy as np
from keras.optimizers import Adam
from loss_history import LossHistory



def train():
    model = load_model()
    model.summary()
    plot_model(model, to_file='model.png')
    json_string = model.to_json()
    open('model_architecture.json', 'w').write(json_string)

    x_train = np.load('final_AAI_Dataset.npz')['trainSet']
    y_train = np.load('final_AAI_Dataset.npz')['trainLabel']
    # y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
    x_test = np.load('final_AAI_Dataset.npz')['testSet']
    y_test = np.load('final_AAI_Dataset.npz')['testLabel']
    # y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

    # x_train = (x_train - np.mean(x_train)) / np.std(x_train)
    # y_train = (y_train - np.mean(y_train)) / np.std(y_train)
    # x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    # y_test = (y_test - np.mean(y_test)) / np.std(y_test)

    print('Training....................')
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    history = LossHistory()
    model.fit(x_train, y_train,
              batch_size=64, epochs=200,
              validation_data=(x_test, y_test),
              callbacks=[history])
    model.save_weights('model_weights.h5')

    history.loss_plot('epoch')




def main():
    train()

if __name__ == '__main__':
    main()
