#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/31 16:52
# @Author  : Hongjian Kang
# @File    : data_augmentation.py

import numpy as np
import cv2

def data_augmentation(image_array, label_array):
    # 对input_array的每一张进行扩充，旋转-45,-30,-15,0,15,30,45放入新array中，同时label放入新array中
    if not image_array.shape[0] == label_array.shape[0]:
        raise ValueError('dimensions not match between two arrays')

    slice, cols, rows = image_array.shape[0], image_array.shape[1], image_array.shape[2]
    new_image_array = []
    new_label_array = []
    N1 = cv2.getRotationMatrix2D((cols / 2, rows / 2), -30, 1)
    N2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), -20, 1)
    N3 = cv2.getRotationMatrix2D((cols / 2, rows / 2), -10, 1)
    N4 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
    N5 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    N6 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 20, 1)
    N7 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
    N = [N1, N2, N3, N4, N5, N6, N7]

    for i in range(slice):
        print('processing  {} / {}'.format(i+1, slice))
        for j in range(len(N)):
            new_image_array.append(cv2.warpAffine(image_array[i], N[j], (rows, cols)))
            new_label_array.append(label_array[i])

    new_image_array = np.array(new_image_array)
    new_label_array = np.array(new_label_array)

    return new_image_array, new_label_array

def main():
    image_array = np.load('AAI_Dataset.npz')['trainSet']
    label_array = np.load('AAI_Dataset.npz')['trainLabel']
    x_test = np.load('AAI_Dataset.npz')['testSet']
    y_test = np.load('AAI_Dataset.npz')['testLabel']
    label_array = np.reshape(label_array, (label_array.shape[0], label_array.shape[1]))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
    new_image_array, new_label_array = data_augmentation(image_array, label_array)

    np.savez('final_AAI_Dataset.npz', trainSet=new_image_array, trainLabel=new_label_array, testSet=x_test, testLabel=y_test)

if __name__ == '__main__':
    main()