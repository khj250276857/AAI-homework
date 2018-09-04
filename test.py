#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/2 9:31
# @Author  : Hongjian Kang
# @File    : test.py

import numpy as np
import cv2
import matplotlib.pyplot as plt

image_array = np.load('new_AAI_Dataset.npz')['trainSet']

plt.figure('1')
plt.imshow(image_array[5])
plt.figure('2')
plt.imshow(image_array[7])
plt.show()
