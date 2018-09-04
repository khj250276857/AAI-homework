import cv2
import os
import numpy as np
import natsort
import glob
from PIL import Image


def makeDataSet():
    fr = open(r"E:\研一下学期学习资料\高级人工智能\datasets/train.txt", "r")
    contents = fr.readlines()
    count = []
    finalCount = []
    trainCount = []
    fileName = []
    for i in range(len(contents)):
        count.append(int(contents[i].split(' ')[1].split('\n')[0]))
        fileName.append(os.path.join(r'E:\研一下学期学习资料\高级人工智能\datasets\train', contents[i].split(' ')[0]))
    for i in range(100):
        finalCount.append(count.count(i+1))
    for i in range(100):
        if finalCount[i] < 35:
            trainCount.append(finalCount[i] - 5)
        else:
            trainCount.append(30)
    # temp = 0
    # for i in range(100):
    #     temp = trainCount[i] + temp
    # print(temp)
    # DataAugmentationDIR = '/home/huang/Desktop/test'
    # dirs = natsort.natsorted(glob.glob(os.path.join(DataAugmentationDIR, '*')))
    # print(len(dirs))
    totalCount = 0
    trainSet = []
    testSet = []
    trainLabel = []
    testLabel = []
    for j in range(100):
        print(str(j+1) + ":" + str(finalCount[j]))
        label = np.zeros((100, 1))
        for i in range(finalCount[j]):
            if finalCount[j] - i > 5 and i < 30:
                print(fileName[i + totalCount])
                image = np.array(Image.open(fileName[i + totalCount]))
                image = cv2.resize(image, (224, 224), cv2.INTER_CUBIC)
                trainSet.append(image)
                label[j] = 1
                trainLabel.append(label)
            if finalCount[j] - i <= 5:
                image = np.array(Image.open(fileName[i + totalCount]))
                image = cv2.resize(image, (224, 224), cv2.INTER_CUBIC)
                testSet.append(image)
                label[j] = 1
                testLabel.append(label)
        totalCount = totalCount + finalCount[j]
    # temp = trainCount[0]
    # labelTag = 0
    # for i in range(len(dirs)):
    #     image = cv2.imread(dirs[i])
    #     trainSet.append(image)
    #     label = np.zeros((100, 1))
    #     if not int(dirs[i].split('_')[1]) < temp:
    #         labelTag = labelTag + 1
    #         temp = trainCount[labelTag] + temp
    #     label[labelTag] = 1
    #     trainLabel.append(label)
    #     print(dirs[i])
    #     print(labelTag)
    np.savez('AAI_Dataset.npz', trainSet=trainSet, trainLabel=trainLabel, testSet=testSet, testLabel=testLabel)


makeDataSet()