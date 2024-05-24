#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os,sys

import torchvision.transforms as transforms
import torch.utils.data as data

import medmnist
from medmnist import INFO

from MyNetworkTemplate import MyNetwork

# 3 (RGB)
nChannels = INFO['dermamnist']['n_channels']
# 7
nClasses = len(INFO['dermamnist']['label'])
DataClass = medmnist.DermaMNIST

# Transforming images to Torch Tensor and Normalizing
data_transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0], std=[1])
])
## Obtaining the training, validation and test datasets
trainingData = DataClass(split='train', transform=data_transform, download=True)
validationData = DataClass(split='val', transform=data_transform, download=True)
testData = DataClass(split='test', transform=data_transform, download=True)

print(trainingData, validationData, testData)
## This code will show a preview of the images
#a = trainingData.montage(length=5)
#plt.imshow(a)
#plt.show()
## Configuring the batch size and creating data loaders
batchSize = 100
trainLoader = data.DataLoader(dataset=trainingData, batch_size=batchSize,
shuffle=True)
validationLoader = data.DataLoader(dataset=validationData, batch_size=batchSize,
shuffle=True)
testLoader = data.DataLoader(dataset=testData, batch_size=batchSize, shuffle=False)
model = MyNetwork(nChannels, nClasses)
model.trainModel(trainLoader, validationLoader)
## Add evaluation code here
