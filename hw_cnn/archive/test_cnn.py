#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from MyNetworkTemplate import MyNetwork
import medmnist
from medmnist import DermaMNIST
from medmnist import INFO

os.chdir("/home/jake/ghub/bios_7747/hw_cnn")

# load data
nChannels = INFO['dermamnist']['n_channels']
nClasses = len(INFO['dermamnist']['label'])
DataClass = DermaMNIST

# Transforming images to Torch Tensor and Normalizing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[1])
])

# Define the directory for the dataset
dataset_directory = 'derma_data'

# Use an absolute path for the dataset
dataset_path = os.path.abspath(dataset_directory)

## Obtaining the training, validation and test datasets
trainingData = DataClass(root=dataset_path,split='train', transform=data_transform)
validationData = DataClass(root=dataset_path, split='val', transform=data_transform)
testData = DataClass(root=dataset_path, split='test', transform=data_transform)

## This code will show a preview of the images
# a = trainingData.montage(length=5)
# plt.imshow(a)
# plt.show()

## Configuring the batch size and creating data loaders
batchSize = 100
trainLoader = data.DataLoader(dataset=trainingData, batch_size=batchSize, shuffle=True)
validationLoader = data.DataLoader(dataset=validationData, batch_size=batchSize, shuffle=True)
testLoader = data.DataLoader(dataset=testData, batch_size=batchSize, shuffle=False)

model = torch.load('cnn_model')
validationPredictions = model.predict(testLoader)