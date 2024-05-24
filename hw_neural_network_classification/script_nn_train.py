#!/usr/bin/env python
import pandas as pd
import os,sys
import numpy as np
import sklearn
import sklearn.preprocessing
from MyNetwork import MyNetwork

# Load data
trainingData = pd.read_excel('GeneExpressionCancer_training.xlsx').to_numpy()
trainingLabels = trainingData[:, -1]
trainingData = trainingData[:, :-1]

validationData = pd.read_excel('GeneExpressionCancer_validation.xlsx').to_numpy()
validationLabels = validationData[:, -1]
validationData = validationData[:, :-1]

testData = pd.read_excel('GeneExpressionCancer_test.xlsx').to_numpy()
testLabels = testData[:, -1]
testData = testData[:, :-1]

# Normalize data
scaler = sklearn.preprocessing.StandardScaler()
trainingData = scaler.fit_transform(trainingData)
validationData = scaler.transform(validationData)
testData = scaler.transform(testData)

# remove previous results
if (os.path.isdir('log_1_layer')):
    os.system('rm -r log_1_layer')
if (os.path.isdir('log_2_layer')):
    os.system('rm -r log_2_layer')
os.system(f"mkdir log_1_layer")
os.system(f"mkdir log_2_layer")

### 2 layers
# call forward by passing in data; returns tensor obj
model = MyNetwork(trainingData.shape[1], n_layers=2)
model.trainModel(trainingData, trainingLabels, validationData, validationLabels, 'log_2_layer')
model.save('model/FinalModel_2_layer')

### 1 layers
model = MyNetwork(trainingData.shape[1], n_layers=1)
model.trainModel(trainingData, trainingLabels, validationData, validationLabels, 'log_1_layer')
model.save('model/FinalModel_1_layer')

# model.eval() will set the mode to evaluation mode
# validationPredictions = model(validationData)

