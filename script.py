import pandas as pd
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

model = MyNetwork(trainingData.shape[1])
model.trainModel(trainingData, trainingLabels, validationData, validationLabels, 'log/MyNetwork')
model.save('model/FinalModel')

# validationPredictions = model(validationData)
