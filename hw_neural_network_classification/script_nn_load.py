#!/usr/bin/env python
import pandas as pd
import torch
from MyNetwork import MyNetwork
import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import os,sys

# Load data
trainingData = pd.read_excel('GeneExpressionCancer_training.xlsx').to_numpy()
trainingLabels = trainingData[:, -1]
trainingData = trainingData[:, :-1]

validationData = pd.read_excel('GeneExpressionCancer_validation.xlsx').to_numpy()
validationLabels = validationData[:, -1]
validationData = validationData[:, :-1]

#testData = pd.read_excel('GeneExpressionCancer_test.xlsx').to_numpy()
#testLabels = testData[:, -1]
#testData = testData[:, :-1]

# Normalize data
scaler = sklearn.preprocessing.StandardScaler()
trainingData = scaler.fit_transform(trainingData)
validationData = scaler.transform(validationData)
#testData = scaler.transform(testData)

model = torch.load('./model/FinalModel')
y_pred_prob = model.predict(validationData)
for truth, p in zip(validationLabels, y_pred_prob):
    print(f'truth: {truth}; p_1: {p}')

fpr, tpr , thres = roc_curve(truth, y_pred_prob)
tn, fp, fn, tp = confusion_matrix(validationLabels, y_pred).ravel()
#specificity = tn / (tn+fp)
#sensitivity = tp / (fn+tp)
#accuracy = (tn + tp) / (tn+tp+fn+fp)
#print(f'accuracy: {accuracy}; sensitivity {sensitivity}; specificity {specificity}')



