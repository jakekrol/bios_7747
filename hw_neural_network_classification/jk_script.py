import pandas as pd
import numpy as np
import sklearn
import sklearn.preprocessing
from my_network import MyNetwork


# Load data

training_data = pd.read_excel("./GeneExpressionCancer_training.xlsx").to_numpy()
training_labels = training_data[:, -1]
training_data = training_data[:, :-1]

# repeat for validation and test
validation_data = pd.read_excel("./GeneExpressionCancer_validation.xlsx").to_numpy()
validation_labels = validation_data[:, -1]
validation_data = validation_data[:, :-1]

test_data = pd.read_excel("./GeneExpressionCancer_test.xlsx").to_numpy()
test_labels = test_data[:, -1]
test_data = test_data[:, :-1]

# normalize data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(training_data)
training_data = scaler.transform(training_data)
test_data = scaler.transform(test_data)
validation_data = scaler.transform(validation_data)

# train nn
model = MyNetwork(training_data.shape[1])
model.train_model(training_data, training_labels, validation_data, validation_labels, 'log/MyNetwork')
model.save('model/final_model')

#  validation
# returns tensor
#validation_predictions = model(validation_data)