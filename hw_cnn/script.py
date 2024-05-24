import os,sys
from torchviz import make_dot
import hiddenlayer as hl
os.chdir("/home/jake/ghub/bios_7747/hw_cnn")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MyNetwork import MyNetwork
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import medmnist
from medmnist import INFO

# Setup device-agnostic code 
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

print(f"Using device: {device}")

# Load data
nChannels = INFO['dermamnist']['n_channels']
nClasses = len(INFO['dermamnist']['label'])
DataClass = medmnist.DermaMNIST

# Transforming images to Torch Tensor and Normalizing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[1])
])

## Obtaining the training, validation and test datasets
trainingData = DataClass(split="train", transform=data_transform, download=True)
validationData = DataClass(split="val", transform=data_transform, download=True)
testData = DataClass(split="test", transform=data_transform, download=True)

## This code will show a preview of the images
#a = trainingData.montage(length=5)
#plt.imshow(a)
#plt.show()

## Configuring the batch size and creating data loaders
batchSize = 100
trainLoader = data.DataLoader(dataset=trainingData, batch_size=batchSize, shuffle=True)
validationLoader = data.DataLoader(dataset=validationData, batch_size=batchSize, shuffle=True)
testLoader = data.DataLoader(dataset=testData, batch_size=batchSize, shuffle=False)

print(f"Number of channels: {nChannels}")
print(f"Number of classes: {nClasses}")

##### Train model

#optims = ["AdaDelta", "Adam", "AdaGrad", "RMSprop"]
#for optim in optims:
#    model = MyNetwork(nChannels, nClasses, optim = optim, logpath=f"./log/MyNetwork/{optim}")
#    model.trainModel(trainLoader, validationLoader)
#    model.save(f'./model/schedule/{optim}')

#model = MyNetwork(nChannels, nClasses, optim = 'AdaDelta', learningRate=1,  logpath=f'./log/MyNetwork/springer', nEpochs=200, architecture = 'springer')
#model.trainModel(trainLoader, validationLoader)
#model.save(f'./model/schedule/arch2')

#model = MyNetwork(nChannels, nClasses, optim='AdaDelta', logpath='./log/MyNetwork/finalModel', nEpochs=40)
#model.trainModel(trainLoader, validationLoader)
#model.save(f'./model/schedule/AdaDeltaEarly')


##### Viz model architecture
#model = torch.load('model/schedule/AdaDeltaEarly')
#batch = next(iter(trainLoader))
#batch = batch[0].to(torch.device("cuda:0"))
#yhat = model(batch)
#make_dot(yhat, params=dict(list(model.named_parameters()))).render("arch1_torchviz", format='png')
#
##arch2
#model = torch.load('model/schedule/arch2')
#batch = next(iter(trainLoader))
#batch = batch[0].to(torch.device("cuda:0"))
#yhat = model.predict(batch)
#make_dot(yhat, params=dict(list(model.named_parameters()))).render("arch2_torchviz", format='png')
#hiddenlayer
#hl_transforms = [ hl.transforms.Prune('Constant') ]
#
#graph = hl.build_graph(model, batch, transforms=hl_transforms)
#graph.theme = hl.graph.THEMES['blue'].copy()
#graph.save('arch1_hl', format='png')


# use model to predict
model = torch.load('model/schedule/AdaDeltaEarly')
valPredictions, valAcc = model.predict(validationData)
testPredictions, testAcc = model.predict(testData)

def calc_sens_spec(predictions):
    # performance metrics to calculate
    # sensitivity
    #TP/(TP+FN)
    #specificity
    #TN/(TN+FP)
    result = {
        "0": {"spec": -1, "sens": -1},
        "1": {"spec": -1, "sens": -1},
        "2": {"spec": -1, "sens": -1},
        "3": {"spec": -1, "sens": -1},
        "4": {"spec": -1, "sens": -1},
        "5": {"spec": -1, "sens": -1},
        "6": {"spec": -1, "sens": -1},
    }
    for cls in predictions.keys():
        # sensitivity
        tp = predictions[str(cls)]["TP"]
        fn = predictions[str(cls)]["FN"]
        sens = tp / (tp+fn)
        # specificity
        tn = predictions[str(cls)]["TN"]
        fp = predictions[str(cls)]["FP"]
        spec = tn / (tn+fp)
        result[str(cls)]["spec"] = spec
        result[str(cls)]["sens"] = sens
    return(result)

sens_spec = calc_sens_spec(testPredictions)

sens_spec

sensitivities = []
specificities = []
for key,value in sens_spec.items():
    sensitivities.append(value["sens"])
    specificities.append(value["spec"])

def plot_bar(sens, spec, title, acc,  filepath):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    width = 0.5
    ax[0].bar(list(range(7)), sens, alpha = 0.5, width=width, color='blue')
    ax[0].set_title('Sensitivity')
    ax[1].bar(list(range(7)), spec, alpha = 0.5, width=width, color='orange')
    ax[1].set_title('Specificity')
    fig.suptitle(f'{title} Accuracy: {round(acc, 3)}', fontsize=20)
    fig.legend(loc='upper center')
    fig.savefig(filepath)
plot_bar(sens=sensitivities, spec=specificities, title = 'AdaDeltaEarly', acc = testAcc, filepath='figs/AdaDeltaEarly/testPerformance.png')
