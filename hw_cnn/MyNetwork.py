import torch
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score

class MyNetwork(torch.nn.Module):

    def __init__(
        self, nInputChannels, nOutputClasses, learningRate = 0.1, 
        nEpochs=100, optim = "AdaDelta", logpath = "./log/MyNetwork/schedule",
        architecture = 'default'): # E-2 to E-4 # Epoch = 100 -> 50

        super().__init__()

        self.nInputChannels = nInputChannels
        self.nOutputClasses = nOutputClasses
        # Setup device-agnostic code 
        if torch.cuda.is_available():
            device = "cuda:0" # NVIDIA GPU
        elif torch.backends.mps.is_available():
            device = "mps" # Apple GPU
        else:
            device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available
        self.device = device

        self.optim = optim
        
        self.nEpochs = nEpochs
        self.learningRate = learningRate
        self.logPath = logpath
        self.architecture = architecture

        ## Image input is nSamples x 3 x 28 x 28


        ## Add your convolutional architecture
        # begin: [28 x 28 x 3]
        if self.architecture == 'default':
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.nInputChannels, kernel_size=3, out_channels=32, padding='valid'), #26x26
                # conv out: [26 x 26 x 32]
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                torch.nn.Dropout(0.2),
                torch.nn.Conv2d(in_channels=32, kernel_size=3,out_channels=64, padding='valid'),
                # conv out: [24 x 24 x 64]
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(64),
                torch.nn.MaxPool2d(kernel_size=3) 
                # max pool out: [8 x 8 x 64]
                # 8*8*64 = 4096 (i.e., the flattened vector length)
            )

            ## Add your fully connected architecture
            self.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.5), 
                torch.nn.Linear(4096, 100),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5), 
                torch.nn.Linear(100, self.nOutputClasses)
            )

        if self.architecture == "arch2":
            # begin [28 x 28 x 3]
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(self.nInputChannels, out_channels=32, kernel_size=(3,3)),
                # conv out: [26 x 26 x 32]
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                torch.nn.Dropout(0.2),
                torch.nn.Conv2d(32, out_channels=32, kernel_size=(3,3)),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                # conv out: [24 x 26 x 32]
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.Dropout(0.2),
            )
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(4608, 200),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(200),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(200, 100),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(100),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(100, self.nOutputClasses)
            )

    def forward(self, x):

        # Calculating convolutional features
        
        x = self.conv(x)
        #print(x.shape)

        # Flattening to feed to fully connected layers
        x = x.view(x.size(0), -1)

        # Making predictions
        x = self.fc(x)

        return x


    def trainModel(self, trainLoader, validationLoader):

        nTrainingSamples = len(trainLoader.dataset)
        nValidationSamples = len(validationLoader.dataset)

        # Moving to training device
        device = torch.device(self.device)
        self.to(device=device)
        # defining weights to account for data imbalance
        # 1 - (# no. of each class)/(total # of samples)
        # 0.3305
        class_weights = torch.tensor([0.9674, 0.9487, 0.8903, 0.9885, 0.8889, 0.3305, 0.9858], dtype=torch.float)
        class_weights = class_weights.to(torch.device(device))
        
        ## Define your loss and optimizer here
        if self.optim == "AdaDelta":
            optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learningRate)#,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        if self.optim == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)#,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        if self.optim == "AdaGrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learningRate)#,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        if self.optim == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learningRate)#,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        # define lr step sceufuler to decrease the learning rate after a certain number of epochs
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        
        # Creating logger
        writer = SummaryWriter(self.logPath)
        
        ## Iterating through epochs
        for epoch in range(self.nEpochs):
            epochLoss = 0.0
            epochAccuracy = 0.0
            epochLoss_val = 0.0
            epochAccuracy_val = 0.0

            ## TRAINING PHASE
            self.train(mode=True)
            for batch, (inputs, targets) in enumerate(trainLoader):
                inputs = inputs.to(torch.device(self.device))
                # set all gradients to zero before moving forward
                optimizer.zero_grad()
                # Making targets a vector with labels instead of a matrix
                targets = targets.to(torch.device(self.device))
                targets = targets.to(torch.long).view(targets.size(0))
                
                ## Forward operation here
                y_pred = self(inputs)

                ## compute loss
                batchLoss = loss(y_pred, targets)
                
                ## Bakproagation operations here
                batchLoss.backward()
                ## parameter updates here
                optimizer.step()

                ## Update average epoch loss and accuracy
                epochLoss += batchLoss.data*inputs.shape[0]/ nTrainingSamples
                
                labels_pred = torch.argmax(y_pred, dim=1)
                correct = (targets == labels_pred).float()
                accuracy = correct.sum()/ correct.numel()
                epochAccuracy += 100 * accuracy * inputs.shape[0]/ nTrainingSamples
                
            scheduler.step()  
        
            print(f'For training: Epoch {epoch+1}/{self.nEpochs} - Loss: {epochLoss:.4f} - Training Accuracy: {epochAccuracy:.2f}%')
            ## Log information (screen and/or Tensorboard)   
            writer.add_scalar('Loss/train', epochLoss, epoch)
            writer.add_scalar('Accuracy/train', epochAccuracy, epoch)
           
            
            ## VALIDATION PHASE
            self.train(False)
            for batch, (inputs_val, targets_val) in enumerate(validationLoader):

                inputs_val = inputs_val.to(torch.device(self.device))
                targets_val = targets_val.to(torch.device(self.device))
                # set all gradients to zero before moving forward
                optimizer.zero_grad()
                # Making targets a vector with labels instead of a matrix
                targets_val = targets_val.to(torch.long).view(targets_val.size(0))
                ## Forward operation here
                y_pred_val = self(inputs_val)
                # compute loss
                batchLoss_val = loss(y_pred_val, targets_val)
                epochLoss_val += batchLoss_val.data * inputs_val.shape[0]/ nValidationSamples
                ## Update average epoch loss and accuracy
                #labels_pred_val = torch.round(y_pred_val) # round probabilities to 0 or 1
                labels_pred_val = torch.argmax(y_pred_val, dim=1)
                correct_val = (targets_val == labels_pred_val).float()
                accuracy_val = correct_val.sum()/correct_val.numel()
                epochAccuracy_val += 100 * accuracy_val * inputs_val.shape[0] / nValidationSamples # average the batch loss
            
            print(f'For validation: Epoch {epoch+1}/{self.nEpochs} - Loss: {epochLoss_val:.4f} - Validation Accuracy: {epochAccuracy_val:.2f}%')
            ## Log information (screen and/or Tensorboard)
            writer.add_scalar('Loss/val', epochLoss_val, epoch)
            writer.add_scalar('Accuracy/val', epochAccuracy_val, epoch)
            
        writer.close()
    
    def predict(self, data):
        self.eval()
        self.to(torch.device(self.device))
        labels = data.labels.flatten().tolist()

        #predictions = {"truth": [], "pred_probs": {}}

        # performance metrics to calculate
        # sensitivity
        #TP/(TP+FN)
        #specificity
        #TN/(TN+FP)
        unique_classes = [0,1,2,3,4,5,6]
        dict_predictions = {
            "0": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "1": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "2": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "3": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "4": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "5": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "6": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            }
        correct = 0
        for image, label in data:
            ### prepare data
            # unlist the label
            label = label[0]
            # prepare image for passing forwarding through model
            image = torch.Tensor(image)
            image = image.to(torch.device(self.device))
            # *** add the batch dimension to fix "4D required input error"
            image = image[None]

            # predict
            y_pred_probs = torch.Tensor(self(image))
            y_pred_probs = y_pred_probs.to(torch.device(self.device))
            # softmax
            y_pred_softmax = torch.softmax(y_pred_probs, dim = 1).tolist()
            # use argmax to get prediction
            y_pred = np.argmax(y_pred_softmax)
            # get pred_prob of ground truth
            #truth_pred_prob = y_pred_softmax[y_pred]

            # count number of correct predictions
            if y_pred == label:
                correct += 1
            ### assign confusion matrix outcome
            for cls in unique_classes:
                # TP or FN
                if (cls == label):
                    # TP
                    if (y_pred == label):
                        dict_predictions[str(cls)]["TP"] += 1
                    # FN
                    else:
                        dict_predictions[str(cls)]["FN"] += 1
                # TN or FP
                if (cls != label):
                    # TN
                    if (y_pred != label):
                        dict_predictions[str(cls)]["TN"] += 1
                    # FP
                    if (y_pred == label):
                        dict_predictions[str(cls)]["FP"] += 1
                dict_predictions[str(cls)]
        accuracy = correct / len(labels)
        return dict_predictions, accuracy

        
    def save(self, path):

        ## Saving the model
        torch.save(self, path)