import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

class MyNetwork(torch.nn.Module):

    def __init__(self, nInputChannels, nOutputClasses, learningRate=1E-2, nEpochs=100):
        super().__init__()

        self.nInputChannels = nInputChannels
        self.nOutputClasses = nOutputClasses
        self.trainingDevice = 'cpu'
        
        self.nEpochs = nEpochs
        self.learningRate = learningRate

        ## Image input is nSamples x 3 x 28 x 28
        ## Add your convolutional architecture
        # tune the padding and stride
        self.conv = torch.nn.Conv2d(
            in_channels =self.inputChannels,
            out_channels = 32, # this is the number of features for each kernel
            kernel_size=3,
            padding='valid'
        ), #26x26 (-2 due to the padding)
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride = 1), #13x13
        torch.nn.Conv2d(64, 128, 3, padding='valid'), #4x4
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2) # 2x2

        ## Add your fully connected architecture
        self.fc = torch.nn.Sequential(
            #dropout
            torch.nn.Linear(128,64),
            torch.nn.Linear(64, self.nOutputClasses)
            # possibly use softmax 
            #torch.nn.SoftMax()
        )

    def forward(self, x):

        # Calculating convolutional features
        x = self.conv(x)

        # Flattening to feed to fully connected layers
        # important for coercing the matrix into a vector
        x = x.view(x.size(0), -1)

        # Making predictions
        x = self.fc(x)

        return x


    def trainModel(self, trainLoader, validationLoader):

        nTrainingSamples = len(trainLoader.dataset)

        # Moving to training device
        device = torch.device(self.trainingDevice)
        self.to(device=device)

        ## Define your loss and optimizer here
        loss = torch.nn.CrossEntropyLoss()

        # Logger
        writer = SummaryWriter(logPath)


        ## Iterating through epochs
        for epoch in range(self.nEpochs):

            ## TRAINING PHASE
            self.train()
            for batch, (inputs, targets) in enumerate(trainLoader):

                # Making targets a vector with labels instead of a matrix
                targets = targets.to(torch.long).view(targets.size(0))

                ## Forward operation here
                # this is passed into softmax
                y_pred = self(inputs)
                
                ## Bakproagation operations here

                ## Parameter updates here
                
                ## Update average epoch loss and accuracy

                ## Log information (screen and/or Tensorboard)

            ## VALIDATION PHASE
            self.train(False)

    def save(self, path):

        ## Saving the model
        torch.save(self, path)