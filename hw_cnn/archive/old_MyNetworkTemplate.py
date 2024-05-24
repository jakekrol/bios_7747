import torch
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
        #############################################  GOT SOME OF THIS CODE IN CLASS        #############################################
        ## Add your convolutional architecture
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.inputChannels,in_channels=3, out_channels=32, padding='valid'), #26x26
            #add a batch normalization here done before activation
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), #13x13
            torch.nn.Conv2d(in_channels=32, out_channels=64, padding='valid'), #11x11
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),#6x6
            torch.nn.Conv2d(in_channels=64, out_channels=128, padding='valid'),
        )
        #for overfitting you can use drop out and use it before the last linear layer

        #############################################  GOT SOME OF THIS CODE IN CLASS        #############################################

        ## Add your fully connected architecture
        self.fc = torch.nn.Sequential(
            #or you can add a droptout here : 0.2
            torch.nn.Linear(128, 60),  
            torch.nn.ReLU(),
            #dropout :0.8
            torch.nn.Linear(64, self.nOutputClasses)  # Changed from (64, 10) to (50, 10)
        )

    def forward(self, x):
        # Calculating convolutional features
        x = self.conv(x)

        # Flattening to feed to fully connected layers
        x = x.view(x.size(0), -1)

        # Making predictions
        x = self.fc(x)

        return x


        #############################################  GOT SOME OF THIS CODE IN CLASS        #############################################
    def trainModel(self, trainLoader, validationLoader, logPath):
        nTrainingSamples = len(trainLoader.dataset)

        # Moving to training device
        device = torch.device('cpu')
        self.to(device=device)
        
        #can add the weights here in the loss arguement
        loss = torch.nn.CrossEntropyLoss()  # Changed from BCELoss to CrossEntropyLoss

        ## Define your loss and optimizer here
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)
        
        

        # Creating logger
        writer = SummaryWriter(logPath)

        ## Iterating through epochs
        for epoch in range(self.nEpochs):
            
            ## TRAINING PHASE
            self.train() 
            for batch, (inputs, targets) in enumerate(trainLoader):
                # Moving data to the device
                targets = targets.to(torch.long).view(targets.size(0))
                optimizer.step()
                ## Forward operation here
                y_pred = self(inputs)
                y_pred = torch.nn.functional.softmax(y_pred, dim=1)
                epochLoss += batchloss.data*inputs.shape[0]/ nTrainingSamples
                
                labels_pred = torch.argmax(y_pred, dim=1)
                correct(targets == labels_pred).float()
                accuracy = correct.sum()/ correct.numel()
                
                epochAccuracy += accuracy * inputs.shape[0]/ nTrainingSamples
                ## Bakproagation operations here
                optimizer.zero_grad()
                batchLoss= loss(y_pred, targets)
                optimizer.step()

                ## Update average epoch loss and accuracy
                # You may need to calculate accuracy based on your task

                ## Log information (screen and/or Tensorboard)
                if batch % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch}, Loss: {batchLoss.item()}")

        #############################################  GOT SOME OF THIS CODE IN CLASS        #############################################

            ## VALIDATION PHASE
            self.eval()
            with torch.no_grad():
                for inputs, targets in validationLoader:
                    # Moving data to the device
                    inputs, targets = inputs.to(device), targets.to(device)

                    ## Forward operation for validation
                    outputs = self(inputs)
                    validationLoss = loss(outputs, targets)

                    # You can log validation loss and accuracy here if needed

        # Save the model after training
        self.save("trained_model.pth")

    def save(self, path):
        ## Saving the model
        torch.save(self.state_dict(), path)

#can use different loss and optimizer functions