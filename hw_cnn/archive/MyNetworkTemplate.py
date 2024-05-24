import torch
import tqdm

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
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, kernel_size=3, out_channels=32, padding='valid'), #26x26
            #add a batch normalization here done before activation
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), #13x13
            torch.nn.Conv2d(in_channels=32, kernel_size=3,out_channels=64, padding='valid'), #11x11
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),#6x6
            torch.nn.Conv2d(in_channels=64, kernel_size=3, out_channels=128, padding='valid'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        ## Add your fully connected architecture
        self.fc = torch.nn.Sequential(
             #or you can add a droptout here : 0.2
            torch.nn.Linear(128, 64),  
            torch.nn.ReLU(),
            #dropout :0.8
            torch.nn.Linear(64, self.nOutputClasses)  # Changed from (64, 10) to (50, 10)
        
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

        # Moving to training device
        device = torch.device(self.trainingDevice)
        self.to(device=device)
        
        # Creating logger
        #writer = SummaryWriter(logPath)

        ## Define your loss and optimizer here
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)
        
        ## Iterating through epochs
        for epoch in range(self.nEpochs):
            self.train(mode=True)
            epochLoss = 0.0
            epochAccuracy = 0.0

            ## TRAINING PHASE
            self.train()
            for batch, (inputs, targets) in enumerate(trainLoader):
                optimizer.zero_grad()
                # Making targets a vector with labels instead of a matrix
                targets = targets.to(torch.long).view(targets.size(0))
                
                ## Forward operation here
                y_pred = self(inputs)

                ## compute loss
                batchLoss= loss(y_pred, targets)
                
                ## Bakproagation operations here
                batchLoss.backward()
                optimizer.step()

                
                ## Update average epoch loss and accuracy
                epochLoss += batchLoss.data*inputs.shape[0]/ nTrainingSamples
                
                labels_pred = torch.argmax(y_pred, dim=1)
                correct = (targets == labels_pred).float()
                accuracy = correct.sum()/ correct.numel()

                epochAccuracy += accuracy * inputs.shape[0]/ nTrainingSamples
                ## Log information (screen and/or Tensorboard)
        
        print(f'For training: Epoch {epoch+1}/{self.nEpochs} - Loss: {epochLoss:.4f} - Training Accuracy: {epochAccuracy:.2f}%')
            ## VALIDATION PHASE
        #self.train(False)

    def predict(self, testData, threshold = -1):
        '''
        Predict classses
        '''
        with torch.no_grad():
        # set to eval mode
            self.eval()

            # Configuring device
            # first positional arg options ['cpu', 'cuda', or 'mps'] for mac
            device = torch.device('cpu') # cuda (NVIDIA GPU) or mps (Mac M1+)
            X = testData.to(device)
            y_pred_prob = self(X)
            y_pred_prob = y_pred_prob.detach().numpy().flatten()
            return(y_pred_prob)
            if (0 < threshold < 1):
                print('*** warning: threshold will coerce output to classes')
                y_pred = y_pred_prob.copy()
                y_pred[y_pred >= 0.5] = 1
                y_pred[y_pred < 0.5] = 0
                return(y_pred)
            else: 
                return(y_pred_prob)
        
    def save(self, path):

        ## Saving the model
        torch.save(self, path)