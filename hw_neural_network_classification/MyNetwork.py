import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

# defining a network begins with subclassing a the Module class
class MyNetwork(torch.nn.Module):

    def __init__(self, nFeatures, n_layers = 2):
        super().__init__()

        self.nFeatures = nFeatures
        self.batchSize = 500
        self.learningRate = 0.01
        self.epochs = 250

        if (n_layers == 2):
            # fully connected network structure
            self.fc = torch.nn.Sequential(
                ### layer 1
                # net input
                torch.nn.Linear(self.nFeatures, 20),
                # activation
                torch.nn.ReLU(),
                ### layer 2
                # net input
                torch.nn.Linear(20, 1),
                # prediction
                torch.nn.Sigmoid(),
            )
        elif (n_layers == 1):
            # fully connected network structure
            self.fc = torch.nn.Sequential(
                ### layer 1
                # net input
                torch.nn.Linear(self.nFeatures, 1),
                # activation
                # Q: is activation correct/appropriate here or can we just feed directly into sigmoid?
                #torch.nn.ReLU(),
                # prediction
                torch.nn.Sigmoid()
            )
        else: 
            print('n_layers must be 1 or 2')
            raise ValueError


    # the forward function from the super class 
    # every nn.Module subclass implements the operations on input data in the forward method
    def forward(self, x):

        x = torch.tensor(x, dtype=torch.float32)

        # return fully connected network structure 
        return self.fc(x)
    
    def trainModel(self, trainData, trainLabels, validationData, validationLabels, logPath):

        # Configuring batches
        nTrainingSamples = trainData.shape[0]
        nTrainingBatches = nTrainingSamples // self.batchSize
        if nTrainingBatches * self.batchSize < nTrainingSamples:
            nTrainingBatches += 1

        nValidationSamples = validationData.shape[0]
        nValidationBatches = nValidationSamples // self.batchSize
        if nValidationBatches * self.batchSize < nValidationSamples:
            nValidationBatches += 1

        # Configuring device
        # first positional arg options ['cpu', 'cuda', or 'mps'] for mac
        device = torch.device('cpu') # cuda (NVIDIA GPU) or mps (Mac M1+)
        self.to(device=device)

        # Configuring optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learningRate)
        # loss function
        loss = torch.nn.BCELoss()

        # Creating logger
        writer = SummaryWriter(logPath)

        # Training
        for epoch in range(self.epochs):

            # keep information for back propagation during training
            self.train(mode=True)
            epochLoss = 0.0
            epochAccuracy = 0.0

            for batch in range(nTrainingBatches):

                optimizer.zero_grad()

                x = torch.tensor(trainData[batch*self.batchSize:(batch+1)*self.batchSize, :]   , 
                    device=device,
                    dtype=torch.float32
                )

                # torch expect matrices, not vec
                y = torch.tensor(trainLabels[batch*self.batchSize:(batch+1)*self.batchSize].reshape((-1,1)), 
                    device=device,
                    dtype=torch.float32
                )

                # Forward pass
                y_pred = self.fc(x)
                
                # Compute loss
                batchLoss = loss(y_pred, y)

                # Backpropagation
                batchLoss.backward()

                # Update parameters
                optimizer.step()

                # Accumulate loss
                # average the batch loss for the epoch
                epochLoss += batchLoss.data * x.shape[0] / nTrainingSamples

                # Compute accuracy
                labels_pred = torch.round(y_pred)
                correct = (y == labels_pred).float()
                accuracy = correct.sum() / correct.numel()
                epochAccuracy += 100 * accuracy * x.shape[0] / nTrainingSamples

            # Validation
            # for bach in range(nValidationBatches):
            # self.train(mode=False)
            # Forward pass
            # Compute validation loss and accuracy

            print(f'Epoch {epoch+1}/{self.epochs} - Loss: {epochLoss:.4f} - Accuracy: {epochAccuracy:.2f}%')

            writer.add_scalar('Loss/train', epochLoss, epoch)
            writer.add_scalar('Accuracy/train', epochAccuracy, epoch)

        writer.close()

    def predict(self, testData, threshold = -1):
        # set to eval mode
        self.eval()

        # Configuring device
        # first positional arg options ['cpu', 'cuda', or 'mps'] for mac
        device = torch.device('cpu') # cuda (NVIDIA GPU) or mps (Mac M1+)
        X_tf = torch.Tensor(testData).to(device)
        y_pred_prob = self(X_tf)
        y_pred_prob = y_pred_prob.detach().numpy().flatten()
        if (0 < threshold < 1):
            print('*** warning: threshold will coerce output to classes')
            y_pred = y_pred_prob.copy()
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            return(y_pred)
        else: 
            return(y_pred_prob)

    def save(self, path):
        torch.save(self, path)

