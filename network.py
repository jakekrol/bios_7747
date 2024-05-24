import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

class MyNetwork(torch.nn.Module):

    def __init__(self, nFeatures):
        super().__init__()

        self.nFeatures = nFeatures
        self.batchSize = 500
        self.learningRate = 0.01
        self.epochs = 250

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.nFeatures, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):

        x = torch.tensor(x, dtype=torch.float32)

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
        device = torch.device('cpu') # cuda (NVIDIA GPU) or mps (Mac M1+)
        self.to(device=device)

        # Configuring optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learningRate)
        loss = torch.nn.BCELoss()

        # Creating logger
        writer = SummaryWriter(logPath)

        # Training
        for epoch in range(self.epochs):

            self.train(mode=True)
            epochLoss = 0.0
            epochAccuracy = 0.0

            for batch in range(nTrainingBatches):

                optimizer.zero_grad()

                x = torch.tensor(trainData[batch*self.batchSize:(batch+1)*self.batchSize, :]   , 
                    device=device,
                    dtype=torch.float32
                )

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
            # Coputer validation loss and accuracy

            print(f'Epoch {epoch+1}/{self.epochs} - Loss: {epochLoss:.4f} - Accuracy: {epochAccuracy:.2f}%')

            writer.add_scalar('Loss/train', epochLoss, epoch)
            writer.add_scalar('Accuracy/train', epochAccuracy, epoch)

        writer.close()

    def save(self, path):
        torch.save(self, path)


