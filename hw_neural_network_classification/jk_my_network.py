import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

# defining a network

class MyNetwork(torch.nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.batch_size = 200
        self.learning_rate = 0.01
        self.epochs = 250
        # fc = fully connected network
        self.fc = torch.nn.Sequential(
            # 20 linear neurons fully connected to the input features
            torch.nn.Linear(n_features, 20),
            # swap with ReLu
            torch.nn.ReLU()
            torch.nn.Linear(20, 1),
            torch.nn.Sigmoid(),
        )

    # the forward function from the super class 
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.fc(x)

    def train_model(self, train_data, train_labels, validation_data, validation_labels, logPath):

        n_train = train_data.shape[0]
        n_training_batches = n_train // self.batch_size
        if n_training_batches * self.batch_size < n_train:
            n_training_batches += 1

        n_validation = validation_data.shape[0]
        n_validation_batches = n_validation // self.batch_size
        if n_validation_batches * self.batch_size < n_validation:
            n_validation_batches += 1

        # configure device
        # first positional arg options ['cpu', 'cuda', or 'mps'] for mac
        device = torch.device('cuda')
        self.to(device=device)

        # config optimizer (stoch grad desc)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        # loss function
        loss = torch.nn.BCELoss()

        # create logger

        writer = SummaryWriter(logPath)

        # training
        for epoch in range(self.epochs):

            # keep information for back propagation during training
            self.train(mode=True)
            epoch_loss = 0.0
            epoch_acc = 0.0

            for batch in range(n_training_batches):
                optimizer.zero_grad()

                x = torch.tensor(train_data[batch * self.batch_size:(batch+1) * self.batch_size, :],
                    device = device,
                    dtype=torch.float32
                )
                # torch expect matrices, not vec
                y = torch.tensor(train_labels[batch * self.batch_size:(batch+1) * self.batch_size].reshape(-1,1),
                    device = device,
                    dtype=torch.float32
                )

                # forward pass
                y_pred = self.fc(x)

                # compute loss
                batch_loss = loss(y_pred, y)

                # backpropagation
                batch_loss.backward()

                # update params
                optimizer.step()

                # accumulate loss
                # average the batch loss for the epoch
                epoch_loss += batch_loss.data * x.shape[0] / n_train

                # compute acc
                # convert pred probs to labels {0,1}
                labels_pred = torch.round(y_pred)
                correct = (y == labels_pred).float()
                accuracy = correct.sum() / correct.shape.numel()
                epoch_accuracy = 100 * accuracy * x.shape[0] / n_train

                # validation
                
                for batch in range(n_validation_batches):
                    self.train(mode=False)
                    #forward pass
                    # compute validation loss and acc
                
                print(f'Epoch {epoch +1/self.epochs}  - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}')

                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)

                writer.close()
    def save(self, path):
        torch.save(self, path)
