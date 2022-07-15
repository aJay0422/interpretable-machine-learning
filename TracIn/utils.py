import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


class mydataset(Dataset):
    def __init__(self, X, Y):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)

    def __getitem(self, item):
        return self.Data[item], self.Label[item]

    def __len__(self):
        return len(self.Label)


def prepare_data():
    trainset = torchvision.datasets.MNIST(
        root='./dataset/mnist/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    testset = torchvision.datasets.MNIST(
        root='./dataset/mnist/',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)

    return trainloader, testloader


def get_loss_acc(model, dataloader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if isinstance(criterion, nn.CrossEntropyLoss):   # classification problem
        correct = 0
        total = 0
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            model.eval()
            for X_batch, Y_batch in dataloader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                total += len(Y_batch)
                num_batches += 1
                outputs = model(X_batch)
                y_pred = torch.argmax(outputs, dim=1)
                correct += torch.sum(y_pred == Y_batch).cpu().numpy()
                loss = criterion(outputs, Y_batch)
                total_loss += loss.cpu().numpy()
            acc = correct / total
            total_loss = total_loss / num_batches

        return total_loss, acc


def train(model, epochs, trainloader, testloader, optimizer, criterion, checkpoint_path):
    """
    Train a basic model and save the checkpoint with the highest accuracy
    :param model: the model to be trained
    :param epochs: number of epochs
    :param trainloader: train data loader
    :param testloader: test data loader
    :param optimizer: optimizer for the training process
    :param criterion: criterion for computing the loss
    :param checkpoint_path: the path to save the checkpoint
    :return: the trained model(best test accuracy)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Trained on {}".format(device))

    # train model
    model.train()
    best_test_acc = 0
    for epoch in range(epochs):
        for X_batch, Y_batch in trainloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # forward
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate current model
        with torch.no_grad():
            model.eval()
            # evaluate train
            train_loss, train_acc = get_loss_acc(model, trainloader, criterion)
            # evaluate test
            test_loss, test_acc = get_loss_acc(model, testloader, criterion)

        print("Epoch: {}/{} train loss: {}  train acc: {}  test loss: {}  test acc: {}".format(
            epoch + 1, epochs, train_loss, train_acc, test_loss, test_acc
        ))

        # save model weights if it's the best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({"model_state_dict": model.state_dict()}, "{}.pt".format(checkpoint_path))
            print("Saved")

    best_checkpoint = torch.load("{}.pt".format(checkpoint_path))
    model.load_state_dict(best_checkpoint["model_state_dict"])

    return model
