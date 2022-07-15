import torch
import torch.nn as nn
import os
from utils import prepare_data, train
from models import MLP2layers


def train_basic_MNIST(checkpoint_folder="./checkpoints/MNIST_basic/", model_name="1"):
    trainloader, testloader = prepare_data()
    model = MLP2layers()
    EPOCHS = 270
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    checkpoint_path = checkpoint_folder + model_name

    train(model, EPOCHS, trainloader, testloader, optimizer, criterion, checkpoint_path)


if __name__ == "__main__":
    train_basic_MNIST()