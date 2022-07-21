import torch

from influence_functions import select_train, select_validation
from data import prepare_CIFAR10
from model import CNN_CIFAR10


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10()
    selected_val = select_validation(net, valloader)

    net.to(device)
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10_epoch3.pth", map_location=device))

    selected_train = select_train(net, trainloader, selected_val)




    stop = None