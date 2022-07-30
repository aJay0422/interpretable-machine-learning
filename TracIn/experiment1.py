import torch
from torch.utils.data import DataLoader
import numpy as np
import os

from model import CNN_CIFAR10
from data import prepare_CIFAR10
from train import train_CNN_CIFAR10
from influence_functions import select_validation, select_train
from utils import mydataset


if __name__ == "__main__":
    experiment_path = "./experiment1"
    settings = [("V1", "T1"), ("V2", "T2"), ("V1", "T5"), ("V2", "T6")]
    load_path = "model_weights/CNN_CIFAR10_epoch3.pth"   # the model has been trained 3 epochs

    for (methodV, methodT) in settings:
        setting_path = experiment_path + "/" + methodV + methodT
        if not os.path.exists(setting_path):
            os.mkdir(setting_path)

        # prepare model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = CNN_CIFAR10()
        net.to(device)
        net.eval()

        # prepare data
        trainloader, valloader, testloader = prepare_CIFAR10()

        # select validation
        val_path = setting_path + "/selected_val_" + methodV + methodT + ".npz"
        if os.path.exists(val_path):
            file = np.load(val_path, allow_pickle=True)
            selected_val = {"X":file["X"], "Y":file["Y"]}
        else:
            selected_val = select_validation(net, valloader, method=methodV)
            np.savez(val_path, X=selected_val["X"], Y=selected_val["Y"])

        train_path = setting_path + "/selected_train_" + methodV + methodT + ".npz"
        if os.path.exists(train_path):
            file = np.load(train_path, allow_pickle=True)
            selected_train = {"X":file["X"], "Y":file["Y"]}
        else:
            selected_train = select_train(net, trainloader, selected_val, method=methodT)
            np.savez(train_path, X=selected_train["X"], Y=selected_train["Y"])

        # create a new trainloader
        selected_trainset = mydataset(selected_train["X"], selected_train["Y"])
        selected_trainloader = DataLoader(selected_trainset, batch_size=256, shuffle=True)

        # load trained model
        net.load_state_dict(torch.load("model_weights/CNN_CIFAR10_epoch3.pth", map_location=device))

        # train model and save checkpoint
        n_train = len(selected_trainset.Label)
        print("Trained on a dataset of {} samples".format(n_train))
        save_path = setting_path + "/" + "CNN_CIFAR10_" + methodV + methodT + ".pth"
        train_CNN_CIFAR10(epochs=50, trainloader=selected_trainloader, testloader=valloader, load_path=load_path, save_path=save_path)
