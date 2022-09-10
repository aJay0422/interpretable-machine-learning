import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import mydataset

def dataset_category_get(category_num):
    """
    Get images with label == category_num for both train and test dataset
    :param category_num: which category of image want to get
    :return: img_all_train: shape(500, 3, 224, 224), 500 images from train dataset
             img_all_test: shape(100, 3, 224, 224), 100 images from test dataset
    """
    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # load CIFAR10 train and test dataset
    train_set = torchvision.datasets.CIFAR10(root="datasets/CIFAR10/", train=True,
                                             download=True, transform=data_transform["train"])
    test_set = torchvision.datasets.CIFAR10(root="datasets/CIFAR10/", train=False,
                                            download=True, transform=data_transform["val"])
    trainloader = DataLoader(train_set, batch_size=10000,
                             shuffle=False, num_workers=0)
    testloader = DataLoader(test_set, batch_size=2000,
                            shuffle=False, num_workers=0)
    train_iter = iter(trainloader)
    train_image, train_label = next(train_iter)
    test_iter = iter(testloader)
    test_image, test_label = next(test_iter)

    img_all_train = torch.zeros(500, 3, 224, 224)
    train_image_num = 0
    for i in range(10000):
        if train_label[i] == category_num:
            img_all_train[train_image_num] = train_image[i]
            train_image_num += 1
        if train_image_num == 500:
            break

    img_all_test = torch.zeros(100, 3, 224, 224)
    test_image_num = 0
    for i in range(2000):
        if test_label[i] == category_num:
            img_all_test[test_image_num] = test_image[i]
            test_image_num += 1
        if test_image_num == 100:
            break

    return img_all_train, img_all_test


def prepare_CIFAR10(img_size=32, mode="tvt", train_shuffle=True):
    if img_size == 32:
        data_transform = {
            "train": transforms.Compose([transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }
    elif img_size == 224:
        data_transform = {
            "train": transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }

    # load CIFAR10 train and test dataset
    train_set = torchvision.datasets.CIFAR10(root="datasets/CIFAR10/", train=True,
                                             download=False, transform=data_transform["train"])
    test_set = torchvision.datasets.CIFAR10(root="datasets/CIFAR10/", train=False,
                                            download=False, transform=data_transform["val"])


    if mode == "tt":
        # train_set = mydataset(X_train, Y_train)
        # test_set = mydataset(X_test, Y_test)
        trainloader = DataLoader(train_set, batch_size=128,
                                 shuffle=train_shuffle)
        testloader = DataLoader(test_set, batch_size=64,
                                shuffle=False)
        print("{} train samples, {} test samples".format(len(train_set.targets), len(test_set.targets)))
        return trainloader, testloader
    elif mode == "tvt":
        torch.manual_seed(42)
        train_set, val_set = random_split(train_set, [40000, 10000])
        trainloader = DataLoader(train_set, batch_size=128,
                                 shuffle=train_shuffle)
        valloader = DataLoader(val_set, batch_size=64,
                               shuffle=False)
        testloader = DataLoader(test_set, batch_size=64,
                                shuffle=False)
        print("{} train samples, {} validation samples, {} test samples".format(len(train_set.indices), len(val_set.indices), len(test_set.targets)))
        return trainloader, valloader, testloader


def prepare_SVloader():
    selected_val = np.load("experiment7/SV_random.npz", allow_pickle=True)
    X_sv = selected_val["X"]
    Y_sv = selected_val["Y"]
    SVloader = DataLoader(mydataset(X_sv, Y_sv), batch_size=20, shuffle=False)
    return SVloader



if __name__ == "__main__":
    train1, val1, test2 = prepare_CIFAR10()
    train2, val2, test2 = prepare_CIFAR10()

    stop = None