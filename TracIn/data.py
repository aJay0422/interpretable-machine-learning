import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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


def prepare_CIFAR10(img_size=32):
    if img_size == 32:
        data_transform = {
            "train": transforms.Compose([transforms.ToTensor(),
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
    X_train = train_set.data.transpose((0, 3, 1, 2))
    X_test = test_set.data.transpose((0, 3, 1, 2))
    Y_train = train_set.targets
    Y_test = test_set.targets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.2)   # split 5000 validation data

    train_set = mydataset(X_train, Y_train)
    val_set = mydataset(X_val, Y_val)
    test_set = mydataset(X_test, Y_test)

    trainloader = DataLoader(train_set, batch_size=256,
                             shuffle=True, num_workers=8)
    valloader = DataLoader(val_set, batch_size=64,
                           shuffle=False, num_workers=8)
    testloader = DataLoader(test_set, batch_size=64,
                            shuffle=False, num_workers=8)
    print(len(X_train), len(X_val), len(X_test))
    return trainloader, valloader, testloader



if __name__ == "__main__":
    prepare_CIFAR10()