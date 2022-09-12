import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class mydataset(Dataset):
    def __init__(self, X, y):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(y)

    def __getitem__(self, item):
        p = np.random.rand(1)
        if p > 0.5:
            return torch.flip(self.Data[item], dims=[2]), self.Label[item]
        return self.Data[item], self.Label[item]

    def __len__(self):
        return len(self.Label)


def get_loss_acc(model, dataloader, criterion=nn.CrossEntropyLoss()):
    model.to(device)
    model.eval()
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
            logits = model(X_batch)
            y_pred = torch.argmax(logits, dim=1)
            correct += torch.sum(y_pred == Y_batch).cpu().numpy()
            loss = criterion(logits, Y_batch)
            total_loss += loss.item()
    acc = correct / total
    total_loss = total_loss / num_batches

    return total_loss, acc


def get_acc_by_cls(model, dataloader, criterion=nn.CrossEntropyLoss()):
    model.to(device)
    with torch.no_grad():
        model.eval()
        Y_pred = []
        Y_true = []
        for X_batch, Y_batch in dataloader:
            Y_true.append(Y_batch.numpy())
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            logits = model(X_batch)
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()
            Y_pred.append(y_pred)

    Y_pred = np.concatenate(Y_pred, axis=0)
    Y_true = np.concatenate(Y_true, axis=0)

    accs = []
    for cls in range(10):
        cls_index = (Y_true == cls)
        acc = np.mean(Y_pred[cls_index] == Y_true[cls_index])
        accs.append(acc)

    return accs


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
    train_set = torchvision.datasets.CIFAR10(root="../TracIn/datasets/CIFAR10/", train=True,
                                             download=False, transform=data_transform["train"])
    test_set = torchvision.datasets.CIFAR10(root="../TracIn/datasets/CIFAR10/", train=False,
                                            download=False, transform=data_transform["val"])
    X_train = train_set.data
    Y_train = train_set.targets
    X_test = test_set.data
    Y_test = test_set.targets

    # Normalize data
    ## Convert to [0,1]
    X_train = X_train / 255.
    X_test = X_test / 255.
    ## Standardize data
    m = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1)
    v = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1)
    X_train = (X_train - m) / v
    X_test = (X_test - m) / v
    ## switch channel to the 2nd dimension
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    if mode == "tt":
        train_set = mydataset(X_train, Y_train)
        test_set = mydataset(X_test, Y_test)
        trainloader = DataLoader(train_set, batch_size=128,
                                 shuffle=train_shuffle,
                                 num_workers=8, pin_memory=True)
        testloader = DataLoader(test_set, batch_size=64,
                                shuffle=False,
                                num_workers=8, pin_memory=True)
        print("{} train samples, {} test samples".format(len(train_set.targets), len(test_set.targets)))
        return trainloader, testloader
    elif mode == "tvt":
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.2, random_state=42)
        train_set = mydataset(X_train, Y_train)
        val_set = mydataset(X_val, Y_val)
        test_set = mydataset(X_test, Y_test)
        trainloader = DataLoader(train_set, batch_size=128,
                                 shuffle=train_shuffle)
        valloader = DataLoader(val_set, batch_size=64,
                               shuffle=False)
        testloader = DataLoader(test_set, batch_size=64,
                                shuffle=False)
        print("{} train samples, {} validation samples, {} test samples".format(len(train_set.Label),
                                                                                len(val_set.Label),
                                                                                len(test_set.Label)))
        return trainloader, valloader, testloader


if __name__ == "__main__":
    trainloader, valloader, testloader = prepare_CIFAR10()