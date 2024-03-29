import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_loss_acc(model, dataloader, criterion=nn.CrossEntropyLoss()):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            X_batch = batch[0]
            Y_batch = batch[1]
            if len(batch) == 3:
                index_batch = batch[2]
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            total += len(Y_batch)
            num_batches += 1
            outputs = model(X_batch)
            y_pred = torch.argmax(outputs, dim=1)
            correct += torch.sum(y_pred == Y_batch).cpu().numpy()
            loss = criterion(outputs, Y_batch)
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
        # print("Class {} Acc is {}".format(cls, acc))

    return accs





class mydataset(Dataset):
    def __init__(self, X, Y):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)

    def __getitem__(self, item):
        return self.Data[item], self.Label[item]

    def __len__(self):
        return len(self.Label)


def get_tracin(a, b):
    """
    get tracin score by 2 lists of gradients
    :param a: a List of gradient
    :param b: a List of gradient
    :return: inner product of 2 gradient
    """
    assert len(a) == len(b), "2 list of gradient must have the same length"
    return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])




