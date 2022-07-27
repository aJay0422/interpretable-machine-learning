import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import grad
import matplotlib.pyplot as plt
import numpy as np

from data import prepare_CIFAR10
from model import CNN_CIFAR10
from utils import mydataset


def norm(a):
    """
    Compute the L2 norm of a vector
    :param a: The target vector, separated into a list
    :return: the L2 norm of the target vector
    """
    return torch.sqrt(sum([torch.dot(at.flatten(), at.flatten()) for at in a]))


def cosine_similarity(vec_a, vec_b):
    """
    Compute the cosine similarity of 2 vectors
    :param vec_a: target vector a, separated into a list
    :param vec_b: target vector b, separated into a list
    :return: the cosine similarity
    """

    norm_a = norm(vec_a)
    norm_b = norm(vec_b)

    inner_product = sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(vec_a, vec_b)])
    return inner_product / (norm_a * norm_b)


if __name__ == "__main__":
    net = CNN_CIFAR10()
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10_3epochs.pth"))

    trainloader, valloader, testloader = prepare_CIFAR10()
    X_all_train = trainloader.dataset.Data
    Y_all_train = trainloader.dataset.Label

    # separate mislabelled data
    perm = np.random.permutation(len(Y_all_train))
    wrong_idx = perm[:4000]
    correct_idx = perm[4000:]
    X_wrong, Y_wrong = X_all_train[wrong_idx], Y_all_train[wrong_idx]
    X_correct, Y_correct = X_all_train[correct_idx], Y_all_train[correct_idx]
    Y_wrong = torch.LongTensor(np.random.choice([0,1,2,3,4,5,6,7,8,9], len(Y_wrong)))

    print(X_wrong.shape, Y_wrong.shape)
    print(X_correct.shape, Y_correct.shape)

    correct_loader = DataLoader(mydataset(X_correct, Y_correct), batch_size=200, shuffle=True)
    wrong_loader = DataLoader(mydataset(X_wrong, Y_wrong), batch_size=200, shuffle=True)
    loss_function = nn.CrossEntropyLoss()

    correct_grads = []
    net.eval()
    for X_batch, Y_batch in correct_loader:
        logits = net(X_batch)
        loss = loss_function(logits, Y_batch)
        grad_correct = grad(loss, net.parameters())
        grad_correct = [g for g in grad_correct]
        correct_grads.append(grad_correct)

    wrong_grads = []
    net.eval()
    for X_batch, Y_batch in wrong_loader:
        logits = net(X_batch)
        loss = loss_function(logits, Y_batch)
        grad_wrong = grad(loss, net.parameters())
        grad_wrong = [g for g in grad_wrong]
        wrong_grads.append(grad_wrong)

    all_grads = correct_grads + wrong_grads

    similarity_matrix = torch.eye(len(all_grads))
    for i in range(len(all_grads)):
        for j in range(i, len(all_grads)):
            similarity = cosine_similarity(all_grads[i], all_grads[j])
            similarity_matrix[i,j] = similarity
            similarity_matrix[j,i] = similarity

    plt.imshow(similarity_matrix, cmap="hot")
    plt.show()




    stop = None
