"""
extract image similarity
"""
import torch
import torch.nn as nn
import torchvision
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from data import prepare_CIFAR10
from utils import device


def resnet18():
    net = torchvision.models.resnet50(pretrained=True)
    net.fc = nn.Identity()
    return net


def save_image_feature(model_name="resnet18"):
    if model_name == "resnet18":
        net = resnet18().to(device)
    trainloader, valloader, testloader = prepare_CIFAR10(train_shuffle=False)
    features_train = []
    with torch.no_grad():
        for X_batch, Y_batch in trainloader:
            X_batch = X_batch.to(device)
            features_batch = net(X_batch)
            features_train.append(features_batch.cpu().numpy())
    features_train = np.concatenate(features_train, axis=0)
    print(features_train.shape)
    np.save("experiment11/image_features_40000train.npy", features_train)


def save_image_similarity():
    image_features = np.load("experiment11/image_features_40000train.npy")
    similarity = image_features @ image_features.T
    print(similarity.shape)
    norms = np.linalg.norm(image_features, axis=1)
    norms = np.array([norms]).T
    norms_matrix = norms @ norms.T
    similarity /= norms_matrix
    np.save("experiment11/cosine_similarity_40000train.npy", similarity)


def check_similarity_by_class():
    trainloader, valloader, testloader = prepare_CIFAR10(train_shuffle=False)
    Y_train = trainloader.dataset.Label
    similarity = np.load("experiment11/cosine_similarity_40000train.npy")
    print(similarity.shape)
    average_similarity_by_class = np.zeros((10, 10))
    for i in range(10):
        for j in range(i, 10):
            cls_i_idx = (Y_train == i)
            cls_j_idx = (Y_train == j)
            sim = similarity[cls_i_idx][:, cls_j_idx]
            if i == 0 and j == 0:
                print(sim)
            average_similarity_by_class[i,j] = np.mean(sim)
            average_similarity_by_class[j,i] = np.mean(sim)
    ax = sns.heatmap(average_similarity_by_class, linewidth=0.5)
    plt.show()
    print(average_similarity_by_class)



if __name__ == "__main__":
    check_similarity_by_class()