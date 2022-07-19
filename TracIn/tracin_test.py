import os
import torch
import torch.nn as nn
from torch.autograd import grad
import time
from model import resnet34

from data import dataset_category_get


# get train and test data from the same category
category_num = 0
img_all_train, img_all_test = dataset_category_get(category_num)   # (500,3,224,224) and (100,3,224,224)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create the network
net = resnet34()
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 10)   # using CIFAR10
net.to(device)

# load pretrained weights
# download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
weight_path = "model_weights/resnet34-333f7ec4.pth"
assert os.path.exists(weight_path), "file {} does not exist.".format(weight_path)
net.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
net.to(device)


# define loss function
loss_function = nn.CrossEntropyLoss()

img_all_train = img_all_train.unsqueeze(dim=1)
img_all_test = img_all_test.unsqueeze(dim=1)

label_train = (torch.ones(1) * category_num).long()
logits_train = net(img_all_train[0].to(device))
loss_train = loss_function(logits_train, label_train.to(device))
grad_z_train = grad(loss_train, net.parameters())
print(grad_z_train)