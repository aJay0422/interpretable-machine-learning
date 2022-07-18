import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def dataset_category_get(category_num):
    """
    Get images with label == category_num for both train and test dataset
    :param category_num: which category of image want to get
    :return: img_all_train: shape(500, 3, 224, 224), 500 images from train dataset
             img_all_test: shape(100, 3, 224, 2240, 100 images from test dataset
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
    for train_image, train_label in trainloader:
        break
    for test_image, test_label in testloader:
        break
    print(torch.sum(test_label==category_num))
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


# if __name__ == "__main__":
#     img_all_train, img_all_test = dataset_category_get(0)
#     import matplotlib.pyplot as plt
#     for i in range(5):
#         plt.imshow(img_all_train[i * 10].permute(1, 2, 0))
#         plt.show()
#         plt.imshow(img_all_test[i * 10].permute(1, 2, 0))
#         plt.show()
