"""
Why a small fraction of high TracIn samples make model so terrible?
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import CNN_CIFAR10
from utils import device, get_loss_acc, get_tracin, get_acc_by_cls
from data import prepare_CIFAR10, prepare_SVloader, mydataset
from train import train_CNN_CIFAR10


def retrain():
    """
    Retrain the CNN_CIFAR10 network
    """
    # set up
    seed = 20220718
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare data
    trainloader, valloader, testloader = prepare_CIFAR10()

    # prepare model
    net = CNN_CIFAR10(num_classes=10)
    net.to(device)

    # train model
    epochs = 230
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    loss_function = nn.CrossEntropyLoss()
    print("Trained on {}".format(device))
    save_path = "experiment10/model_weights_230epochs/CNN_CIFAR10.pth"

    best_test_acc = 0
    for epoch in range(epochs):
        net.train()
        net.to(device)
        for X_batch, Y_batch in trainloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            # forward
            logits = net(X_batch)
            loss = loss_function(logits, Y_batch)
            # backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # evaluate
        with torch.no_grad():
            net.eval()
            # evaluate train
            train_loss, train_acc = get_loss_acc(net, trainloader, loss_function)
            # evaluate test
            test_loss, test_acc = get_loss_acc(net, testloader, loss_function)

        print("Epoch{}/{}  train loss: {}  test loss: {}  train acc: {}  test acc: {}".format(epoch + 1, epochs,
                                                                                              train_loss, test_loss,
                                                                                              train_acc, test_acc))

        # save model weights if it's the best
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), save_path)
            print("Saved")

        # save checkpoints
        checkpoint_path = f"experiment10/model_weights_230epochs/CNN_CIFAR10_epoch{epoch+1}.pth"
        torch.save(net.state_dict(), checkpoint_path)



def check_correlation(mv="random"):
    """
    Check whether SV represents Validation set well enough
    """
    # prepare data
    trainloader, valloader, testloader = prepare_CIFAR10()

    if mv == "high":
        file = np.load("experiment7/from_beginning(loss)/highlow0.1/selected_val.npz", allow_pickle=True)
    elif mv == "low":
        file = np.load("experiment7/from_beginning(loss)/lowlow0.1/selected_val.npz", allow_pickle=True)
    elif mv == "random":
        file = np.load("experiment7/SV_random.npz", allow_pickle=True)
    X_SV = file["X"]
    Y_SV = file["Y"]
    X_SV = torch.Tensor(X_SV).to(device)
    Y_SV = torch.LongTensor(Y_SV).to(device)

    net = CNN_CIFAR10().to(device)
    accs_val = []
    accs_sv = []
    losses_val = []
    losses_sv = []
    epochs = 230
    for i in range(epochs):
        # load weights
        net.load_state_dict(torch.load(f"experiment10/model_weights_230epochs/CNN_CIFAR10_epoch{i+1}.pth",
                                       map_location=device))
        # evaluate test set
        loss_val, acc_val = get_loss_acc(net, testloader)
        losses_val.append(loss_val)
        accs_val.append(acc_val)

        # evaluate selected val
        loss_function = nn.CrossEntropyLoss()
        with torch.no_grad():
            net.eval()
            logits = net(X_SV)
            loss_sv = loss_function(logits, Y_SV).cpu().numpy()
            Y_SV_pred = torch.argmax(logits, dim=1).cpu().numpy()
            acc_sv = np.mean(Y_SV_pred == Y_SV.cpu().numpy())
            losses_sv.append(loss_sv)
            accs_sv.append(acc_sv)

    # compute correlation
    loss_cor = np.corrcoef(losses_val, losses_sv)
    acc_cor = np.corrcoef(accs_val, accs_sv)
    print("Loss Correlation: {}".format(loss_cor))
    print("Accuracy Correlation: {}".format(acc_cor))

    # plot loss
    plt.plot(np.arange(epochs), losses_val, label="Validation")
    plt.plot(np.arange(epochs), losses_sv, label="Selected Val")
    plt.title("Loss Comparison, Correlation={}".format(loss_cor[0,1]))
    plt.legend()
    plt.show()

    # plot accuracy
    plt.plot(np.arange(epochs), accs_val, label="Validation")
    plt.plot(np.arange(epochs), accs_sv, label="Selected Val")
    plt.title("Accuracy Comparison, COrrelation={}".format(acc_cor[0,1]))
    plt.legend()
    plt.show()


def norm(a):
    return torch.sqrt(sum([torch.dot(at.flatten(), at.flatten()) for at in a]))


def save_TracIn(cosine=False):
    """
    Compute TracIn score by class with first 25 checkpoints.
    """
    net = CNN_CIFAR10().to(device)
    net.eval()
    loss_function = nn.CrossEntropyLoss()

    # prepare data
    trainloader, valloader, testloader = prepare_CIFAR10(mode="tvt")
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    selected_val = np.load("experiment7/SV_random.npz", allow_pickle=True)
    X_sv = torch.Tensor(selected_val["X"])
    Y_sv = torch.LongTensor(selected_val["Y"])

    for epoch in [1]:
        net.load_state_dict(torch.load(f"experiment10/model_weights_70epochs_wrong/CNN_CIFAR10_{epoch}.pth",
                                       map_location=device))
        net.eval()
        TracIn_by_cls = {}
        for cls in range(10):
            X_train_cls = X_train[Y_train == cls]
            Y_train_cls = Y_train[Y_train == cls]
            X_sv_cls = X_sv[Y_sv == cls]
            Y_sv_cls = Y_sv[Y_sv == cls]
            n_train = len(X_train_cls)
            n_sv = len(X_sv_cls)
            TracIn_this_cls = np.zeros((n_train, n_sv))

            # calculate test gradient
            test_gradients = []
            logits = net(X_sv_cls.to(device))
            for i in range(n_sv):
                loss = loss_function(logits[i].unsqueeze(0), Y_sv_cls[i].unsqueeze(0).to(device))
                grad_tmp = grad(loss, net.parameters(), retain_graph=True)
                grad_tmp = [g for g in grad_tmp]
                test_gradients.append(grad_tmp)

            # calculate train gradient and score
            for i in range(n_train):
                logit = net(X_train_cls[i].unsqueeze(0).to(device))
                loss = loss_function(logit, Y_train_cls[i].unsqueeze(0).to(device))
                grad_train = grad(loss, net.parameters())
                grad_train = [g for g in grad_train]
                for j, grad_test in enumerate(test_gradients):
                    score = get_tracin(grad_train, grad_test)
                    if cosine:
                        score /= (norm(grad_train) * norm(grad_test))
                    TracIn_this_cls[i, j] = score
            TracIn_by_cls[cls] = TracIn_this_cls

        print("Epoch {} finished".format(epoch))
        if cosine:
            np.save(f"experiment10/TracIn_by_class_cosine_wrong/TracIn_epoch{epoch}.npy", TracIn_by_cls)
        else:
            np.save(f"experiment10/TracIn_by_class/TracIn_epoch{epoch}.npy", TracIn_by_cls)
        if epoch == 1:
            print(TracIn_by_cls[0])


def get_TracIn(ckpts=np.arange(1,26), cosine=False):
    TracIn_by_cls = {i: np.zeros((4000, 20)) for i in range(10)}
    for epoch in ckpts:
        if cosine:
            path = f"experiment10/TracIn_by_class_cosine/TracIn_epoch{epoch}.npy"
        else:
            path = f"experiment10/TracIn_by_class/TracIn_epoch{epoch}.npy"
        TracIn_by_cls_epoch = np.load(path, allow_pickle=True).item()
        for cls in range(10):
            TracIn_by_cls[cls] += TracIn_by_cls_epoch[cls]
    return TracIn_by_cls


def select_train(TracIn_by_cls, ratio=0.1, mode="high", category=[0,1,2,3,4,5,6,7,8,9]):
    trainloader, valloader, testloader = prepare_CIFAR10()
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label

    X_train_selected = []
    Y_train_selected = []
    for cls in range(10):
        if cls not in category:
            continue
        cls_TracIn = TracIn_by_cls[cls]
        X_train_cls = X_train[Y_train == cls]
        Y_train_cls = Y_train[Y_train == cls]
        n_train_cls = len(Y_train_cls)
        n_keep_cls = int(n_train_cls * ratio)

        sort_index = np.argsort(np.mean(cls_TracIn, axis=1))[::-1]
        if mode == "high":
            keep_index = list(sort_index[:n_keep_cls])
        elif mode == "low":
            keep_index = list(sort_index[-n_keep_cls:])
        elif mode == "mid":
            half_drop = int((n_train_cls - n_keep_cls) / 2)
            keep_index = list(sort_index[half_drop:-half_drop])
        elif mode.startswith('quantile'):
            q = int(mode[-1])
            keep_index = list(sort_index[::-1][q * n_keep_cls: (q+1) * n_keep_cls])
        elif mode == "even":
            keep_index = []
            print(len(sort_index))
            for q in range(10):
                keep_index_q = np.random.choice(sort_index[q*400:(q+1)*400], size=int(n_keep_cls / 10),
                                                replace=False)
                keep_index.append(keep_index_q)
            keep_index = np.concatenate(keep_index, axis=0)

        X_train_selected.append(X_train_cls[keep_index])
        Y_train_selected.append(Y_train_cls[keep_index])
    X_train_selected = torch.cat(X_train_selected, dim=0)
    Y_train_selected = torch.cat(Y_train_selected, dim=0)
    trainloader_selected = DataLoader(mydataset(X_train_selected, Y_train_selected),
                                      batch_size=256, shuffle=True)
    return trainloader_selected


def experiment(ratio=0.1, mode="high", category=[0,1,2,3,4,5,6,7,8,9], cosine=False):
    trainloader, valloader, testloader = prepare_CIFAR10()
    TracIn_by_cls = get_TracIn(ckpts=[1,2,3], cosine=cosine)
    trainloader_selected = select_train(TracIn_by_cls, ratio=ratio, mode=mode, category=category)
    if len(category) == 10:
        cls_suffix = ""
    else:
        cls_suffix = "".join(list(map(str, category)))
    if cosine:
        save_path = f"experiment10/select_train_cosine/CNN_CIFAR10_{mode}{ratio}_{cls_suffix}.pth"
    else:
        save_path = f"experiment10/select_train/CNN_CIFAR10_{mode}{ratio}_{cls_suffix}.pth"
    train_CNN_CIFAR10(70, trainloader_selected, valloader, save_path=save_path)


def evaluation():
    selected_val = np.load("experiment7/SV_random.npz", allow_pickle=True)
    X_sv = torch.Tensor(selected_val["X"])
    Y_sv = torch.LongTensor(selected_val["Y"])
    trainloader, valloader, testloader = prepare_CIFAR10()

    net = CNN_CIFAR10().to(device)
    net.load_state_dict(torch.load("experiment10/select_train/CNN_CIFAR10_high0.1.pth", map_location=device))

    # get sv accuracy
    logits = net(X_sv.to(device))
    Y_sv_pred = torch.argmax(logits, dim=1)
    sv_acc = np.mean(Y_sv_pred.eq(Y_sv.to(device)).cpu().numpy())

    # get train accuracy
    train_acc = get_loss_acc(net, trainloader)

    # get val accuracy
    val_acc = get_loss_acc(net, valloader)

    # get test accuracy
    test_acc = get_loss_acc(net, testloader)

    print(sv_acc)
    print(train_acc)
    print(val_acc)
    print(test_acc)


def prepare_CIFAR10_wrong(mode="tvt", seed=42, wrong=True):
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    train_set = torchvision.datasets.CIFAR10(root="datasets/CIFAR10/", train=True,
                                             download=False, transform=data_transform["train"])
    test_set = torchvision.datasets.CIFAR10(root="datasets/CIFAR10/", train=False,
                                            download=False, transform=data_transform["val"])
    X_train = train_set.data.transpose((0, 3, 1, 2))
    X_test = test_set.data.transpose((0, 3, 1, 2))
    Y_train = train_set.targets
    Y_test = test_set.targets
    if mode == "tt":
        if wrong:
            # assign wrong labels
            n_wrong = int(0.1 * len(Y_train))
            np.random.seed(seed)
            wrong_index = np.random.permutation(len(Y_train))[:n_wrong]
            for index in wrong_index:
                true_label = Y_train[index]
                wrong_labels = [0,1,2,3,4,5,6,7,8,9]
                wrong_labels.remove(true_label)
                Y_train[index] = np.random.choice(wrong_labels)
        train_set = mydataset(X_train, Y_train)
        test_set = mydataset(X_test, Y_test)
        trainloader = DataLoader(train_set, batch_size=256,
                                 shuffle=True)
        testloader = DataLoader(test_set, batch_size=64,
                                shuffle=False)
        print(len(X_train), len(X_test))
        return trainloader, testloader

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.2, random_state=42)

    if wrong:
        # assign wrong labels
        n_wrong = int(0.1 * len(Y_train))
        np.random.seed(seed)
        wrong_index = np.random.permutation(len(Y_train))[:n_wrong]
        for index in wrong_index:
            true_label = Y_train[index]
            wrong_labels = [0,1,2,3,4,5,6,7,8,9]
            wrong_labels.remove(true_label)
            Y_train[index] = np.random.choice(wrong_labels)

    train_set = mydataset(X_train, Y_train)
    val_set = mydataset(X_val, Y_val)
    test_set = mydataset(X_test, Y_test)
    trainloader = DataLoader(train_set, batch_size=256,
                             shuffle=True)
    valloader = DataLoader(val_set, batch_size=64,
                           shuffle=True)
    testloader = DataLoader(test_set, batch_size=64,
                            shuffle=False)
    print(len(X_train), len(X_val), len(X_test))
    return trainloader, valloader, testloader


def experiment2(ratio=0.1, mode="high", category=[0,1,2,3,4,5,6,7,8,9]):
    trainloader, valloader, testloader = prepare_CIFAR10_wrong()
    TracIn_by_cls = get_TracIn()
    trainloader_selected = select_train(TracIn_by_cls, ratio=ratio, mode=mode, category=category)
    if len(category) == 10:
        cls_suffix = ""
    else:
        cls_suffix = "".join(list(map(str, category)))
    save_path = f"experiment10/select_train/CNN_CIFAR10_{mode}{ratio}_{cls_suffix}.pth"
    train_CNN_CIFAR10(70, trainloader_selected, valloader, save_path=save_path)


def check_label():
    trainloader, valloader, testloader = prepare_CIFAR10()
    trainloader_wrong, valloader_wrong, testloader_wron = prepare_CIFAR10_wrong()
    TracIn_wrong = np.load("experiment10/TracIn_by_class_cosine_wrong/TracIn_epoch1.npy", allow_pickle=True).item()
    Y_wrong = trainloader_wrong.dataset.Label
    Y_true = trainloader.dataset.Label
    for cls in range(10):
        Y_true_cls = Y_true[Y_true == cls]
        Y_wrong_cls = Y_wrong[Y_true == cls]
        TracIn_wrong_cls = TracIn_wrong[cls]
        sort_index = np.argsort(np.mean(TracIn_wrong_cls, axis=1))[::-1]




def evaluate_TracIn_CosIn():
    for epoch in range(1, 2):
        TracIn = np.load(f"experiment10/TracIn_by_class/TracIn_epoch{epoch}.npy", allow_pickle=True).item()
        CosIn = np.load(f"experiment10/TracIn_by_class_cosine/TracIn_epoch{epoch}.npy", allow_pickle=True).item()
        TracIn_scores = []
        CosIn_scores = []
        for cls in range(10):
            TracIn_scores.append(TracIn[cls].reshape(-1))
            CosIn_scores.append(CosIn[cls].reshape(-1))
        for cls in range(10):
            corr_cls = np.corrcoef(TracIn_scores[cls], CosIn_scores[cls])[0,1]
            print(f"Correlation of class {cls} is {corr_cls}")
        TracIn_scores = np.concatenate(TracIn_scores, axis=0)
        CosIn_scores = np.concatenate(CosIn_scores, axis=0)
        corr = np.corrcoef(TracIn_scores, CosIn_scores)
        print(f"Correlation at epoch{epoch} is {corr}")

def evaluation_CosIn():
    keep_ratios = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    accs_all = {}
    trainloader, valloader, testloader = prepare_CIFAR10()

    # get baseline acc
    net = CNN_CIFAR10().to(device)
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))
    _, baseline_acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())

    # get random keep accuracy
    accs = np.zeros((len(keep_ratios), 5))
    accs[-1,:] = baseline_acc
    for i, keep_ratio in enumerate(keep_ratios[:-1]):
        for j in range(5):
            net.load_state_dict(torch.load(f"experiment7/random_keep/CNN_CIFAR10_rndkeep{keep_ratio}_{j+1}.path", map_location=device))
            _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
            accs[i, j] = acc
    print(accs)
    accs_all["random"] = np.mean(accs, axis=1)
    print(accs_all["random"])

    MT = ["high", "mid", "even"]
    for mt in MT:
        name = mt
        accs = np.zeros(len(keep_ratios))
        accs[-1] = baseline_acc
        for i, keep_ratio in enumerate(keep_ratios[:-1]):
            net.load_state_dict(torch.load(f"experiment10/select_train_cosine/CNN_CIFAR10_{mt}{keep_ratio}_.pth",
                                           map_location=device))
            _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
            accs[i] = acc
        accs_all[name] = accs

    fig = plt.figure(figsize=(10, 10))
    for name in accs_all:
        plt.plot(keep_ratios, accs_all[name], ".-", label=name)
    plt.xlabel("Keep Ratio")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # net = CNN_CIFAR10().to(device)
    # trainloader, valloader, testloader = prepare_CIFAR10()
    # svloader = prepare_SVloader()
    # for q in range(10):
    #     print("Quantile {}".format(q))
    #     net.load_state_dict(torch.load(f"experiment10/select_train_cosine/CNN_CIFAR10_quantile{q}0.1_.pth",
    #                                    map_location=device))
    #     _, acc_sv = get_loss_acc(net, svloader)
    #     _, acc_test = get_loss_acc(net, testloader)
    #     print("SV Acc: {}".format(acc_sv))
    #     print("Test Acc: {}".format(acc_test))
    # print(np.mean(accs))
    # for i, acc in enumerate(accs):
    #     print("Class {}, Acc {}".format(i, acc))

    # save_TracIn(cosine=True)
    # trainloader, valloader, testloader = prepare_CIFAR10_wrong()
    # net = CNN_CIFAR10().to(device)
    # save_path = "experiment10/model_weights_70epochs_wrong/CNN_CIFAR10.pth"
    # epoch_path = "experiment10/model_weights_70epochs_wrong/CNN_CIFAR10_{}.pth"
    # train_CNN_CIFAR10(70, trainloader, valloader, save_path=save_path, epoch_path=epoch_path)

    # experiment(ratio=0.1, mode="even", cosine=True)
    # experiment(ratio=0.2, mode="even", cosine=True)
    # experiment(ratio=0.4, mode="even", cosine=True)
    # experiment(ratio=0.6, mode="even", cosine=True)
    # experiment(ratio=0.8, mode="even", cosine=True)
    # experiment(ratio=0.2, mode="mid", cosine=True)
    # experiment(ratio=0.4, mode="mid", cosine=True)
    # experiment(ratio=0.6, mode="mid", cosine=True)
    # experiment(ratio=0.8, mode="mid", cosine=True)

    evaluation_CosIn()


    # CosIn_epoch1 = np.load("experiment10/TracIn_by_class_cosine/TracIn_epoch1.npy",
    #                        allow_pickle=True).item()
    # CosIn_epoch2 = np.load("experiment10/TracIn_by_class_cosine/TracIn_epoch2.npy",
    #                        allow_pickle=True).item()
    # CosIn_epoch3 = np.load("experiment10/TracIn_by_class_cosine/TracIn_epoch3.npy",
    #                        allow_pickle=True).item()
    # stop = None

    # evaluate_TracIn_CosIn()
    # save_TracIn(cosine=True)