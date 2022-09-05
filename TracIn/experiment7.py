import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from model import CNN_CIFAR10
from utils import device, get_loss_acc, get_acc_by_cls
from influence_functions import tracin_multi_multi, tracin_self
from data import prepare_CIFAR10, mydataset
from train import train_CNN_CIFAR10


LABEL_DECODER = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

def ST_check():
    """
    检查不同样本量下ST中各类别样本个数
    """
    experiment_path = "experiment6/"
    ds = [9, 3, 1, 0.3, 0.1, 0.03]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    for t, d in enumerate(ds):
        i = t // 3
        j = t % 3
        folder_name = "easyhigh" + str(d)
        folder_path = experiment_path + folder_name
        file = np.load(folder_path + "/selected_train.npz", allow_pickle=True)
        Y = file["Y"]
        ax[i, j].hist(Y)
        ax[i, j].title.set_text("ST size = {}".format(len(Y)))
    plt.show()


def cls_difficulty_check():
    trainloader, valloader, testloader = prepare_CIFAR10()

    # epoch 3
    net = CNN_CIFAR10()
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10_epoch3.pth", map_location=device))
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    entropies = []
    X_val = valloader.dataset.Data
    Y_val = valloader.dataset.Label
    with torch.no_grad():
        net.eval()
        for X, Y in zip(X_val, Y_val):
            X = X.to(device).unsqueeze(0)
            Y = Y.to(device).unsqueeze(0)
            logit = net(X)
            loss = loss_function(logit, Y)
            entropies.append(loss.cpu().item())
    entropies = np.array(entropies)
    df = pd.DataFrame({"entropy": entropies, "label": Y_val})
    df_mean = df.groupby("label").mean()
    series1 = df_mean["entropy"]

    # at convergence
    net = CNN_CIFAR10()
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    entropies = []
    X_val = valloader.dataset.Data
    Y_val = valloader.dataset.Label
    with torch.no_grad():
        net.eval()
        for X, Y in zip(X_val, Y_val):
            X = X.to(device).unsqueeze(0)
            Y = Y.to(device).unsqueeze(0)
            logit = net(X)
            loss = loss_function(logit, Y)
            entropies.append(loss.cpu().item())
    entropies = np.array(entropies)
    print(np.min(entropies))
    df = pd.DataFrame({"entropy": entropies, "label": Y_val})
    df_mean = df.groupby("label").mean()
    series2 = df_mean["entropy"]

    df = pd.DataFrame({"epoch 3": series1,
                       "convergence": series2})
    df.plot.bar()
    plt.xticks(np.arange(10), ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"], rotation=45)
    plt.ylabel("average loss")
    plt.title("Loss by Class")
    plt.tight_layout()
    plt.show()


def cls_acc_check():
    trainloader, valloader, testloader = prepare_CIFAR10()

    # epoch 3
    net = CNN_CIFAR10()
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10_epoch3.pth", map_location=device))
    net.to(device)
    corrects = []
    X_val = valloader.dataset.Data
    Y_val = valloader.dataset.Label
    with torch.no_grad():
        net.eval()
        for X, Y in zip(X_val, Y_val):
            X = X.to(device).unsqueeze(0)
            Y = Y.to(device)
            logit = net(X)
            Y_pred = torch.argmax(logit, dim=1).squeeze()
            corrects.append(Y_pred.eq(Y).cpu().numpy())
    corrects = np.array(corrects)
    df = pd.DataFrame({"corrects": corrects, "label": Y_val})
    df_mean = df.groupby("label").mean()
    series1 = df_mean["corrects"]

    # at convergence
    net = CNN_CIFAR10()
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))
    net.to(device)
    corrects = []
    X_val = valloader.dataset.Data
    Y_val = valloader.dataset.Label
    with torch.no_grad():
        net.eval()
        for X, Y in zip(X_val, Y_val):
            X = X.to(device).unsqueeze(0)
            Y = Y.to(device)
            logit = net(X)
            Y_pred = torch.argmax(logit, dim=1).squeeze()
            corrects.append(Y_pred.eq(Y).cpu().numpy())
    corrects = np.array(corrects)
    df = pd.DataFrame({"corrects": corrects, "label": Y_val})
    df_mean = df.groupby("label").mean()
    series2 = df_mean["corrects"]

    df = pd.DataFrame({"epoch 3": series1,
                       "convergence": series2})
    df.plot.bar()
    plt.xticks(np.arange(10),
               ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"], rotation=45)
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy by Class")
    plt.tight_layout()
    plt.show()


def EL2N(logits, target):
    """
    Compute the EL2N distance between a batch of logits and their target
    :param logits: shape [n_samples, n_classes]
    :param target: shape [n_samples]
    :return: a scaler
    """
    # convert logits to probs
    logits = torch.exp(logits)
    probs = logits / torch.sum(logits, dim=1).unsqueeze(1)
    num_classes = probs.shape[1]

    # one hot encode target
    target = nn.functional.one_hot(target, num_classes=num_classes)

    # compute distance
    distance = torch.sum(torch.pow(probs - target, 2), dim=1)

    return distance


def select_validation(net, valloader, n=30, mode="high", criterion="loss"):
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10_epoch3.pth", map_location=device))
    net.to(device)

    X_val = valloader.dataset.Data
    Y_val = valloader.dataset.Label

    if mode == "random":
        criterion = None

    if criterion == "EL2N":
        # calculate EL2N score for each sample in valloader
        scores = []
        with torch.no_grad():
            net.eval()
            for X_batch, Y_batch in valloader:
                X = X_batch.to(device)
                Y = Y_batch.to(device)
                logits = net(X)
                score = EL2N(logits, Y)
                scores.append(score.cpu())
        scores = torch.cat(scores, dim=0).numpy()
    elif criterion == "loss":
        # calculate crossentropy loss for each sample in valloader
        loss_function = nn.CrossEntropyLoss()
        scores = []
        with torch.no_grad():
            net.eval()
            for X, Y in zip(X_val, Y_val):
                X = X.to(device).unsqueeze(0)
                Y = Y.to(device).unsqueeze(0)
                logit = net(X)
                score = loss_function(logit, Y)
                scores.append(score.cpu().item())
        scores = np.array(scores)
    X_return = []
    Y_return = []

    if mode == "high":
        for i in range(10):
            X_cls = X_val[Y_val == i]
            Y_cls = Y_val[Y_val == i]
            score_cls = scores[Y_val == i]
            sort_index = np.argsort(score_cls)
            X_return.append(X_cls[sort_index[-(n // 10):]])
            Y_return.append(Y_cls[sort_index[-(n // 10):]])
    elif mode == "low":
        for i in range(10):
            X_cls = X_val[Y_val == i]
            Y_cls = Y_val[Y_val == i]
            score_cls = scores[Y_val == i]
            sort_index = np.argsort(score_cls)
            X_return.append(X_cls[sort_index[:(n // 10)]])
            Y_return.append(Y_cls[sort_index[:(n // 10)]])
    elif mode == "random":
        for i in range(10):
            keep_index = np.random.permutation(1000)[:(n // 10)]
            cls_index = (Y_val == i)
            X_return.append(X_val[cls_index][keep_index])
            Y_return.append(Y_val[cls_index][keep_index])



    X_return = torch.cat(X_return, dim=0)
    Y_return = torch.cat(Y_return, dim=0)

    return {"X": X_return, "Y": Y_return}


def save_SV(mode="random", n=200):
    trainloader, valloader, testloader = prepare_CIFAR10()
    if mode == "random":
        X_return = []
        Y_return = []
        X_val = valloader.dataset.Data
        Y_val = valloader.dataset.Label
        for i in range(10):
            keep_index = np.random.permutation(1000)[:(n // 10)]
            cls_index = (Y_val == i)
            X_return.append(X_val[cls_index][keep_index])
            Y_return.append(Y_val[cls_index][keep_index])

    X_return = torch.cat(X_return, dim=0)
    Y_return = torch.cat(Y_return, dim=0)

    np.savez("experiment7/SV_random.npz", X=X_return, Y=Y_return)


def save_TracIn(ckpts=[20, 40, 60], SV_mode="random"):
    net = CNN_CIFAR10().to(device)
    trainloader, valloader, testloader = prepare_CIFAR10()
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label

    selected_val = np.load(f"experiment7/SV_{SV_mode}.npz", allow_pickle=True)
    X_val = torch.Tensor(selected_val["X"])
    Y_val = torch.LongTensor(selected_val["Y"])
    print(X_val.size())
    print(Y_val.size())

    TracIn_by_cls = {}
    for cls in range(10):
        cls_idx_train = (Y_train == cls)
        cls_idx_val = (Y_val == cls)
        n_train_cls = len(Y_train[cls_idx_train])
        n_val_cls = len(Y_val[cls_idx_val])
        TracIn_by_cls[cls] = np.zeros((n_train_cls, n_val_cls))
        for i in ckpts:
            net.load_state_dict(torch.load(f"model_weights/weights_70epochs/CNN_CIFAR10_epoch{i}.pth",
                                           map_location=device))
            TracIn_by_cls[cls] += tracin_multi_multi(net, X_train[cls_idx_train], X_val[cls_idx_val],
                                                     Y_train[cls_idx_train], Y_val[cls_idx_val])
        print(f"Class {cls} finished")

    np.savez(f"experiment7/TracIn_random_{ckpts[0]}_{ckpts[1]}_{ckpts[2]}.npz", TracIn_by_cls=TracIn_by_cls)


def save_TracIn2(ckpts=[20, 40, 60], SV_mode="random"):
    net = CNN_CIFAR10().to(device)
    trainloader, valloader, testloader = prepare_CIFAR10()
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label

    selected_val = np.load(f"experiment7/SV_{SV_mode}.npz", allow_pickle=True)
    X_val = torch.Tensor(selected_val["X"])
    Y_val = torch.LongTensor(selected_val["Y"])

    n_train = len(Y_train)
    n_val = len(Y_val)
    TracIn_all = np.zeros((n_train, n_val))
    for i in ckpts:
        net.load_state_dict(torch.load(f"model_weights/weights_70epochs/CNN_CIFAR10_epoch{i}.pth",
                                       map_location=device))
        TracIn_all += tracin_multi_multi(net, X_train, X_val,
                                         Y_train, Y_val)
        print(f"checkpoint {i} finished")

    np.savez(f"experiment7/TracIn_random_{ckpts[0]}_{ckpts[1]}_{ckpts[2]}_all.npz", TracIn_all=TracIn_all)



def select_train(net, trainloader, selected_val, keep_ratio, mode="high", TracIn_name="TracIn_random_20_40_60.npz"):
    # prepare model
    net.to(device)

    # prepare data
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    X_val = selected_val["X"]
    Y_val = selected_val["Y"]

    # compute TracIn by class
    # TracIn_by_cls = {}
    # for cls in range(10):
    #     cls_idx_train = (Y_train == cls)
    #     cls_idx_val = (Y_val == cls)
    #     n_train_tmp = len(Y_train[cls_idx_train])
    #     n_val_tmp = len(Y_val[cls_idx_val])
    #     TracIn_by_cls[cls] = np.zeros((n_train_tmp, n_val_tmp))
    #     for i in [20, 40, 60]:
    #         # load weight
    #         net.load_state_dict(torch.load(f"model_weights/weights_70epochs/CNN_CIFAR10_epoch{i}.pth",
    #                                        map_location=device))
    #         TracIn_by_cls[cls] += tracin_multi_multi(net, X_train[cls_idx_train], X_val[cls_idx_val],
    #                                             Y_train[cls_idx_train], Y_val[cls_idx_val])
    TracIn_by_cls = np.load(f"experiment7/{TracIn_name}", allow_pickle=True)["TracIn_by_cls"].item()

    n_keep_per_cls = int(len(X_train) / 10 * keep_ratio)

    X_selected = []
    Y_selected = []
    if mode == "high":
        for cls in range(10):
            TracIn_mean = np.mean(TracIn_by_cls[cls], axis=1)
            index = list(np.argsort(TracIn_mean)[-n_keep_per_cls:])
            X_selected.append(X_train[Y_train == cls][index])
            Y_selected.append(Y_train[Y_train == cls][index])
    elif mode == "low":
        for cls in range(10):
            TracIn_mean = np.mean(TracIn_by_cls[cls], axis=1)
            index = list(np.argsort(TracIn_mean)[:n_keep_per_cls])
            X_selected.append(X_train[Y_train == cls][index])
            Y_selected.append(Y_train[Y_train == cls][index])
    elif mode == "mid":
        for cls in range(10):
            TracIn_mean = np.mean(TracIn_by_cls[cls], axis=1)
            half_drop = int((len(TracIn_mean) - n_keep_per_cls) // 2)
            index = list(np.argsort(TracIn_mean)[half_drop:-half_drop])
            print(len(index), "for this class")
            X_selected.append(X_train[Y_train == cls][index])
            Y_selected.append(Y_train[Y_train == cls][index])

    X_selected = np.concatenate(X_selected, axis=0)
    Y_selected = np.concatenate(Y_selected, axis=0)
    return {"X": X_selected, "Y": Y_selected}


def experiment():
    experiment_path = "experiment7/from_beginning(loss)"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    MV = ["random"]
    MT = ["high", "low", "mid"]
    keep_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]
    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10().to(device)
    for mv in MV:
        for mt in MT:
            for keep_ratio in keep_ratios:
                setting_path = experiment_path + f"/{mv}{mt}{keep_ratio}"
                if not os.path.exists(setting_path):
                    os.mkdir(setting_path)

                # select validation and train
                selected_val = np.load(f"experiment7/SV_{mv}.npz", allow_pickle=True)
                X_val = selected_val["X"]
                Y_val = selected_val["Y"]
                selected_val = {}
                selected_val["X"] = torch.Tensor(X_val)
                selected_val["Y"] = torch.LongTensor(Y_val)
                selected_train = select_train(net, trainloader, selected_val, keep_ratio, mt)
                # np.savez(setting_path + "/selected_val.npz", X=selected_val["X"], Y=selected_val["Y"])
                # np.savez(setting_path + "/selected_train.npz", X=selected_train["X"], Y=selected_train["Y"])

                # create a new trainloader
                selected_trainset = mydataset(selected_train["X"], selected_train["Y"])
                selected_trainloader = DataLoader(selected_trainset, batch_size=256, shuffle=True)

                # train model and save checkpoint
                print(f"SV: {mv}, ST: {mt}, keep: {keep_ratio} finished")
                save_path = setting_path + "/CNN_CIFAR10.pth"
                train_CNN_CIFAR10(70, selected_trainloader, valloader,
                                  save_path=save_path)


def experiment_random_keep(seed=42, keep_ratios=[0.1, 0.2, 0.4, 0.6, 0.8]):
    experiment_path = "experiment7/random_keep/"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    trainloader, valloader, testloader = prepare_CIFAR10()
    for keep_ratio in keep_ratios:
        for i in range(5):
            this_seed = seed + i
            n_train = len(trainloader.dataset.Label)
            n_keep = int(n_train * keep_ratio)
            np.random.seed(this_seed)
            keep_index = list(np.random.permutation(n_train)[:n_keep])
            X_train_keep = trainloader.dataset.Data[keep_index]
            Y_train_keep = trainloader.dataset.Label[keep_index]
            trainloader_keep = DataLoader(mydataset(X_train_keep, Y_train_keep),
                                          batch_size=256, shuffle=True)

            save_path = experiment_path + f"CNN_CIFAR10_rndkeep{keep_ratio}_{i+1}.path"
            train_CNN_CIFAR10(70, trainloader_keep, valloader, seed=20220718+i, save_path=save_path)


def evaluation():
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
            net.load_state_dict(torch.load(f"experiment7/random_keep/CNN_CIFAR10_rndkeep{keep_ratio}_{i+1}.path", map_location=device))
            _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
            accs[i, j] = acc
    accs_all["random"] = accs

    MV = ["random"]
    MT = ["high", "mid", "low"]
    for mv in MV:
        for mt in MT:
            name = mv + " " + mt
            accs = np.zeros(len(keep_ratios))
            accs[-1] = baseline_acc
            for i, keep_ratio in enumerate(keep_ratios[:-1]):
                if mv != "random" and keep_ratio == 0.6:
                    continue
                net.load_state_dict(torch.load(f"experiment7/from_beginning(loss)/{mv}{mt}{keep_ratio}/CNN_CIFAR10.pth",
                                               map_location=device))
                _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
                accs[i] = acc
            accs_all[name] = accs

    fig = plt.figure(figsize=(10, 10))
    plt.plot(keep_ratios, np.mean(accs_all["random"], axis=1), label="random")
    print("random")
    print(accs_all["random"])
    for mv in MV:
        for mt in MT:
            name = mv + " " + mt
            plt.plot(keep_ratios, accs_all[name], ".-", label=name)
            print(name)
            print(accs_all[name])
    plt.xlabel("Keep Ratio")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.show()


def reproduce_EL2N():
    net = CNN_CIFAR10().to(device)
    net.load_state_dict(torch.load("model_weights/weights_70epochs/CNN_CIFAR10_epoch8.pth", map_location=device))
    trainloader, valloader, testloader = prepare_CIFAR10()
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    scores = []
    with torch.no_grad():
        net.eval()
        for X, Y in zip(X_train, Y_train):
            X = X.to(device).unsqueeze(0)
            Y = Y.to(device).unsqueeze(0)
            logit = net(X)
            score = EL2N(logit, Y).squeeze()
            scores.append(score.cpu().numpy())
    scores = np.array(scores)

    keep_ratios = [0.1, 0.2, 0.4, 0.8]
    n_train = len(Y_train)
    for keep_ratio in keep_ratios:
        n_keep = int(n_train * keep_ratio)
        keep_index = np.argsort(scores)[-n_keep:]
        trainset_keep = mydataset(X_train[keep_index], Y_train[keep_index])
        trainloader_keep = DataLoader(trainset_keep, batch_size=256, shuffle=True)
        print("Trained on {} samples".format(len(keep_index)))

        train_CNN_CIFAR10(70, trainloader_keep, valloader,
                          save_path=f"experiment7/reproduce_EL2N/CNN_CIFAR10_keep{keep_ratio}.pth")


def reproduce_selfTracIn():
    net = CNN_CIFAR10().to(device)
    trainloader, valloader, testloader = prepare_CIFAR10()
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    n_train = len(Y_train)

    # compute TracIn
    TracIn_all = np.zeros(n_train)
    for i in [20, 40, 60]:
        net.load_state_dict(torch.load(f"model_weights/weights_70epochs/CNN_CIFAR10_epoch{i}.pth",
                                       map_location=device))
        TracIn_all += tracin_self(net, X_train, Y_train)

    keep_ratios = [0.1, 0.2, 0.4, 0.8]
    for keep_ratio in keep_ratios:
        n_keep = int(n_train * keep_ratio)
        keep_index = np.argsort(TracIn_all)[-n_keep:]
        trainset_keep = mydataset(X_train[keep_index], Y_train[keep_index])
        trainloader_keep = DataLoader(trainset_keep, batch_size=256, shuffle=True)
        print("Trained on {} samples".format(len(keep_index)))

        train_CNN_CIFAR10(70, trainloader_keep, valloader,
                          save_path=f"experiment7/reproduce_selfTracIn/CNN_CIFAR10_keep{keep_ratio}.pth")


def experiment_ckpts(TracIn_name="TracIn_random_20_40_60.npz"):
    experiment_path = "experiment7/diff_ckpts"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    ckpts_str = TracIn_name.split("_")
    ckpts_str = "_" + "_".join(ckpts_str[2:5])
    ckpts_str = ckpts_str[:-4]

    MV = ["random"]
    MT = ["high", "mid"]
    keep_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]
    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10().to(device)
    for mv in MV:
        for mt in MT:
            for keep_ratio in keep_ratios:
                # select validation and train
                selected_val = np.load(f"experiment7/SV_{mv}.npz", allow_pickle=True)
                X_val = selected_val["X"]
                Y_val = selected_val["Y"]
                selected_val = {}
                selected_val["X"] = torch.Tensor(X_val)
                selected_val["Y"] = torch.LongTensor(Y_val)
                selected_train = select_train(net, trainloader, selected_val,
                                              keep_ratio, mt, TracIn_name)

                # create a new trainloader
                selected_trainset = mydataset(selected_train["X"], selected_train["Y"])
                selected_trainloader = DataLoader(selected_trainset,
                                                  batch_size=256,
                                                  shuffle=True)

                # train model and save checkpoints
                print(f"SV: {mv}, ST: {mt}, keep: {keep_ratio} ckpts: {ckpts_str[1:]}finished")
                save_path = experiment_path + f"/CNN_CIFAR10{ckpts_str}_{mv}_{mt}_{keep_ratio}.pth"
                train_CNN_CIFAR10(70, selected_trainloader, valloader,
                                  save_path=save_path)


def evaluation_ckpts(ckpts="1_2_3"):
    keep_ratios = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    accs_all = {}
    trainloader, valloader, testloader = prepare_CIFAR10()

    # get baseline acc
    net = CNN_CIFAR10().to(device)
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))
    _, baseline_acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())

    # get random keep accuracy
    accs = np.zeros(len(keep_ratios))
    accs[-1] = baseline_acc
    for i, keep_ratio in enumerate(keep_ratios[:-1]):
        net.load_state_dict(torch.load(f"experiment7/random_keep/CNN_CIFAR10_rndkeep{keep_ratio}_1.path", map_location=device))
        _, acc = get_loss_acc(net, testloader)
        accs[i] = acc

    accs_all["random"] = accs

    MV = ["random"]
    MT = ["high", "mid"]
    for mv in MV:
        for mt in MT:
            name = mv + " " + mt
            accs = np.zeros(len(keep_ratios))
            accs[-1] = baseline_acc
            for i, keep_ratio in enumerate(keep_ratios[:-1]):
                load_path = f"experiment7/diff_ckpts/CNN_CIFAR10_{ckpts}_{mv}_{mt}_{keep_ratio}.pth"
                net.load_state_dict(torch.load(load_path, map_location=device))
                _, acc = get_loss_acc(net, testloader)
                accs[i] = acc
            accs_all[name] = accs

    fig = plt.figure(figsize=(10, 10))
    for name in accs_all.keys():
        plt.plot(keep_ratios, accs_all[name], ".-", label=name)
        print(name)
        print(accs_all[name])
    plt.title("ckpts from {}".format(ckpts))
    plt.legend()
    plt.show()


def evaluation_ckpts2(settings=["1_2_3",
                               "5_15_25",
                               "10_16_20",
                               "16_20_22",
                               "20_40_60",
                               "20_43_45"]):
    keep_ratios = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    accs_all = {}
    trainloader, valloader, testloader = prepare_CIFAR10()

    # get baseline acc
    net = CNN_CIFAR10().to(device)
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))
    _, baseline_acc = get_loss_acc(net, testloader)

    # get random keep accuracy
    accs = np.zeros(len(keep_ratios))
    accs[-1] = baseline_acc
    for i, keep_ratio in enumerate(keep_ratios[:-1]):
        net.load_state_dict(torch.load(f"experiment7/random_keep/CNN_CIFAR10_rndkeep{keep_ratio}_1.path", map_location=device))
        _, acc = get_loss_acc(net, testloader)
        accs[i] = acc
    accs_all["random"] = accs

    MT = ["high", "mid"]
    for mt in MT:
        for setting in settings:
            accs = np.zeros(len(keep_ratios))
            accs[-1] = baseline_acc
            for i, keep_ratio in enumerate(keep_ratios[:-1]):
                load_path = f"experiment7/diff_ckpts/CNN_CIFAR10_{setting}_random_{mt}_{keep_ratio}.pth"
                net.load_state_dict(torch.load(load_path, map_location=device))
                _, acc = get_loss_acc(net, testloader)
                accs[i] = acc
            accs_all[setting] = accs
        fig = plt.figure(figsize=(10, 10))
        for name in accs_all.keys():
            plt.plot(keep_ratios, accs_all[name], ".-", label=name)
        plt.title(f"Method for SV: {mt}")
        plt.legend()
        plt.show()


def experiment_strategy():
    """
    different strategies based on different keep ratios
    TracIn scores are calculated using ckpt 20, 40 and 60
    """
    experiment_path = "experiment7/strategy"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    keep_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]
    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10().to(device)

    for keep_ratio in keep_ratios:
        # select validation and train
        selected_val = np.load(f"experiment7/SV_random.npz", allow_pickle=True)
        X_val = selected_val["X"]
        Y_val = selected_val["Y"]
        selected_val = {}
        selected_val["X"] = torch.Tensor(X_val)
        selected_val["Y"] = torch.LongTensor(Y_val)

        # select train set using different strategy
        TracIn_by_cls = np.load(f"experiment7/TracIn_random_20_40_60.npz",
                                allow_pickle=True)["TracIn_by_cls"].item()
        X_train = trainloader.dataset.Data
        Y_train = trainloader.dataset.Label
        n_keep_per_cls = int(len(X_train) / 10 * keep_ratio)

        X_selected = []
        Y_selected = []

        if keep_ratio in [0.1, 0.2]:
            for cls in range(10):
                TracIn_mean = np.mean(TracIn_by_cls[cls], axis=1)
                half_drop = int((len(TracIn_mean) - n_keep_per_cls) // 2)
                index = list(np.argsort(TracIn_mean)[half_drop:-half_drop])
                X_selected.append(X_train[Y_train == cls][index])
                Y_selected.append(Y_train[Y_train == cls][index])
        elif keep_ratio == 0.4:
            for cls in range(10):
                TracIn_mean = np.mean(TracIn_by_cls[cls], axis=1)
                lower = np.quantile(TracIn_mean, q=0.4)
                upper = np.quantile(TracIn_mean, q=0.8)
                index = (TracIn_mean >= lower) & (TracIn_mean < upper)
                print(np.sum(index), "chosen")
                X_selected.append(X_train[Y_train == cls][index])
                Y_selected.append(Y_train[Y_train == cls][index])
        elif keep_ratio in [0.6, 0.8]:
            for cls in range(10):
                TracIn_mean = np.mean(TracIn_by_cls[cls], axis=1)
                index = list(np.argsort(TracIn_mean)[-n_keep_per_cls:])
                X_selected.append(X_train[Y_train == cls][index])
                Y_selected.append(Y_train[Y_train == cls][index])

        X_selected = np.concatenate(X_selected, axis=0)
        Y_selected = np.concatenate(Y_selected, axis=0)

        # create a new trainloader
        selected_trainset = mydataset(X_selected, Y_selected)
        selected_trainloader = DataLoader(selected_trainset,
                                          batch_size=256,
                                          shuffle=True)

        # train model and save checkpoints
        save_path = experiment_path + f"/CNN_CIFAR10_strategy_{keep_ratio}.pth"
        train_CNN_CIFAR10(70, selected_trainloader, valloader,
                          save_path=save_path)


def evaluation_strategy():
    keep_ratios = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    accs_all = {}
    trainloader, valloader, testloader = prepare_CIFAR10()

    # get baseline acc
    net = CNN_CIFAR10().to(device)
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))
    _, baseline_acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())

    # get random keep accuracy
    accs = np.zeros(len(keep_ratios))
    accs[-1] = baseline_acc
    for i, keep_ratio in enumerate(keep_ratios[:-1]):
        net.load_state_dict(
            torch.load(f"experiment7/random_keep/CNN_CIFAR10_rndkeep{keep_ratio}_1.path", map_location=device))
        _, acc = get_loss_acc(net, testloader)
        accs[i] = acc
    accs_all["random"] = accs

    MV = ["random"]
    MT = ["high", "mid", "strategy"]
    for mv in MV:
        for mt in MT:
            name = mv + " " + mt
            accs = np.zeros(len(keep_ratios))
            accs[-1] = baseline_acc
            for i, keep_ratio in enumerate(keep_ratios[:-1]):
                if mt != "strategy":
                    net.load_state_dict(torch.load(f"experiment7/from_beginning(loss)/{mv}{mt}{keep_ratio}/CNN_CIFAR10.pth",
                                                   map_location=device))

                elif mt == "strategy":
                    net.load_state_dict(torch.load(f"experiment7/strategy/CNN_CIFAR10_strategy_{keep_ratio}.pth",
                                                   map_location=device))
                _, acc = get_loss_acc(net, testloader)
                accs[i] = acc
            accs_all[name] = accs

    fig = plt.figure(figsize=(10, 10))
    for name in accs_all.keys():
        plt.plot(keep_ratios, accs_all[name], ".-", label=name)
        print(name)
        print(accs_all[name])
    plt.legend()
    plt.show()


def experiment_TracIn_all():
    """
    select train samples not by class
    """
    experiment_path = "experiment7/TracIn_all"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    keep_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]
    trainloader, valloader, testloader = prepare_CIFAR10()

    for keep_ratio in keep_ratios:
        # select validation
        selected_val = np.load(f"experiment7/SV_random.npz", allow_pickle=True)
        X_val = selected_val["X"]
        Y_val = selected_val["Y"]
        selected_val = {}
        selected_val["X"] = torch.Tensor(X_val)
        selected_val["Y"] = torch.LongTensor(Y_val)

        # select train samples base on TracIn score
        TracIn_all = np.load(f"experiment7/TracIn_random_20_40_60_all.npz", allow_pickle=True)["TracIn_all"]
        X_train = trainloader.dataset.Data
        Y_train = trainloader.dataset.Label
        n_train = len(Y_train)
        n_val = len(Y_val)
        n_keep = int(n_train * keep_ratio)

        selected_index = []
        TracIn_all_mean = np.mean(TracIn_all, axis=1)
        selected_index = np.argsort(TracIn_all_mean)[-n_keep:]

        print(len(selected_index), keep_ratio)
        X_selected = X_train[selected_index]
        Y_selected = Y_train[selected_index]
        # plt.hist(Y_selected)
        # plt.title("Keep Ratio is {}".format(keep_ratio))
        # plt.show()

        # create a new trainloader
        selected_trainset = mydataset(X_selected, Y_selected)
        selected_trainloader = DataLoader(selected_trainset,
                                          batch_size=256,
                                          shuffle=True)

        # train model and save checkpoints
        save_path = experiment_path + f"/CNN_CIFAR10_TracIN_all_{keep_ratio}.pth"
        train_CNN_CIFAR10(70, selected_trainloader, valloader,
                          save_path=save_path)


def save_loss_reduction_ratio():
    trainloader, valloader, testloader = prepare_CIFAR10()
    loss_epoch = []
    net = CNN_CIFAR10().to(device)
    for i in range(1, 71):
        net.load_state_dict(torch.load(f"model_weights/weights_70epochs/CNN_CIFAR10_epoch{i}.pth",
                                       map_location=device))
        loss, acc = get_loss_acc(net, trainloader)
        loss_epoch.append(loss)

    np.save("model_weights/weights_70epochs/loss_by_epoch.npy", loss_epoch)
    loss_reduction_ratio = (-np.diff(loss_epoch) / loss_epoch[:-1])
    np.save("model_weights/weights_70epochs/loss_reduction_ratio_by_epoch.npy", loss_reduction_ratio)
    plt.plot(np.arange(1, 71), loss_epoch)
    plt.show()
    print(np.argsort(loss_reduction_ratio)[::-1])


def eyeball_highTracIn(mode="by class"):
    trainloader, valloader, testloader = prepare_CIFAR10()
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label

    if mode == "by class":
        TracIn_by_cls = np.load("experiment7/TracIn_random_20_40_60.npz", allow_pickle=True)["TracIn_by_cls"].item()
        fig, ax = plt.subplots(10, 10, figsize=(20, 20))
        for cls in range(10):
            TracIn_cls = TracIn_by_cls[cls]
            X_train_cls = X_train[Y_train == cls]
            plot_index = np.argsort(np.mean(TracIn_cls, axis=1))[:10]
            for i in range(10):
                image = np.transpose(X_train_cls[plot_index[i]], (1, 2, 0))
                ax[cls, i].imshow(image / 255)

        plt.show()
    elif mode == "all":
        TracIn_all = np.load("experiment7/TracIn_random_20_40_60_all.npz", allow_pickle=True)["TracIn_all"]
        selected_val = np.load("experiment7/SV_random.npz", allow_pickle=True)
        Y_val = selected_val["Y"]
        print(Y_val)
        print(TracIn_all.shape)
        fig, ax = plt.subplots(10, 10, figsize=(20, 20))
        for cls in range(10):
            TracIn_cls = TracIn_all[:, (Y_val == cls)]
            plot_index = np.argsort(np.mean(TracIn_cls, axis=1))[::-1][:10]
            for i in range(10):
                if i == 0:
                    ax[cls, i].set_ylabel(LABEL_DECODER[cls], rotation=90)
                image = np.transpose(X_train[plot_index[i]], (1, 2, 0)) / 255
                ax[cls, i].imshow(image)
                ax[cls, i].title.set_text(LABEL_DECODER[Y_train[plot_index[i]]])
                ax[cls, i].get_xaxis().set_visible(False)
                if i != 0:
                    ax[cls, i].get_yaxis().set_visible(False)
        plt.legend()
        plt.show()


def poison_model(ckpts="1_2_3"):
    trainloader, valloader, testloader = prepare_CIFAR10()
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    TracIn_by_cls = np.load(f"experiment7/TracIn_random_{ckpts}.npz", allow_pickle=True)["TracIn_by_cls"].item()
    keep_ratios = [0.1, 0.2, 0.3, 0.4]

    for keep_ratio in keep_ratios:
        X_poison = []
        Y_poison = []
        n_keep = int(40000 * keep_ratio)
        n_keep_cls = n_keep // 10
        for cls in range(10):
            TracIn = TracIn_by_cls[cls]
            keep_index = list(np.argsort(np.mean(TracIn, axis=1))[::-1][:n_keep_cls])
            X_poison.append(X_train[Y_train == cls][keep_index])
            Y_poison.append(Y_train[Y_train == cls][keep_index])
        X_poison = np.concatenate(X_poison, axis=0)
        Y_poison = np.concatenate(Y_poison, axis=0)
        trainloader_poison = DataLoader(mydataset(X_poison, Y_poison),
                                        batch_size=256, shuffle=True)
        load_path = "model_weights/CNN_CIFAR10.pth"
        save_path = f"experiment7/model_poison/CNN_CIFAR10_{ckpts}_{keep_ratio}.pth"
        train_CNN_CIFAR10(70, trainloader_poison, valloader, load_path=load_path, save_path=save_path)


def evaluate_poison(ckpts="1_2_3"):
    net = CNN_CIFAR10().to(device)
    trainloader, valloader, testloader = prepare_CIFAR10()

    # get baseline acc
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))
    _, baseline_acc = get_loss_acc(net, testloader)

    # get dropped acc
    keep_ratios = [0.1, 0.2, 0.3, 0.4]
    accs = []
    for keep_ratio in keep_ratios:
        model_path = f"experiment7/model_poison/CNN_CIFAR10_{ckpts}_{keep_ratio}.pth"
        net.load_state_dict(torch.load(model_path, map_location=device))
        _, acc = get_loss_acc(net, testloader)
        accs.append(acc)

    plt.plot(keep_ratios, [baseline_acc for i in range(4)], "--", label="Baseline")
    plt.plot(keep_ratios, accs, ".-", label="Poisoned")
    plt.xlabel("keep ratio")
    plt.ylabel("test accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # net = CNN_CIFAR10().to(device)
    # trainloader, valloader, testloader = prepare_CIFAR10()
    # for i in range(1, 6):
    #     net.load_state_dict(torch.load(f"experiment7/random_keep/CNN_CIFAR10_rndkeep0.1_1.path", map_location=device))
    #     accs = get_acc_by_cls(net, testloader)
    #     print(accs)

    experiment_random_keep()