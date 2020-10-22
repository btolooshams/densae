"""
Copyright (c) 2020 Bahareh Tolooshams

train

:author: Bahareh Tolooshams
"""


import torch
import torch.optim as optim
import torchvision
import numpy as np
import pickle
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
from sacred import Experiment

from sacred import SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


import sys

sys.path.append("src/")

import model, generator, trainer, utils

from conf import config_ingredient

import warnings

warnings.filterwarnings("ignore")

ex = Experiment("train", ingredients=[config_ingredient])

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


@ex.automain
def run(cfg):

    hyp = cfg["hyp"]

    print(hyp)

    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    PATH = "../results/{}/{}".format(hyp["experiment_name"], random_date)
    os.makedirs(PATH)

    filename = os.path.join(PATH, "hyp.pickle")
    with open(filename, "wb") as file:
        pickle.dump(hyp, file)

    writer = SummaryWriter(os.path.join(PATH))

    print("load data.")
    if hyp["dataset"] == "folder":
        train_loader = generator.get_train_path_loader(
            hyp["batch_size"], hyp["train_path"], crop_dim=hyp["crop_dim"], shuffle=True
        )
        test_loader = generator.get_test_path_loader(1, hyp["test_path"], shuffle=False)
    elif hyp["dataset"] == "mnist":
        train_loader, val_loader, test_loader = generator.get_MNIST_loaders(
            hyp["batch_size"], hyp["shuffle"]
        )
    else:
        print("dataset is not implemented!")

    # disjoint classification training
    if hyp["classification"]:
        net = torch.load(hyp["model_path"], map_location=hyp["device"])
        classifier = model.Classifier(hyp)

        enc_tr_loader, enc_val_loader, enc_te_loader = generator.get_encoding_loaders(
            train_loader, val_loader, test_loader, net, hyp
        )

        optimizer = optim.Adam(
            classifier.parameters(),
            lr=hyp["lr"],
            eps=hyp["eps"],
            weight_decay=hyp["weight_decay"],
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=hyp["lr_step"], gamma=hyp["lr_decay"]
        )

        criterion_class = torch.nn.CrossEntropyLoss()

        print("train classifier.")
        acc = trainer.train_classifier(
            classifier,
            enc_tr_loader,
            hyp,
            criterion_class,
            optimizer,
            scheduler,
            writer,
            PATH,
            enc_val_loader,
            enc_te_loader,
        )
    else:
        print("create model.")
        if hyp["warm_start"]:
            net = torch.load(hyp["model_path"], map_location=hyp["device"])
            net.device = hyp["device"]
        else:
            if hyp["network"] == "CSCNetTiedHyp":
                net = model.CSCNetTiedHyp(hyp)
            elif hyp["network"] == "CSCNetTiedLS":
                net = model.CSCNetTiedLS(hyp)
            elif hyp["network"] == "DenSaE":
                net = model.DenSaE(hyp)
            else:
                print("model does not exist!")

        torch.save(net, os.path.join(PATH, "model_init.pt"))

        if hyp["loss"] == "l2":
            criterion = utils.L2loss()
        else:
            print("loss is not implemented!")

        if hyp["train_joint_ae_class"]:
            classifier = model.Classifier(hyp)
            criterion_class = torch.nn.CrossEntropyLoss()

            params = []
            for param in net.parameters():
                params.append(param)
            for param in classifier.parameters():
                params.append(param)

            optimizer = optim.Adam(
                params, lr=hyp["lr"], eps=hyp["eps"], weight_decay=hyp["weight_decay"]
            )
        else:
            optimizer = optim.Adam(
                net.parameters(),
                lr=hyp["lr"],
                eps=hyp["eps"],
                weight_decay=hyp["weight_decay"],
            )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=hyp["lr_step"], gamma=hyp["lr_decay"]
        )

        print("train auto-encoder.")
        if hyp["dataset"] == "mnist":
            if hyp["train_joint_ae_class"]:
                net = trainer.train_join_ae_class_mnist(
                    net,
                    classifier,
                    train_loader,
                    hyp,
                    criterion,
                    criterion_class,
                    optimizer,
                    scheduler,
                    writer,
                    PATH,
                    val_loader,
                    test_loader,
                )
            else:
                net = trainer.train_ae_mnist(
                    net,
                    train_loader,
                    hyp,
                    criterion,
                    optimizer,
                    scheduler,
                    writer,
                    PATH,
                    val_loader,
                    test_loader,
                )
        else:
            net = trainer.train_ae(
                net,
                train_loader,
                hyp,
                criterion,
                optimizer,
                scheduler,
                writer,
                PATH,
                test_loader,
            )
