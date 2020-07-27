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
        test_loader = generator.get_path_loader(1, hyp["test_path"], shuffle=False)
    else:
        print("dataset is not implemented!")

    print("create model.")
    #### Bu
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

    optimizer = optim.Adam(
        net.parameters(), lr=hyp["lr"], eps=1e-3, weight_decay=hyp["weight_decay"]
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=hyp["lr_step"], gamma=hyp["lr_decay"]
    )

    print("train auto-encoder.")
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
