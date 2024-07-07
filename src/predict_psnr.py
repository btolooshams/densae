"""
Copyright (c) 2020 Bahareh Tolooshams

predict

:author: Bahareh Tolooshams
"""

import torch
import numpy as np
import os

from sacred import Experiment

import sys

sys.path.append("src/")

import model, generator, utils
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

ex = Experiment("predict")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


@ex.automain
def predict():

    test_path = "../data/BSD68/"
    data_path = "../results/trained_models/denoising"

    experiment_name_list = os.listdir(data_path)
    experiment_name_list = [x for x in experiment_name_list if f".pt" in x]

    experiment_name_list = [x for x in experiment_name_list if f"1A63B" in x]

    print(experiment_name_list)

    for experiment_name in experiment_name_list:

    
        print("experiment_name", experiment_name)
        noiseSTD = int(experiment_name.split("_")[-1][:-3])
    
        # create model
        net = torch.load(os.path.join(data_path, experiment_name), map_location=device,
        )

        net.device = device
        net.eval()

        # load data
        test_loader = generator.get_test_path_loader(1, test_path, shuffle=False)

        psnr = list()
        with torch.no_grad():
            for idx, (img_org, _) in tqdm(
                enumerate(test_loader), disable=True
            ):
                img_noisy = (img_org + noiseSTD / 255 * torch.randn(img_org.shape)).to(
                    device
                )

                img_hat, r = net(img_noisy)

                img_hat = torch.clamp(img_hat, 0, 1)

                psnr_i = utils.PSNR(
                    img_org[0, 0, :, :].clone().detach().cpu().numpy(),
                    img_hat[0, 0, :, :].clone().detach().cpu().numpy(),
                )

                psnr.append(psnr_i)
            
        print("PSNR:", "mean", np.mean(np.array(psnr)), "std", np.std(np.array(psnr)))
