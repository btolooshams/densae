"""
Copyright (c) 2020 Bahareh Tolooshams

configurations

:author: Bahareh Tolooshams
"""

import torch

from sacred import Experiment, Ingredient

config_ingredient = Ingredient("cfg")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


###############################
########### default ###########
###############################
@config_ingredient.config
def cfg():
    hyp = {
        "experiment_name": "default",
        "dataset": "folder",
        "network": "DenSaE",
        "noiseSTD": 25,
        "dictionary_dim": 7,
        "stride": 5,
        "num_conv": 64,
        "num_conv_A": 32,
        "num_conv_B": 32,
        "L": 10,
        "num_iters": 15,
        "twosided": True,
        "batch_size": 1,
        "num_epochs": 250,
        "normalize": False,
        "lr": 1e-4,
        "lr_decay": 0.80,
        "lr_step": 50,
        "info_period": 10000,
        "crop_dim": (128, 128),
        "lam": 0.15,
        "rho": 0.2,
        "weight_decay": 0,
        "supervised": True,
        "shuffle": True,
        "denoising": True,
        "loss": "l2",
        "b": 0.01,
        "train_path": "../data/CBSD432/",
        "test_path": "../data/BSD68/",
        "device": device,
    }


###############################
####### cscnet_tied_ls ########
###############################


@config_ingredient.named_config
def noise15_cscnet_tied_ls():
    hyp = {
        "experiment_name": "noise15_cscnet_tied_ls",
        "network": "CSCNetTiedLS",
        "num_conv": 64,
        "noiseSTD": 15,
    }


@config_ingredient.named_config
def noise25_cscnet_tied_ls():
    hyp = {
        "experiment_name": "noise25_cscnet_tied_ls",
        "network": "CSCNetTiedLS",
        "num_conv": 64,
        "noiseSTD": 25,
    }


@config_ingredient.named_config
def noise50_cscnet_tied_ls():
    hyp = {
        "experiment_name": "noise50_cscnet_tied_ls",
        "network": "CSCNetTiedLS",
        "num_conv": 64,
        "noiseSTD": 50,
    }


@config_ingredient.named_config
def noise75_cscnet_tied_ls():
    hyp = {
        "experiment_name": "noise75_cscnet_tied_ls",
        "network": "CSCNetTiedLS",
        "num_conv": 64,
        "noiseSTD": 75,
    }


###############################
####### cscnet_tied_hyp #######
###############################
@config_ingredient.named_config
def noise15_cscnet_tied_hyp():
    hyp = {
        "experiment_name": "noise15_cscnet_tied_hyp",
        "network": "CSCNetTiedHyp",
        "num_conv": 64,
        "lam": 0.085,
        "noiseSTD": 15,
    }


@config_ingredient.named_config
def noise25_cscnet_tied_hyp():
    hyp = {
        "experiment_name": "noise25_cscnet_tied_hyp",
        "network": "CSCNetTiedHyp",
        "num_conv": 64,
        "lam": 0.16,
        "noiseSTD": 25,
    }


@config_ingredient.named_config
def noise50_cscnet_tied_hyp():
    hyp = {
        "experiment_name": "noise50_cscnet_tied_hyp",
        "network": "CSCNetTiedHyp",
        "num_conv": 64,
        "lam": 0.36,
        "noiseSTD": 50,
    }


@config_ingredient.named_config
def noise75_cscnet_tied_hyp():
    hyp = {
        "experiment_name": "noise75_cscnet_tied_hyp",
        "network": "CSCNetTiedHyp",
        "num_conv": 64,
        "lam": 0.56,
        "noiseSTD": 75,
    }


###############################
####### densae tied hyp #######
###############################
# 1A 63B
@config_ingredient.named_config
def noise15_densae_1A_63B_hyp():
    hyp = {
        "experiment_name": "noise15_densae_1A_63B_hyp",
        "network": "DenSaE",
        "num_conv_A": 1,
        "num_conv_B": 63,
        "lam": 0.085,
        "noiseSTD": 15,
    }


@config_ingredient.named_config
def noise25_densae_1A_63B_hyp():
    hyp = {
        "experiment_name": "noise25_densae_1A_63B_hyp",
        "network": "DenSaE",
        "num_conv_A": 1,
        "num_conv_B": 63,
        "lam": 0.16,
        "noiseSTD": 25,
    }


@config_ingredient.named_config
def noise50_densae_1A_63B_hyp():
    hyp = {
        "experiment_name": "noise50_densae_1A_63B_hyp",
        "network": "DenSaE",
        "num_conv_A": 1,
        "num_conv_B": 63,
        "lam": 0.36,
        "noiseSTD": 50,
    }


@config_ingredient.named_config
def noise75_densae_1A_63B_hyp():
    hyp = {
        "experiment_name": "noise75_densae_1A_63B_hyp",
        "network": "DenSaE",
        "num_conv_A": 1,
        "num_conv_B": 63,
        "lam": 0.56,
        "noiseSTD": 75,
    }


# 4A 60B
@config_ingredient.named_config
def noise15_densae_4A_60B_hyp():
    hyp = {
        "experiment_name": "noise15_densae_4A_60B_hyp",
        "network": "DenSaE",
        "num_conv_A": 4,
        "num_conv_B": 60,
        "lam": 0.085,
        "noiseSTD": 15,
    }


@config_ingredient.named_config
def noise25_densae_4A_60B_hyp():
    hyp = {
        "experiment_name": "noise25_densae_4A_60B_hyp",
        "network": "DenSaE",
        "num_conv_A": 4,
        "num_conv_B": 60,
        "lam": 0.16,
        "noiseSTD": 25,
    }


@config_ingredient.named_config
def noise50_densae_4A_60B_hyp():
    hyp = {
        "experiment_name": "noise50_densae_4A_60B_hyp",
        "network": "DenSaE",
        "num_conv_A": 4,
        "num_conv_B": 60,
        "lam": 0.36,
        "noiseSTD": 50,
    }


@config_ingredient.named_config
def noise75_densae_4A_60B_hyp():
    hyp = {
        "experiment_name": "noise75_densae_4A_60B_hyp",
        "network": "DenSaE",
        "num_conv_A": 4,
        "num_conv_B": 60,
        "lam": 0.56,
        "noiseSTD": 75,
    }


# 8A 56B
@config_ingredient.named_config
def noise15_densae_8A_56B_hyp():
    hyp = {
        "experiment_name": "noise15_densae_8A_56B_hyp",
        "network": "DenSaE",
        "num_conv_A": 8,
        "num_conv_B": 56,
        "lam": 0.085,
        "noiseSTD": 15,
    }


@config_ingredient.named_config
def noise25_densae_8A_56B_hyp():
    hyp = {
        "experiment_name": "noise25_densae_8A_56B_hyp",
        "network": "DenSaE",
        "num_conv_A": 8,
        "num_conv_B": 56,
        "lam": 0.16,
        "noiseSTD": 25,
    }


@config_ingredient.named_config
def noise50_densae_8A_56B_hyp():
    hyp = {
        "experiment_name": "noise50_densae_8A_56B_hyp",
        "network": "DenSaE",
        "num_conv_A": 8,
        "num_conv_B": 56,
        "lam": 0.36,
        "noiseSTD": 50,
    }


@config_ingredient.named_config
def noise75_densae_8A_56B_hyp():
    hyp = {
        "experiment_name": "noise75_densae_8A_56B_hyp",
        "network": "DenSaE",
        "num_conv_A": 8,
        "num_conv_B": 56,
        "lam": 0.56,
        "noiseSTD": 75,
    }


# 16A 48B
@config_ingredient.named_config
def noise15_densae_16A_48B_hyp():
    hyp = {
        "experiment_name": "noise15_densae_16A_48B_hyp",
        "network": "DenSaE",
        "num_conv_A": 16,
        "num_conv_B": 48,
        "lam": 0.085,
        "noiseSTD": 15,
    }


@config_ingredient.named_config
def noise25_densae_16A_48B_hyp():
    hyp = {
        "experiment_name": "noise25_densae_16A_48B_hyp",
        "network": "DenSaE",
        "num_conv_A": 16,
        "num_conv_B": 48,
        "lam": 0.16,
        "noiseSTD": 25,
    }


@config_ingredient.named_config
def noise50_densae_16A_48B_hyp():
    hyp = {
        "experiment_name": "noise50_densae_16A_48B_hyp",
        "network": "DenSaE",
        "num_conv_A": 16,
        "num_conv_B": 48,
        "lam": 0.36,
        "noiseSTD": 50,
    }


@config_ingredient.named_config
def noise75_densae_16A_48B_hyp():
    hyp = {
        "experiment_name": "noise75_densae_16A_48B_hyp",
        "network": "DenSaE",
        "num_conv_A": 16,
        "num_conv_B": 48,
        "lam": 0.56,
        "noiseSTD": 75,
    }


# 32A 32B
@config_ingredient.named_config
def noise15_densae_32A_32B_hyp():
    hyp = {
        "experiment_name": "noise15_densae_32A_32B_hyp",
        "network": "DenSaE",
        "num_conv_A": 32,
        "num_conv_B": 32,
        "lam": 0.085,
        "noiseSTD": 15,
    }


@config_ingredient.named_config
def noise25_densae_32A_32B_hyp():
    hyp = {
        "experiment_name": "noise25_densae_32A_32B_hyp",
        "network": "DenSaE",
        "num_conv_A": 32,
        "num_conv_B": 32,
        "lam": 0.16,
        "noiseSTD": 25,
    }


@config_ingredient.named_config
def noise50_densae_32A_32B_hyp():
    hyp = {
        "experiment_name": "noise50_densae_32A_32B_hyp",
        "network": "DenSaE",
        "num_conv_A": 32,
        "num_conv_B": 32,
        "lam": 0.36,
        "noiseSTD": 50,
    }


@config_ingredient.named_config
def noise75_densae_32A_32B_hyp():
    hyp = {
        "experiment_name": "noise75_densae_32A_32B_hyp",
        "network": "DenSaE",
        "num_conv_A": 32,
        "num_conv_B": 32,
        "lam": 0.56,
        "noiseSTD": 75,
    }
