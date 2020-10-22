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
        "strideA": 5,
        "strideB": 5,
        "split_stride": 5,
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
        "model_period": 25,
        "loss_period": 25,
        "crop_dim": (128, 128),
        "lam": 0.15,
        "rho": 1e10,
        "weight_decay": 0,
        "supervised": True,
        "shuffle": True,
        "denoising": True,
        "loss": "l2",
        "b": 0.01,
        "eps": 1e-3,
        "D_input": 400,
        "D_output": 10,
        "classification": False,
        "classify_use_only_u": False,
        "random_remove_u": 0,
        "beta": 0,
        "train_joint_ae_class": False,
        "warm_start": False,
        "dense": False,
        "reshape": (10, 10),
        "train_path": "../data/CBSD432/",
        "test_path": "../data/BSD68/",
        "model_path": "...pt",
        "device": device,
    }


##########################################################
##########################################################
#################### denoising - BSD68 ###################
##########################################################
##########################################################

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
####### densae_tied_hyp #######
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


##########################################################
##########################################################
################# classification - mnist #################
##########################################################
##########################################################

###############################
####### cscnet_tied_ls ########
###############################


@config_ingredient.named_config
def mnist_cscnet_400units_numiter15():
    hyp = {
        "experiment_name": "mnist/cscnet_400units_numiter15",
        "dense": True,
        "dataset": "mnist",
        "network": "CSCNetTiedLS",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "reshape": (20, 20),
        "num_iters": 15,
        "stride": 1,
        "split_stride": 1,
        "L": 50,
        "b": 0.00,
        "twosided": False,
        "normalize": True,
        "batch_size": 16,
        "num_epochs": 150,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 10,
        "model_period": 10,
        "loss_period": 10,
    }


@config_ingredient.named_config
def mnist_classification_cscnet_400units_numiter15():
    hyp = {
        "classification": True,
        "network": "CSCNetTiedLS",
        "model_path": "../results/trained_models/classification/model_cscnet_tied_ls_disjoint_ae.pt",
        "experiment_name": "mnist/disjoint/cscnet_400units_numiter15",
        "dataset": "mnist",
        "D_input": 400,
        "D_output": 10,
        "batch_size": 16,
        "num_epochs": 1000,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 1000,
        "eps": 1e-15,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_cscnet_400units_numiter15_joint_betap5():
    hyp = {
        "experiment_name": "mnist/joint/cscnet_400units_numiter15_betap5",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_cscnet_tied_ls_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "CSCNetTiedLS",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "D_input": 400,
        "D_output": 10,
        "reshape": (20, 20),
        "num_iters": 15,
        "beta": 0.5,
        "stride": 1,
        "split_stride": 1,
        "L": 50,
        "b": 0.00,
        "twosided": False,
        "normalize": True,
        "batch_size": 16,
        "num_epochs": 500,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_cscnet_400units_numiter15_joint_betap75():
    hyp = {
        "experiment_name": "mnist/joint/cscnet_400units_numiter15_betap75",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_cscnet_tied_ls_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "CSCNetTiedLS",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "D_input": 400,
        "D_output": 10,
        "reshape": (20, 20),
        "num_iters": 15,
        "beta": 0.75,
        "stride": 1,
        "split_stride": 1,
        "L": 50,
        "b": 0.00,
        "twosided": False,
        "normalize": True,
        "batch_size": 16,
        "num_epochs": 500,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_cscnet_400units_numiter15_joint_betap95():
    hyp = {
        "experiment_name": "mnist/joint/cscnet_400units_numiter15_betap95",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_cscnet_tied_ls_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "CSCNetTiedLS",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "D_input": 400,
        "D_output": 10,
        "reshape": (20, 20),
        "num_iters": 15,
        "beta": 0.95,
        "stride": 1,
        "split_stride": 1,
        "L": 50,
        "b": 0.00,
        "twosided": False,
        "normalize": True,
        "batch_size": 16,
        "num_epochs": 500,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_cscnet_400units_numiter15_joint_beta1():
    hyp = {
        "experiment_name": "mnist/joint/cscnet_400units_numiter15_beta1",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_cscnet_tied_ls_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "CSCNetTiedLS",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "D_input": 400,
        "D_output": 10,
        "reshape": (20, 20),
        "num_iters": 15,
        "beta": 1,
        "stride": 1,
        "split_stride": 1,
        "L": 50,
        "b": 0.00,
        "twosided": False,
        "normalize": True,
        "batch_size": 16,
        "num_epochs": 500,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


###############################
####### cscnet_tied_hyp #######
###############################


@config_ingredient.named_config
def mnist_cscnet_400units_numiter15_hyp_bp01():
    hyp = {
        "experiment_name": "mnist/cscnet_400units_numiter15_hyp_bp01",
        "dense": True,
        "dataset": "mnist",
        "network": "CSCNetTiedHyp",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "reshape": (20, 20),
        "num_iters": 15,
        "stride": 1,
        "split_stride": 1,
        "L": 50,
        "b": 0.01,
        "twosided": False,
        "normalize": True,
        "batch_size": 16,
        "num_epochs": 150,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 10,
        "model_period": 10,
        "loss_period": 10,
    }


@config_ingredient.named_config
def mnist_classification_cscnet_400units_numiter15_hyp_bp01():
    hyp = {
        "classification": True,
        "network": "CSCNetTiedHyp",
        "model_path": "../results/trained_models/classification/model_cscnet_tied_hyp_disjoint_ae.pt",
        "experiment_name": "mnist/disjoint/cscnet_400units_numiter15_hyp_bp01",
        "dataset": "mnist",
        "D_input": 400,
        "D_output": 10,
        "batch_size": 16,
        "num_epochs": 1000,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 1000,
        "eps": 1e-15,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_cscnet_400units_numiter15_hyp_bp01_joint_betap5():
    hyp = {
        "experiment_name": "mnist/joint/cscnet_400units_numiter15_hyp_bp01_betap5",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_cscnet_tied_hyp_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "CSCNetTiedHyp",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "D_input": 400,
        "D_output": 10,
        "reshape": (20, 20),
        "num_iters": 15,
        "beta": 0.5,
        "stride": 1,
        "split_stride": 1,
        "L": 50,
        "b": 0.01,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_cscnet_400units_numiter15_hyp_bp01_joint_betap75():
    hyp = {
        "experiment_name": "mnist/joint/cscnet_400units_numiter15_hyp_bp01_betap75",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_cscnet_tied_hyp_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "CSCNetTiedHyp",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "D_input": 400,
        "D_output": 10,
        "reshape": (20, 20),
        "num_iters": 15,
        "beta": 0.75,
        "stride": 1,
        "split_stride": 1,
        "L": 50,
        "b": 0.01,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_cscnet_400units_numiter15_hyp_bp01_joint_betap95():
    hyp = {
        "experiment_name": "mnist/joint/cscnet_400units_numiter15_hyp_bp01_betap95",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_cscnet_tied_hyp_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "CSCNetTiedHyp",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "D_input": 400,
        "D_output": 10,
        "reshape": (20, 20),
        "num_iters": 15,
        "beta": 0.95,
        "stride": 1,
        "split_stride": 1,
        "L": 50,
        "b": 0.01,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_cscnet_400units_numiter15_hyp_bp01_joint_beta1():
    hyp = {
        "experiment_name": "mnist/joint/cscnet_400units_numiter15_hyp_bp01_beta1",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_cscnet_tied_hyp_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "CSCNetTiedHyp",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "D_input": 400,
        "D_output": 10,
        "reshape": (20, 20),
        "num_iters": 15,
        "beta": 1,
        "stride": 1,
        "split_stride": 1,
        "L": 50,
        "b": 0.01,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


###############################
####### densae_tied_hyp #######
###############################
# 5A 395B
@config_ingredient.named_config
def mnist_densae_400units_5A395B_numiter15_hyp_bp01():
    hyp = {
        "experiment_name": "mnist/densae_400units_5A395B_numiter15_hyp_bp01",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "num_conv_A": 5,
        "num_conv_B": 395,
        "reshape": ((1, 5), (5, 79)),
        "num_iters": 15,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": True,
        "batch_size": 16,
        "num_epochs": 150,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 10,
        "model_period": 10,
        "loss_period": 10,
    }


@config_ingredient.named_config
def mnist_classification_densae_400units_5A395B_numiter15_hyp_bp01():
    hyp = {
        "classification": True,
        "classify_use_only_u": False,
        "random_remove_u": 0,
        "network": "DenSaE",
        "model_path": "../results/trained_models/classification/model_densae_5A395B_disjoint_ae.pt",
        "experiment_name": "mnist/disjoint/densae_400units_5A395B_numiter15_hyp_bp01",
        "dataset": "mnist",
        "D_input": 400,
        "D_output": 10,
        "batch_size": 16,
        "num_epochs": 1000,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 1000,
        "eps": 1e-15,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_densae_400units_5A395B_numiter15_hyp_bp01_joint_betap5():
    hyp = {
        "experiment_name": "mnist/joint/densae_400units_5A395B_numiter15_hyp_bp01_betap5",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_densae_5A395B_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv_A": 5,
        "num_conv_B": 395,
        "D_input": 400,
        "D_output": 10,
        "reshape": ((1, 5), (5, 79)),
        "num_iters": 15,
        "beta": 0.5,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_densae_400units_5A395B_numiter15_hyp_bp01_joint_betap75():
    hyp = {
        "experiment_name": "mnist/joint/densae_400units_5A395B_numiter15_hyp_bp01_betap75",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_densae_5A395B_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv_A": 5,
        "num_conv_B": 395,
        "D_input": 400,
        "D_output": 10,
        "reshape": ((1, 5), (5, 79)),
        "num_iters": 15,
        "beta": 0.75,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_densae_400units_5A395B_numiter15_hyp_bp01_joint_betap95():
    hyp = {
        "experiment_name": "mnist/joint/densae_400units_5A395B_numiter15_hyp_bp01_betap95",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_densae_5A395B_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv_A": 5,
        "num_conv_B": 395,
        "D_input": 400,
        "D_output": 10,
        "reshape": ((1, 5), (5, 79)),
        "num_iters": 15,
        "beta": 0.95,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_densae_400units_5A395B_numiter15_hyp_bp01_joint_beta1():
    hyp = {
        "experiment_name": "mnist/joint/densae_400units_5A395B_numiter15_hyp_bp01_beta1",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_densae_5A395B_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv_A": 5,
        "num_conv_B": 395,
        "D_input": 400,
        "D_output": 10,
        "reshape": ((1, 5), (5, 79)),
        "num_iters": 15,
        "beta": 1,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


# 25A 375B
@config_ingredient.named_config
def mnist_densae_400units_25A375B_numiter15_hyp_bp01():
    hyp = {
        "experiment_name": "mnist/densae_400units_25A375B_numiter15_hyp_bp01",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "num_conv_A": 25,
        "num_conv_B": 375,
        "reshape": ((5, 5), (15, 25)),
        "num_iters": 15,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": True,
        "batch_size": 16,
        "num_epochs": 150,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 10,
        "model_period": 10,
        "loss_period": 10,
    }


@config_ingredient.named_config
def mnist_classification_densae_400units_25A375B_numiter15_hyp_bp01():
    hyp = {
        "classification": True,
        "classify_use_only_u": False,
        "random_remove_u": 0,
        "network": "DenSaE",
        "model_path": "../results/trained_models/classification/model_densae_25A375B_disjoint_ae.pt",
        "experiment_name": "mnist/disjoint/densae_400units_25A375B_numiter15_hyp_bp01",
        "dataset": "mnist",
        "D_input": 400,
        "D_output": 10,
        "batch_size": 16,
        "num_epochs": 1000,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 1000,
        "eps": 1e-15,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_densae_400units_25A375B_numiter15_hyp_bp01_joint_betap5():
    hyp = {
        "experiment_name": "mnist/joint/densae_400units_25A375B_numiter15_hyp_bp01_betap5",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_densae_25A375B_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv_A": 25,
        "num_conv_B": 375,
        "D_input": 400,
        "D_output": 10,
        "reshape": ((5, 5), (15, 25)),
        "num_iters": 15,
        "beta": 0.5,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_densae_400units_25A375B_numiter15_hyp_bp01_joint_betap75():
    hyp = {
        "experiment_name": "mnist/joint/densae_400units_25A375B_numiter15_hyp_bp01_betap75",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_densae_25A375B_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv_A": 25,
        "num_conv_B": 375,
        "D_input": 400,
        "D_output": 10,
        "reshape": ((5, 5), (15, 25)),
        "num_iters": 15,
        "beta": 0.75,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_densae_400units_25A375B_numiter15_hyp_bp01_joint_betap95():
    hyp = {
        "experiment_name": "mnist/joint/densae_400units_25A375B_numiter15_hyp_bp01_betap95",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_densae_25A375B_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv_A": 25,
        "num_conv_B": 375,
        "D_input": 400,
        "D_output": 10,
        "reshape": ((5, 5), (15, 25)),
        "num_iters": 15,
        "beta": 0.95,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_densae_400units_25A375B_numiter15_hyp_bp01_joint_beta1():
    hyp = {
        "experiment_name": "mnist/joint/densae_400units_25A375B_numiter15_hyp_bp01_beta1",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_densae_25A375B_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv_A": 25,
        "num_conv_B": 375,
        "D_input": 400,
        "D_output": 10,
        "reshape": ((5, 5), (15, 25)),
        "num_iters": 15,
        "beta": 1,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


# 50A 350B
@config_ingredient.named_config
def mnist_densae_400units_50A350B_numiter15_hyp_bp01():
    hyp = {
        "experiment_name": "mnist/densae_400units_50A350B_numiter15_hyp_bp01",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "num_conv_A": 50,
        "num_conv_B": 350,
        "reshape": ((5, 10), (14, 25)),
        "num_iters": 15,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": True,
        "batch_size": 16,
        "num_epochs": 150,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 10,
        "model_period": 10,
        "loss_period": 10,
    }


@config_ingredient.named_config
def mnist_classification_densae_400units_50A350B_numiter15_hyp_bp01():
    hyp = {
        "classification": True,
        "classify_use_only_u": False,
        "random_remove_u": 0,
        "network": "DenSaE",
        "model_path": "../results/trained_models/classification/model_densae_50A350B_disjoint_ae.pt",
        "experiment_name": "mnist/disjoint/densae_400units_50A350B_numiter15_hyp_bp01",
        "dataset": "mnist",
        "D_input": 400,
        "D_output": 10,
        "batch_size": 16,
        "num_epochs": 1000,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 1000,
        "eps": 1e-15,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


# 100A 300B
@config_ingredient.named_config
def mnist_densae_400units_100A300B_numiter15_hyp_bp01():
    hyp = {
        "experiment_name": "mnist/densae_400units_100A300B_numiter15_hyp_bp01",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "num_conv_A": 100,
        "num_conv_B": 300,
        "reshape": ((10, 10), (15, 20)),
        "num_iters": 15,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": True,
        "batch_size": 16,
        "num_epochs": 150,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 10,
        "model_period": 10,
        "loss_period": 10,
    }


@config_ingredient.named_config
def mnist_classification_densae_400units_100A300B_numiter15_hyp_bp01():
    hyp = {
        "classification": True,
        "classify_use_only_u": False,
        "random_remove_u": 0,
        "network": "DenSaE",
        "model_path": "../results/trained_models/classification/model_densae_100A300B_disjoint_ae.pt",
        "experiment_name": "mnist/disjoint/densae_400units_100A300B_numiter15_hyp_bp01",
        "dataset": "mnist",
        "D_input": 400,
        "D_output": 10,
        "batch_size": 16,
        "num_epochs": 1000,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 1000,
        "eps": 1e-15,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


# 200A 200B
@config_ingredient.named_config
def mnist_densae_400units_200A200B_numiter15_hyp_bp01():
    hyp = {
        "experiment_name": "mnist/densae_400units_200A200B_numiter15_hyp_bp01",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv": 400,
        "num_conv_A": 200,
        "num_conv_B": 200,
        "reshape": ((10, 20), (10, 20)),
        "num_iters": 15,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": True,
        "batch_size": 16,
        "num_epochs": 150,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 10,
        "model_period": 10,
        "loss_period": 10,
    }


@config_ingredient.named_config
def mnist_classification_densae_400units_200A200B_numiter15_hyp_bp01():
    hyp = {
        "classification": True,
        "classify_use_only_u": False,
        "random_remove_u": 0,
        "network": "DenSaE",
        "model_path": "../results/trained_models/classification/model_densae_200A200B_disjoint_ae.pt",
        "experiment_name": "mnist/disjoint/densae_400units_200A200B_numiter15_hyp_bp01",
        "dataset": "mnist",
        "D_input": 400,
        "D_output": 10,
        "batch_size": 16,
        "num_epochs": 1000,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 1000,
        "eps": 1e-15,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_densae_400units_200A200B_numiter15_hyp_bp01_joint_betap5():
    hyp = {
        "experiment_name": "mnist/joint/densae_400units_200A200B_numiter15_hyp_bp01_betap5",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_densae_200A200B_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv_A": 200,
        "num_conv_B": 200,
        "D_input": 400,
        "D_output": 10,
        "reshape": ((10, 20), (10, 20)),
        "num_iters": 15,
        "beta": 0.5,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_densae_400units_200A200B_numiter15_hyp_bp01_joint_betap75():
    hyp = {
        "experiment_name": "mnist/joint/densae_400units_200A200B_numiter15_hyp_bp01_betap75",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_densae_200A200B_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv_A": 200,
        "num_conv_B": 200,
        "D_input": 400,
        "D_output": 10,
        "reshape": ((10, 20), (10, 20)),
        "num_iters": 15,
        "beta": 0.75,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_densae_400units_200A200B_numiter15_hyp_bp01_joint_betap95():
    hyp = {
        "experiment_name": "mnist/joint/densae_400units_200A200B_numiter15_hyp_bp01_betap95",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_densae_200A200B_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv_A": 200,
        "num_conv_B": 200,
        "D_input": 400,
        "D_output": 10,
        "reshape": ((10, 20), (10, 20)),
        "num_iters": 15,
        "beta": 0.95,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }


@config_ingredient.named_config
def mnist_densae_400units_200A200B_numiter15_hyp_bp01_joint_beta1():
    hyp = {
        "experiment_name": "mnist/joint/densae_400units_200A200B_numiter15_hyp_bp01_beta1",
        "train_joint_ae_class": True,
        "warm_start": True,
        "model_path": "../results/trained_models/classification/model_densae_200A200B_disjoint_ae.pt",
        "dense": True,
        "dataset": "mnist",
        "network": "DenSaE",
        "noiseSTD": 0,
        "dictionary_dim": 28,
        "num_conv_A": 200,
        "num_conv_B": 200,
        "D_input": 400,
        "D_output": 10,
        "reshape": ((10, 20), (10, 20)),
        "num_iters": 15,
        "beta": 1,
        "stride": 1,
        "strideA": 1,
        "strideB": 1,
        "split_stride": 1,
        "L": 50,
        "lam": 0.5,
        "twosided": False,
        "normalize": False,
        "batch_size": 16,
        "num_epochs": 200,
        "lr": 1e-3,
        "lr_decay": 1,
        "lr_step": 100,
        "eps": 1e-15,
        "denoising": False,
        "info_period": 5,
        "model_period": 5,
        "loss_period": 5,
    }
