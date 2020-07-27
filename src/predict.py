"""
Copyright (c) 2020 Bahareh Tolooshams

predict

:author: Bahareh Tolooshams
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec

from sacred import Experiment

import sys

sys.path.append("src/")

import generator, trainer, utils, plot_helpers

import warnings

warnings.filterwarnings("ignore")

ex = Experiment("predict")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


@ex.automain
def predict():

    name = "noise15_cscnet_tied_ls"
    random_date = "2020_05_27_14_51_29"
    csc1_densae0 = 1
    plot_bias = 1
    noiseSTD = 15
    num_epochs = 249

    experiment_name = os.path.join(name)
    test_path = "../data/visu/"
    img_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

    PATH = "../results/{}/{}".format(experiment_name, random_date)

    print(experiment_name)
    print(random_date)

    print("create model.")
    net_init = torch.load(os.path.join(PATH, "model_init.pt"), map_location=device)
    net = torch.load(
        os.path.join(PATH, "model_epoch{}.pt".format(num_epochs)), map_location=device,
    )

    net_init.device = device
    net.device = device

    if csc1_densae0:
        B_init = net_init.get_param("B").clone().detach().cpu().numpy()
        B = net.get_param("B").clone().detach().cpu().numpy()

        num_conv = B_init.shape[0]
        a = np.int(np.ceil(np.sqrt(num_conv)))
        fig = plt.figure(figsize=(a, a))
        gs1 = gridspec.GridSpec(a, a)
        gs1.update(wspace=0.025, hspace=0.05)
        conv_ctr = 0
        for conv in range(num_conv):
            ax1 = plt.subplot(gs1[conv])
            plt.imshow(B_init[conv, 0, :, :], cmap="gray")
            plt.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect("equal")
            conv_ctr += 1
            plt.subplots_adjust(wspace=None, hspace=None)
        plt.savefig(os.path.join(PATH, "B_init.png"))

        fig = plt.figure(figsize=(a, a))
        gs1 = gridspec.GridSpec(a, a)
        gs1.update(wspace=0.025, hspace=0.05)
        conv_ctr = 0
        for conv in range(num_conv):
            ax1 = plt.subplot(gs1[conv])
            plt.imshow(B[conv, 0, :, :], cmap="gray")
            plt.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect("equal")
            conv_ctr += 1
            plt.subplots_adjust(wspace=None, hspace=None)
        plt.savefig(os.path.join(PATH, "B_learned.png"))

        B_init = net_init.get_param("B")
        B = net.get_param("B")
    else:
        A_init = net_init.get_param("A").clone().detach().cpu().numpy()
        A = net.get_param("A").clone().detach().cpu().numpy()

        B_init = net_init.get_param("B").clone().detach().cpu().numpy()
        B = net.get_param("B").clone().detach().cpu().numpy()

        num_conv = A_init.shape[0]
        a = np.int(np.ceil(np.sqrt(num_conv)))
        fig = plt.figure(figsize=(a, a))
        gs1 = gridspec.GridSpec(a, a)
        gs1.update(wspace=0.025, hspace=0.05)
        for conv in range(num_conv):
            ax1 = plt.subplot(gs1[conv])
            plt.imshow(A_init[conv, 0, :, :], cmap="gray")
            plt.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect("equal")
            plt.subplots_adjust(wspace=None, hspace=None)
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        plt.savefig(os.path.join(PATH, "A_init.png"))

        fig = plt.figure(figsize=(a, a))
        gs1 = gridspec.GridSpec(a, a)
        gs1.update(wspace=0.025, hspace=0.05)
        for conv in range(num_conv):
            ax1 = plt.subplot(gs1[conv])
            plt.imshow(A[conv, 0, :, :], cmap="gray")
            plt.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect("equal")
            plt.subplots_adjust(wspace=None, hspace=None)
        plt.savefig(os.path.join(PATH, "A_learned.png"))

        num_conv = B_init.shape[0]

        a = np.int(np.ceil(np.sqrt(num_conv)))
        fig = plt.figure(figsize=(a, a))
        gs1 = gridspec.GridSpec(a, a)
        gs1.update(wspace=0.025, hspace=0.05)
        for conv in range(num_conv):
            ax1 = plt.subplot(gs1[conv])
            plt.imshow(B_init[conv, 0, :, :], cmap="gray")
            plt.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect("equal")
            plt.subplots_adjust(wspace=None, hspace=None)
        plt.savefig(os.path.join(PATH, "B_init.png"))

        fig = plt.figure(figsize=(a, a))
        gs1 = gridspec.GridSpec(a, a)
        gs1.update(wspace=0.025, hspace=0.05)
        for conv in range(num_conv):
            ax1 = plt.subplot(gs1[conv])
            plt.imshow(B[conv, 0, :, :], cmap="gray")
            plt.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect("equal")
            plt.subplots_adjust(wspace=None, hspace=None)
        plt.savefig(os.path.join(PATH, "B_learned.png"))

        A = net.get_param("A")
        B = net.get_param("B").clone()

    print("load data.")
    test_loader = generator.get_path_loader(1, test_path, shuffle=False)

    with torch.no_grad():
        ctr = 0
        for img_org, tar in test_loader:

            img_noisy = (img_org + noiseSTD / 255 * torch.randn(img_org.shape)).to(
                device
            )
            img = img_org.to(device)

            img_hat, r = net(img_noisy)


            torchvision.utils.save_image(
                img.clone(), os.path.join(PATH, "{}_img.png".format(img_list[ctr])),
            )
            torchvision.utils.save_image(
                img_noisy.clone(),
                os.path.join(PATH, "{}_noisy.png".format(img_list[ctr])),
            )
            torchvision.utils.save_image(
                bu_hat.clone(),
                os.path.join(PATH, "{}_img_hat.png".format(img_list[ctr])),
            )

            if not csc1_densae0:
                ax_hat = r[-1]
                bu_hat = r[-2]

                torchvision.utils.save_image(
                    ax_hat.clone(),
                    os.path.join(PATH, "{}_ax_hat.png".format(img_list[ctr])),
                )
                torchvision.utils.save_image(
                    bu_hat.clone(),
                    os.path.join(PATH, "{}_bu_hat.png".format(img_list[ctr])),
                )

            if plot_bias:
                bias = (torch.squeeze(net.b)).clone().detach().cpu().numpy()

                # upadte plot parameters
                plot_helpers.update_plot_parameters(
                    text_font=25,
                    title_font=25,
                    axes_font=25,
                    legend_font=25,
                    number_font=25,
                )

                fig = plot_helpers.newfig(scale=1, scale_height=1)
                ax = fig.add_subplot(111)
                ax.grid(False)
                hist, bins = np.histogram(np.log10(np.abs(bias)), 100)
                width = 0.9 * (bins[1] - bins[0])
                center = (bins[:-1] + bins[1:]) / 2

                plt.bar(
                    center, hist, align="center", width=width, linewidth=width,
                )

                plt.ylim(0, 10)
                plt.ylabel("$\mathrm{Count}$", fontsize=30)
                plt.xlabel("$\mathrm{Filter\;biases\;(log\;scale)}$", fontsize=30)

                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                plt.subplots_adjust(wspace=None, hspace=None)
                ax.grid(False)
                fig.tight_layout(pad=0.1, w_pad=0.0, h_pad=0)
                plt.savefig(os.path.join(PATH, "bias_{}.eps".format(noiseSTD)))

            ctr += 1
