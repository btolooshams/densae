"""
Copyright (c) 2020 Bahareh Tolooshams

plot helpers

:author: Bahareh Tolooshams
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np


def newfig(scale, scale_height=1, square=False):
    """
    Use this function to initialize an image to plot on
    """
    plt.clf()
    fig = plt.figure(figsize=figsize(scale, scale_height, square))
    return fig


def figsize(scale, scale_height=1, square=False):
    """
    This functions defines a figure size with golden ratio, or a square figure size that fits a letter size paper
    figsize(1) will return a fig_size whose width fits a 8 by 11 letter size paper, and height determined by the golde ratio.
    If for some reason you want to strech your plot vertically, use the scale_height argument to 1.5 for example.
    figsize(1, square=True) returns a square figure size that fits a letter size paper.
    """
    fig_width_pt = 416.83269  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    if square:
        fig_size = [fig_width, fig_width]
    else:
        fig_height = fig_width * golden_mean * scale_height  # height in inches
        fig_size = [fig_width, fig_height]
    return fig_size


def update_plot_parameters(
    text_font=50, title_font=50, axes_font=46, legend_font=39, number_font=40
):
    """
    Helper function update plot paramters
    :param text_font: text_font
    :param title_font: title_font
    :param axes_font: axes_font
    :param legend_font: legend_font
    :param number_font: number_font
    :return: none
    """
    sns.set_style("whitegrid")
    sns.set_context("poster")

    ## These handles changing matplotlib background to unify fonts and fontsizes
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [
            "Times"
        ],  # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": axes_font,  # LaTeX default is 10pt font.
        "axes.titlesize": title_font,
        "font.size": text_font,
        "legend.fontsize": legend_font,  # Make the legend/label fonts a little smaller
        "xtick.labelsize": number_font,
        "ytick.labelsize": number_font,
        "figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
        "pgf.preamble": [
            r"\usepackage{fontspec}",
            r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
        ],
    }

    mpl.rcParams.update(pgf_with_latex)

def plot_code(x, net, epoch, writer, dense, reshape=None):
    if dense:
        # plot code
        num_conv = net.num_conv
        a = np.int(np.ceil(np.sqrt(net.num_conv)))
        fig = plt.figure(figsize=(a, a))
        gs1 = gridspec.GridSpec(a, a)
        gs1.update(wspace=0.025, hspace=0.05)
        x = torch.squeeze(x[0]).clone().detach().cpu().numpy()
        x = x.reshape(reshape[0],reshape[1])
        ax1 = plt.subplot(111)
        plt.imshow(x, cmap="gray")
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        writer.add_figure("code", fig, epoch + 1)
    else:
        # plot code
        num_conv = net.num_conv
        a = np.int(np.ceil(np.sqrt(net.num_conv)))
        fig = plt.figure(figsize=(a, a))
        gs1 = gridspec.GridSpec(a, a)
        gs1.update(wspace=0.025, hspace=0.05)
        x = torch.squeeze(x[0]).clone().detach().cpu().numpy()
        for conv in range(num_conv):
            ax1 = plt.subplot(gs1[conv])
            plt.imshow(x_hat[conv], cmap="gray")
            plt.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect("equal")
            plt.subplots_adjust(wspace=None, hspace=None)
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        writer.add_figure("code", fig, epoch + 1)
    return

def plot_dict(net, epoch, writer):
    # plot dict
    num_conv = net.num_conv
    a = np.int(np.ceil(np.sqrt(net.num_conv)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)
    for conv in range(num_conv):
        ax1 = plt.subplot(gs1[conv])
        plt.imshow(
            net.B[conv, 0, :, :].clone().detach().cpu().numpy(), cmap="gray"
        )
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    writer.add_figure("dict", fig, epoch + 1)
    return

def plot_img(img, img_hat, epoch, writer):
    # plot img
    fig = plt.figure()
    ax = plt.subplot(121)
    plt.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    plt.imshow(
        torch.squeeze(img[0]).clone().detach().cpu().numpy(), cmap="gray"
    )
    ax = plt.subplot(122)
    plt.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    plt.imshow(
        torch.squeeze(img_hat[0]).clone().detach().cpu().numpy(), cmap="gray"
    )
    writer.add_figure("img", fig, epoch + 1)
    return

def plot_axbu(Ax, Bu, epoch, writer):
    # plot img
    fig = plt.figure()
    ax = plt.subplot(121)
    plt.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    plt.title("Ax")
    plt.imshow(
        torch.squeeze(Ax[0]).clone().detach().cpu().numpy(), cmap="gray"
    )
    ax = plt.subplot(122)
    plt.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    plt.title("Bu")
    plt.imshow(
        torch.squeeze(Bu[0]).clone().detach().cpu().numpy(), cmap="gray"
    )
    writer.add_figure("Ax_and_Bu", fig, epoch + 1)
    return

def plot_dict_axbu(net, epoch, writer):
    # plot dict
    num_conv = net.num_conv_A
    a = np.int(np.ceil(np.sqrt(net.num_conv_A)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)
    for conv in range(num_conv):
        ax1 = plt.subplot(gs1[conv])
        plt.imshow(
            net.A[conv, 0, :, :].clone().detach().cpu().numpy(), cmap="gray"
        )
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    writer.add_figure("dict-A", fig, epoch + 1)

    # plot dict
    num_conv = net.num_conv_B
    a = np.int(np.ceil(np.sqrt(net.num_conv_B)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)
    for conv in range(num_conv):
        ax1 = plt.subplot(gs1[conv])
        plt.imshow(
            net.B[conv, 0, :, :].clone().detach().cpu().numpy(), cmap="gray"
        )
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    writer.add_figure("dict-B", fig, epoch + 1)
    return

def plot_code_axbu(x, u, net, epoch, writer, dense, reshape=None):
    if dense:
        # plot code
        num_conv = net.num_conv_A
        a = np.int(np.ceil(np.sqrt(net.num_conv_A)))
        fig = plt.figure(figsize=(a, a))
        gs1 = gridspec.GridSpec(a, a)
        gs1.update(wspace=0.025, hspace=0.05)
        x = torch.squeeze(x[0]).clone().detach().cpu().numpy()
        x = x.reshape(reshape[0][0], reshape[0][1])
        ax1 = plt.subplot(111)
        plt.imshow(x, cmap="gray")
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        writer.add_figure("code-x", fig, epoch + 1)

        num_conv = net.num_conv_B
        a = np.int(np.ceil(np.sqrt(net.num_conv_B)))
        fig = plt.figure(figsize=(a, a))
        gs1 = gridspec.GridSpec(a, a)
        gs1.update(wspace=0.025, hspace=0.05)
        u = torch.squeeze(u[0]).clone().detach().cpu().numpy()
        u = u.reshape(reshape[1][0], reshape[1][1])
        ax1 = plt.subplot(111)
        plt.imshow(u, cmap="gray")
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        writer.add_figure("code-u", fig, epoch + 1)
    else:
        # plot code
        num_conv = net.num_conv_A
        a = np.int(np.ceil(np.sqrt(net.num_conv_A)))
        fig = plt.figure(figsize=(a, a))
        gs1 = gridspec.GridSpec(a, a)
        gs1.update(wspace=0.025, hspace=0.05)
        x = torch.squeeze(x[0]).clone().detach().cpu().numpy()
        for conv in range(num_conv):
            ax1 = plt.subplot(gs1[conv])
            plt.imshow(x[conv], cmap="gray")
            plt.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect("equal")
            plt.subplots_adjust(wspace=None, hspace=None)
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        writer.add_figure("code-x", fig, epoch + 1)

        num_conv = net.num_conv_B
        a = np.int(np.ceil(np.sqrt(net.num_conv_B)))
        fig = plt.figure(figsize=(a, a))
        gs1 = gridspec.GridSpec(a, a)
        gs1.update(wspace=0.025, hspace=0.05)
        u = torch.squeeze(u[0]).clone().detach().cpu().numpy()
        for conv in range(num_conv):
            ax1 = plt.subplot(gs1[conv])
            plt.imshow(u[conv], cmap="gray")
            plt.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect("equal")
            plt.subplots_adjust(wspace=None, hspace=None)
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        writer.add_figure("code-u", fig, epoch + 1)
    return
