"""
Copyright (c) 2020 Bahareh Tolooshams

utils

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np


class L2loss(torch.nn.Module):
    def __init__(self):
        super(L2loss, self).__init__()

    def forward(self, y, yhat):
        loss = (y - yhat).pow(2).sum() / y.shape[0]
        return loss


def PSNR(x, x_hat):
    mse = np.mean((x - x_hat) ** 2)
    max_x = np.max(x)
    return 20 * np.log10(max_x) - 10 * np.log10(mse)


def calc_pad_sizes(x, dictionary_dim=8, stride=1):
    left_pad = stride
    right_pad = (
        0
        if (x.shape[3] + left_pad - dictionary_dim) % stride == 0
        else stride - ((x.shape[3] + left_pad - dictionary_dim) % stride)
    )
    top_pad = stride
    bot_pad = (
        0
        if (x.shape[2] + top_pad - dictionary_dim) % stride == 0
        else stride - ((x.shape[2] + top_pad - dictionary_dim) % stride)
    )
    right_pad += stride
    bot_pad += stride
    return left_pad, right_pad, top_pad, bot_pad


def conv_power_method(D, image_size, num_iters=100, stride=1, device="cpu"):
    needles_shape = [
        int(((image_size[0] - D.shape[-2]) / stride) + 1),
        int(((image_size[1] - D.shape[-1]) / stride) + 1),
    ]
    x = torch.randn(1, D.shape[0], *needles_shape, device=device).type_as(D)
    for _ in range(num_iters):
        c = torch.norm(x.reshape(-1))
        x = x / c
        y = F.conv_transpose2d(x, D, stride=stride)
        x = F.conv2d(y, D, stride=stride)
    return torch.norm(x.reshape(-1))
