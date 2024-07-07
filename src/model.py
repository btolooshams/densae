"""
Copyright (c) 2020 Bahareh Tolooshams

models

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np

import utils


class Classifier(torch.nn.Module):
    def __init__(self, hyp):
        super(Classifier, self).__init__()

        self.D_input = hyp["D_input"]
        self.D_output = hyp["D_output"]
        self.device = hyp["device"]

        self.classifier = torch.nn.Linear(self.D_input, self.D_output)
        self.classifier = self.classifier.to(self.device)

    def forward(self, x):
        return self.classifier(x)


class CSCNetTiedHyp(torch.nn.Module):
    def __init__(self, hyp, B=None):
        super(CSCNetTiedHyp, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.lam = hyp["lam"]

        if B is None:
            B = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            B = F.normalize(B, p="fro", dim=(-1, -2))
        self.register_parameter("B", torch.nn.Parameter(B))

        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("B").data = F.normalize(
            self.get_param("B").data, p="fro", dim=(-1, -2)
        )

    def split_image(self, y):
        if self.stride == 1:
            return y, torch.ones_like(y)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            y, self.dictionary_dim, self.stride
        )
        y_batched_padded = torch.zeros(
            y.shape[0],
            self.stride ** 2,
            y.shape[1],
            top_pad + y.shape[2] + bot_pad,
            left_pad + y.shape[3] + right_pad,
            device=self.device,
        ).type_as(y)
        valids_batched = torch.zeros_like(y_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            y_padded = F.pad(
                y,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(y),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            y_batched_padded[:, num, :, :, :] = y_padded
            valids_batched[:, num, :, :, :] = valids
        y_batched_padded = y_batched_padded.reshape(-1, *y_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return y_batched_padded, valids_batched

    def forward(self, y):
        y_batched_padded, valids_batched = self.split_image(y)

        num_batches = y_batched_padded.shape[0]

        D_enc1 = F.conv2d(
            y_batched_padded, self.get_param("B"), stride=self.stride
        ).shape[2]
        D_enc2 = F.conv2d(
            y_batched_padded, self.get_param("B"), stride=self.stride
        ).shape[3]

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        x_tmp = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            Bx = F.conv_transpose2d(x_tmp, self.get_param("B"), stride=self.stride)
            res = y_batched_padded - Bx

            x_new = (
                x_tmp + F.conv2d(res, self.get_param("B"), stride=self.stride) / self.L
            )

            if self.twosided:
                x_new = self.relu(torch.abs(x_new) - self.lam / self.L) * torch.sign(
                    x_new
                )
            else:
                x_new = self.relu(x_new - self.lam / self.L)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            x_tmp = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        y_hat = (
            torch.masked_select(
                F.conv_transpose2d(x_new, self.get_param("B"), stride=self.stride),
                valids_batched.byte(),
            ).reshape(y.shape[0], self.stride ** 2, *y.shape[1:])
        ).mean(dim=1, keepdim=False)

        return y_hat, x_new


class CSCNetTiedLS(torch.nn.Module):
    def __init__(self, hyp, B=None):
        super(CSCNetTiedLS, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]

        if B is None:
            B = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            B = F.normalize(B, p="fro", dim=(-1, -2))

        self.register_parameter("B", torch.nn.Parameter(B))

        self.b = torch.nn.Parameter(
            torch.zeros(1, self.num_conv, 1, 1, device=self.device) + (hyp["b"])
        )

        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("B").data = F.normalize(
            self.get_param("B").data, p="fro", dim=(-1, -2)
        )

    def split_image(self, y):
        if self.stride == 1:
            return y, torch.ones_like(y)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            y, self.dictionary_dim, self.stride
        )
        y_batched_padded = torch.zeros(
            y.shape[0],
            self.stride ** 2,
            y.shape[1],
            top_pad + y.shape[2] + bot_pad,
            left_pad + y.shape[3] + right_pad,
            device=self.device,
        ).type_as(y)
        valids_batched = torch.zeros_like(y_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            y_padded = F.pad(
                y,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(y),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            y_batched_padded[:, num, :, :, :] = y_padded
            valids_batched[:, num, :, :, :] = valids
        y_batched_padded = y_batched_padded.reshape(-1, *y_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return y_batched_padded, valids_batched

    def shrinkage(self, z):
        if self.twosided:
            z = self.relu(torch.abs(z) - self.b) * torch.sign(z)
        else:
            z = self.relu(z - self.b)
        return z

    def forward(self, y):
        if self.stride > 1:
            y_batched_padded, valids_batched = self.split_image(y)
        else:
            y_batched_padded = y

        num_batches = y_batched_padded.shape[0]

        D_enc1 = F.conv2d(
            y_batched_padded, self.get_param("B"), stride=self.stride
        ).shape[2]
        D_enc2 = F.conv2d(
            y_batched_padded, self.get_param("B"), stride=self.stride
        ).shape[3]

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        x_tmp = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            Bx = F.conv_transpose2d(x_tmp, self.get_param("B"), stride=self.stride)
            res = y_batched_padded - Bx

            x_new = (
                x_tmp + F.conv2d(res, self.get_param("B"), stride=self.stride) / self.L
            )
            x_new = self.shrinkage(x_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            x_tmp = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        if self.stride > 1:
            y_hat = (
                torch.masked_select(
                    F.conv_transpose2d(x_new, self.get_param("B"), stride=self.stride),
                    valids_batched.byte(),
                ).reshape(y.shape[0], self.stride ** 2, *y.shape[1:])
            ).mean(dim=1, keepdim=False)
        else:
            y_hat = F.conv_transpose2d(x_new, self.get_param("B"), stride=self.stride)

        return y_hat, x_new


class DenSaE(torch.nn.Module):
    def __init__(self, hyp, A=None, B=None):
        super(DenSaE, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv_A = hyp["num_conv_A"]
        self.num_conv_B = hyp["num_conv_B"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.strideA = hyp["strideA"]
        self.strideB = hyp["strideB"]
        self.twosided = hyp["twosided"]
        self.split_stride = hyp["split_stride"]
        self.lam = hyp["lam"]
        self.rho = hyp["rho"]

        if A is None:
            A = torch.randn(
                (self.num_conv_A, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            A = F.normalize(A, p="fro", dim=(-1, -2))

        if B is None:
            B = torch.randn(
                (self.num_conv_B, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            B = F.normalize(B, p="fro", dim=(-1, -2))

        self.register_parameter("A", torch.nn.Parameter(A))
        self.register_parameter("B", torch.nn.Parameter(B))

        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("A").data = F.normalize(
            self.get_param("A").data, p="fro", dim=(-1, -2)
        )
        self.get_param("B").data = F.normalize(
            self.get_param("B").data, p="fro", dim=(-1, -2)
        )

    def split_image(self, y):
        if self.split_stride == 1:
            return y, torch.ones_like(y)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            y, self.dictionary_dim, self.split_stride
        )
        y_batched_padded = torch.zeros(
            y.shape[0],
            self.split_stride ** 2,
            y.shape[1],
            top_pad + y.shape[2] + bot_pad,
            left_pad + y.shape[3] + right_pad,
            device=self.device,
        ).type_as(y)
        valids_batched = torch.zeros_like(y_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.split_stride) for j in range(self.split_stride)]
        ):
            y_padded = F.pad(
                y,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(y),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            y_batched_padded[:, num, :, :, :] = y_padded
            valids_batched[:, num, :, :, :] = valids
        y_batched_padded = y_batched_padded.reshape(-1, *y_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return y_batched_padded, valids_batched

    def shrinkage(self, z):
        if self.twosided:
            z = self.relu(torch.abs(z) - self.lam / self.L) * torch.sign(z)
        else:
            z = self.relu(z - self.lam / self.L)
        return z

    def forward(self, y):
        y_batched_padded, valids_batched = self.split_image(y)

        num_batches = y_batched_padded.shape[0]

        [D_x1, D_x2] = F.conv2d(
            y_batched_padded, self.get_param("A"), stride=self.strideA
        ).shape[2:]

        [D_u1, D_u2] = F.conv2d(
            y_batched_padded, self.get_param("B"), stride=self.strideB
        ).shape[2:]

        x_old = torch.zeros(
            num_batches, self.num_conv_A, D_x1, D_x2, device=self.device
        )
        x_tmp = torch.zeros(
            num_batches, self.num_conv_A, D_x1, D_x2, device=self.device
        )
        x_new = torch.zeros(
            num_batches, self.num_conv_A, D_x1, D_x2, device=self.device
        )

        u_old = torch.zeros(
            num_batches, self.num_conv_B, D_u1, D_u2, device=self.device
        )
        u_tmp = torch.zeros(
            num_batches, self.num_conv_B, D_u1, D_u2, device=self.device
        )
        u_new = torch.zeros(
            num_batches, self.num_conv_B, D_u1, D_u2, device=self.device
        )

        del D_x1
        del D_x2
        del D_u1
        del D_u2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            Ax = F.conv_transpose2d(
                (1 + 1.0 / self.rho) * x_tmp, self.get_param("A"), stride=self.strideA
            )
            Bu = F.conv_transpose2d(u_tmp, self.get_param("B"), stride=self.strideB)

            res = y_batched_padded - (Ax + Bu)

            x_new = (
                x_tmp + F.conv2d(res, self.get_param("A"), stride=self.strideA) / self.L
            )
            u_new = (
                u_tmp + F.conv2d(res, self.get_param("B"), stride=self.strideB) / self.L
            )

            u_new = self.shrinkage(u_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2

            x_tmp = x_new + (t_old - 1) / t_new * (x_new - x_old)
            u_tmp = u_new + (t_old - 1) / t_new * (u_new - u_old)

            x_old = x_new
            u_old = u_new
            t_old = t_new

        if self.split_stride > 1:
            Ax_hat = (
                torch.masked_select(
                    F.conv_transpose2d(x_new, self.get_param("A"), stride=self.strideA),
                    valids_batched.byte(),
                ).reshape(y.shape[0], self.split_stride ** 2, *y.shape[1:])
            ).mean(dim=1, keepdim=False)
            Bu_hat = (
                torch.masked_select(
                    F.conv_transpose2d(u_new, self.get_param("B"), stride=self.strideB),
                    valids_batched.byte(),
                ).reshape(y.shape[0], self.split_stride ** 2, *y.shape[1:])
            ).mean(dim=1, keepdim=False)
        else:
            Ax_hat = F.conv_transpose2d(x_new, self.get_param("A"), stride=self.strideA)
            Bu_hat = F.conv_transpose2d(u_new, self.get_param("B"), stride=self.strideB)

        y_hat = Ax_hat + Bu_hat

        return y_hat, [x_new, u_new, Ax_hat, Bu_hat]


class DenSaEhyp(torch.nn.Module):
    def __init__(self, hyp, A=None, B=None):
        super(DenSaEhyp, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv_A = hyp["num_conv_A"]
        self.num_conv_B = hyp["num_conv_B"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.strideA = hyp["strideA"]
        self.strideB = hyp["strideB"]
        self.twosided = hyp["twosided"]
        self.split_stride = hyp["split_stride"]
        self.lam = hyp["lam"]
        self.rho = hyp["rho"]

        if A is None:
            A = torch.randn(
                (self.num_conv_A, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            A = F.normalize(A, p="fro", dim=(-1, -2))

        if B is None:
            B = torch.randn(
                (self.num_conv_B, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            B = F.normalize(B, p="fro", dim=(-1, -2))

        self.register_parameter("A", torch.nn.Parameter(A))
        self.register_parameter("B", torch.nn.Parameter(B))

        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("A").data = F.normalize(
            self.get_param("A").data, p="fro", dim=(-1, -2)
        )
        self.get_param("B").data = F.normalize(
            self.get_param("B").data, p="fro", dim=(-1, -2)
        )

    def split_image(self, y):
        if self.split_stride == 1:
            return y, torch.ones_like(y)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            y, self.dictionary_dim, self.split_stride
        )
        y_batched_padded = torch.zeros(
            y.shape[0],
            self.split_stride ** 2,
            y.shape[1],
            top_pad + y.shape[2] + bot_pad,
            left_pad + y.shape[3] + right_pad,
            device=self.device,
        ).type_as(y)
        valids_batched = torch.zeros_like(y_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.split_stride) for j in range(self.split_stride)]
        ):
            y_padded = F.pad(
                y,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(y),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            y_batched_padded[:, num, :, :, :] = y_padded
            valids_batched[:, num, :, :, :] = valids
        y_batched_padded = y_batched_padded.reshape(-1, *y_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return y_batched_padded, valids_batched

    def shrinkage(self, z):
        if self.twosided:
            z = self.relu(torch.abs(z) - self.lam / self.L) * torch.sign(z)
        else:
            z = self.relu(z - self.lam / self.L)
        return z

    def forward(self, y):
        y_batched_padded, valids_batched = self.split_image(y)

        num_batches = y_batched_padded.shape[0]

        [D_x1, D_x2] = F.conv2d(
            y_batched_padded, self.get_param("A"), stride=self.strideA
        ).shape[2:]

        [D_u1, D_u2] = F.conv2d(
            y_batched_padded, self.get_param("B"), stride=self.strideB
        ).shape[2:]

        x_old = torch.zeros(
            num_batches, self.num_conv_A, D_x1, D_x2, device=self.device
        )
        x_tmp = torch.zeros(
            num_batches, self.num_conv_A, D_x1, D_x2, device=self.device
        )
        x_new = torch.zeros(
            num_batches, self.num_conv_A, D_x1, D_x2, device=self.device
        )

        u_old = torch.zeros(
            num_batches, self.num_conv_B, D_u1, D_u2, device=self.device
        )
        u_tmp = torch.zeros(
            num_batches, self.num_conv_B, D_u1, D_u2, device=self.device
        )
        u_new = torch.zeros(
            num_batches, self.num_conv_B, D_u1, D_u2, device=self.device
        )

        del D_x1
        del D_x2
        del D_u1
        del D_u2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            Ax = F.conv_transpose2d(
                (1 + 1.0 / self.rho) * x_tmp, self.get_param("A"), stride=self.strideA
            )
            Bu = F.conv_transpose2d(u_tmp, self.get_param("B"), stride=self.strideB)

            res = y_batched_padded - (Ax + Bu)

            x_new = (
                x_tmp + F.conv2d(res, self.get_param("A"), stride=self.strideA) / self.L
            )
            u_new = (
                u_tmp + F.conv2d(res, self.get_param("B"), stride=self.strideB) / self.L
            )

            u_new = self.shrinkage(u_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2

            x_tmp = x_new + (t_old - 1) / t_new * (x_new - x_old)
            u_tmp = u_new + (t_old - 1) / t_new * (u_new - u_old)

            x_old = x_new
            u_old = u_new
            t_old = t_new

        if self.split_stride > 1:
            Ax_hat = (
                torch.masked_select(
                    F.conv_transpose2d(x_new, self.get_param("A"), stride=self.strideA),
                    valids_batched.byte(),
                ).reshape(y.shape[0], self.split_stride ** 2, *y.shape[1:])
            ).mean(dim=1, keepdim=False)
            Bu_hat = (
                torch.masked_select(
                    F.conv_transpose2d(u_new, self.get_param("B"), stride=self.strideB),
                    valids_batched.byte(),
                ).reshape(y.shape[0], self.split_stride ** 2, *y.shape[1:])
            ).mean(dim=1, keepdim=False)
        else:
            Ax_hat = F.conv_transpose2d(x_new, self.get_param("A"), stride=self.strideA)
            Bu_hat = F.conv_transpose2d(u_new, self.get_param("B"), stride=self.strideB)

        y_hat = Ax_hat + Bu_hat

        return y_hat, [x_new, u_new, Ax_hat, Bu_hat]
