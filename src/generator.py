"""
Copyright (c) 2020 Bahareh Tolooshams

generator and data loader

:author: Bahareh Tolooshams
"""

import torch, torchvision
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def get_MNIST_loaders(batch_size, shuffle=False, train_batch=None, test_batch=None):
    if train_batch == None:
        train_loader, val_loader = get_MNIST_loader(batch_size, trainable=True, shuffle=shuffle)
    else:
        train_loader, val_loader = get_MNIST_loader(train_batch, trainable=True, shuffle=shuffle)

    if test_batch == None:
        test_loader = get_MNIST_loader(batch_size, trainable=False, shuffle=False)
    else:
        test_loader = get_MNIST_loader(test_batch, trainable=False, shuffle=False)
    return train_loader, val_loader, test_loader


def get_MNIST_loader(batch_size, trainable=True, shuffle=False, num_workers=4):
    dataset = torchvision.datasets.MNIST(
                "../data",
                train=trainable,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor(),]
                ),
            )
    if trainable:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [50000, 10000])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        return train_loader, val_loader
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        return loader

def get_test_path_loader(batch_size, image_path, shuffle=False, num_workers=4):
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            root=image_path,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader


def get_train_path_loader(
    batch_size, image_path, crop_dim=(128, 128), shuffle=True, num_workers=4
):
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            root=image_path,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.RandomCrop(
                        crop_dim,
                        padding=None,
                        pad_if_needed=True,
                        fill=0,
                        padding_mode="constant",
                    ),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader

class EncodingDataset(torch.utils.data.Dataset):
    def __init__(self, data_loader, net, hyp, device=None, ind=1, transform=None, seed=None):
        self.x = []
        self.c = []
        print("create encoding dataset.")
        for idx, (img, c) in tqdm(enumerate(data_loader), disable=True):
            img = img.to(device)

            _, code = net(img)

            if hyp["network"] == "DenSaEhyp":
                x, u, _, _ = code

                u = u.clone().detach().requires_grad_(False)
                u = u.reshape(-1, u.shape[1]*u.shape[2]*u.shape[3])
                x = x.clone().detach().requires_grad_(False)
                x = x.reshape(-1, x.shape[1]*x.shape[2]*x.shape[3])

                if hyp["classify_use_only_u"]:
                    if hyp["random_remove_u"]:
                        u = u[:, ind[:-hyp["random_remove_u"]]]
                    x = u
                else:
                    if hyp["random_remove_u"]:
                        u = u[:, ind[:-hyp["random_remove_u"]]]
                    x = torch.cat([x,u], dim=-1)

                x = F.normalize(x, dim=1)
            else:
                x = code
                x = x.clone().detach().requires_grad_(False)
                x = x.reshape(-1, x.shape[1]*x.shape[2]*x.shape[3])
                x = F.normalize(x, dim=1)

            self.x.append(x)
            self.c.append(c)

        self.x = torch.cat(self.x)
        self.c = torch.cat(self.c)
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]

        if self.transform:
            x = self.transform(x).float()

        return x, self.c[idx]

def get_encoding_loaders(train_loader, val_loader, test_loader, net, hyp):
    if hyp["random_remove_u"]:
        u_shape = net.num_conv_B
        ind = np.linspace(1,u_shape-1,u_shape)
        np.random.shuffle(ind)
    else:
        ind = 1
    train_dataset = EncodingDataset(train_loader, net, hyp, hyp["device"], ind)
    val_dataset = EncodingDataset(val_loader, net, hyp, hyp["device"], ind)
    test_dataset = EncodingDataset(test_loader, net, hyp, hyp["device"], ind)
    enc_tr_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyp["batch_size"])
    enc_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hyp["batch_size"])
    enc_te_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hyp["batch_size"])
    return enc_tr_loader, enc_val_loader, enc_te_loader
