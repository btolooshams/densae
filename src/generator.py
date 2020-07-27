"""
Copyright (c) 2020 Bahareh Tolooshams

generator and data loader

:author: Bahareh Tolooshams
"""

import torch, torchvision


def get_path_loader(batch_size, image_path, shuffle=False, num_workers=4):
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
