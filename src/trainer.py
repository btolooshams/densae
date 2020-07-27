"""
Copyright (c) 2020 Bahareh Tolooshams

trainer

:author: Bahareh Tolooshams
"""

import torch
from tqdm import tqdm
import os
import numpy as np

import utils


def train_ae(
    net,
    data_loader,
    hyp,
    criterion,
    optimizer,
    scheduler,
    writer,
    PATH="",
    test_loader=None,
):

    info_period = hyp["info_period"]
    device = hyp["device"]
    normalize = hyp["normalize"]
    supervised = hyp["supervised"]
    noiseSTD = hyp["noiseSTD"]
    num_epochs = hyp["num_epochs"]
    denoising = hyp["denoising"]

    if normalize:
        net.normalize()

    if denoising:
        if test_loader is not None:
            with torch.no_grad():
                psnr = []
                t = 0
                for idx_test, (img_test, _) in enumerate(test_loader):
                    t += 1

                    psnr_i = 0
                    N = 1
                    for _ in range(N):

                        img_test_noisy = (
                            img_test + noiseSTD / 255 * torch.randn(img_test.shape)
                        ).to(device)

                        img_test_hat, _ = net(img_test_noisy)

                        img_test_hat = torch.clamp(img_test_hat, 0, 1)

                        psnr_i += utils.PSNR(
                            img_test[0, 0, :, :].clone().detach().cpu().numpy(),
                            img_test_hat[0, 0, :, :].clone().detach().cpu().numpy(),
                        )

                    psnr.append(psnr_i / N)

                    noisy_psnr = utils.PSNR(
                        img_test[0, 0, :, :].clone().detach().cpu().numpy(),
                        img_test_noisy[0, 0, :, :].clone().detach().cpu().numpy(),
                    )

            writer.add_scalar("test_psnr", np.mean(np.array(psnr)), 0)
            print("PSNR: {}".format(np.round(np.mean(np.array(psnr)), decimals=4)))

            np.save(os.path.join(PATH, "psnr_init.npy"), np.array(psnr))

    for epoch in tqdm(range(num_epochs), disable=True):

        scheduler.step()
        loss_all = 0
        for idx, (img, _) in tqdm(enumerate(data_loader), disable=True):
            optimizer.zero_grad()

            img_noisy = (img + noiseSTD / 255 * torch.randn(img.shape)).to(device)
            img = img.to(device)

            img_hat, _ = net(img_noisy)

            if supervised:
                loss = criterion(img, img_hat)
            else:
                loss = criterion(img_noisy, img_hat)

            loss_all += float(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if normalize:
                net.normalize()

            torch.cuda.empty_cache()

        writer.add_scalar("training_loss", loss_all, epoch + 1)

        if denoising:
            if test_loader is not None:
                with torch.no_grad():
                    psnr = []
                    t = 0
                    for idx_test, (img_test, _) in enumerate(test_loader):
                        t += 1

                        psnr_i = 0
                        N = 1
                        for _ in range(N):
                            img_test_noisy = (
                                img_test + noiseSTD / 255 * torch.randn(img_test.shape)
                            ).to(device)

                            img_test_hat, _ = net(img_test_noisy)

                            img_test_hat = torch.clamp(img_test_hat, 0, 1)

                            psnr_i += utils.PSNR(
                                img_test[0, 0, :, :].clone().detach().cpu().numpy(),
                                img_test_hat[0, 0, :, :].clone().detach().cpu().numpy(),
                            )

                        psnr.append(psnr_i / N)

                writer.add_scalar("test_psnr", np.mean(np.array(psnr)), epoch + 1)
                print(
                    "epoch {}: PSNR {}".format(
                        epoch, np.round(np.mean(np.array(psnr)), decimals=4)
                    )
                )

                np.save(
                    os.path.join(PATH, "psnr_epoch{}.npy".format(epoch)),
                    np.mean(np.array(psnr)),
                )

        torch.save(loss_all, os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))
        torch.save(net, os.path.join(PATH, "model_epoch{}.pt".format(epoch)))

    writer.close()

    return net
