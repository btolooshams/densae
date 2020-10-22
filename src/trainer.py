"""
Copyright (c) 2020 Bahareh Tolooshams

trainer

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

import utils, plot_helpers


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
    loss_period = hyp["loss_period"]
    model_period = hyp["model_period"]
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
        if epoch > 0:
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

        if (epoch + 1) % info_period == 0:
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

        if (epoch + 1) % loss_period == 0:
            torch.save(loss_all, os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))
        if (epoch + 1) % model_period == 0:
            torch.save(net, os.path.join(PATH, "model_epoch{}.pt".format(epoch)))

    writer.close()

    return net


def train_ae_mnist(
    net,
    data_loader,
    hyp,
    criterion,
    optimizer,
    scheduler,
    writer,
    PATH="",
    val_loader=None,
    test_loader=None,
):

    info_period = hyp["info_period"]
    loss_period = hyp["loss_period"]
    model_period = hyp["model_period"]
    device = hyp["device"]
    normalize = hyp["normalize"]
    supervised = hyp["supervised"]
    noiseSTD = hyp["noiseSTD"]
    num_epochs = hyp["num_epochs"]

    if normalize:
        net.normalize()

    for epoch in tqdm(range(num_epochs), disable=True):
        if epoch > 0:
            scheduler.step()
        loss_all = 0
        for idx, (img, _) in tqdm(enumerate(data_loader), disable=True):
            optimizer.zero_grad()

            img_noisy = (img + noiseSTD / 255 * torch.randn(img.shape)).to(device)
            img = img.to(device)

            img_hat, out = net(img_noisy)

            loss = criterion(img, img_hat)

            loss_all += float(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if normalize:
                net.normalize()

            torch.cuda.empty_cache()

        if (epoch + 1) % info_period == 0:
            writer.add_scalar("training_loss", loss_all, epoch + 1)

            if hyp["network"] == "CSCNetTiedLS":
                # plot bias
                bias = torch.squeeze(net.b).clone().detach().cpu().numpy()
                fig = plt.figure()
                ax = plt.subplot(111)
                plt.plot(bias, ".")
                writer.add_figure("bias", fig, epoch + 1)

            if hyp["network"] == "CSCNetTiedLS":
                x_hat = out
            elif hyp["network"] == "CSCNetTiedHyp":
                x_hat = out
            elif hyp["network"] == "DenSaE":
                [x_hat, u_hat, Ax_hat, Bu_hat] = out

            plot_helpers.plot_img(img, img_hat, epoch, writer)

            if hyp["network"] == "DenSaE":
                plot_helpers.plot_axbu(Ax_hat, Bu_hat, epoch, writer)
                plot_helpers.plot_code_axbu(
                    x_hat, u_hat, net, epoch, writer, hyp["dense"], hyp["reshape"]
                )
                plot_helpers.plot_dict_axbu(net, epoch, writer)
            else:
                plot_helpers.plot_code(
                    x_hat, net, epoch, writer, hyp["dense"], hyp["reshape"]
                )
                plot_helpers.plot_dict(net, epoch, writer)

        if (epoch + 1) % loss_period == 0:
            torch.save(loss_all, os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))
        if (epoch + 1) % model_period == 0:
            torch.save(net, os.path.join(PATH, "model_epoch{}.pt".format(epoch)))

        if (epoch + 1) % info_period == 0:
            print(
                "epoch [{}/{}], loss:{:.10f}".format(
                    epoch + 1, hyp["num_epochs"], loss.item()
                )
            )

    writer.close()

    return net


def train_join_ae_class_mnist(
    net,
    classifier,
    data_loader,
    hyp,
    criterion,
    criterion_class,
    optimizer,
    scheduler,
    writer,
    PATH="",
    val_loader=None,
    test_loader=None,
):

    info_period = hyp["info_period"]
    loss_period = hyp["loss_period"]
    model_period = hyp["model_period"]
    device = hyp["device"]
    normalize = hyp["normalize"]
    supervised = hyp["supervised"]
    noiseSTD = hyp["noiseSTD"]
    num_epochs = hyp["num_epochs"]
    beta = hyp["beta"]

    if normalize:
        net.normalize()

    for epoch in tqdm(range(num_epochs), disable=True):
        if epoch > 0:
            scheduler.step()
        loss_all = 0
        for idx, (img, c) in tqdm(enumerate(data_loader), disable=True):
            optimizer.zero_grad()

            img_noisy = (img + noiseSTD / 255 * torch.randn(img.shape)).to(device)
            img = img.to(device)
            c = c.to(device)

            img_hat, out = net(img_noisy)

            if hyp["network"] == "DenSaEv2" or hyp["network"] == "DenSaE":
                x, u, _, _ = out

                u = u.reshape(-1, u.shape[1] * u.shape[2] * u.shape[3])
                x = x.reshape(-1, x.shape[1] * x.shape[2] * x.shape[3])
                code = torch.cat([x, u], dim=-1)
                code = F.normalize(code, dim=1)
            else:
                code = out
                code = code.reshape(-1, code.shape[1] * code.shape[2] * code.shape[3])
                code = F.normalize(code, dim=1)

            c_hat = classifier(code)

            loss_ae = criterion(img, img_hat)
            loss_class = criterion_class(c_hat, c)

            loss = (1 - beta) * loss_ae + beta * loss_class

            loss_all += float(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if normalize:
                net.normalize()

            torch.cuda.empty_cache()

        if (epoch + 1) % info_period == 0:
            train_acc = test_network_joint(data_loader, net, classifier, hyp)
            val_acc = test_network_joint(val_loader, net, classifier, hyp)
            test_acc = test_network_joint(test_loader, net, classifier, hyp)

            writer.add_scalar("train-acc", train_acc, epoch + 1)
            writer.add_scalar("val-acc", val_acc, epoch + 1)
            writer.add_scalar("test-acc", test_acc, epoch + 1)
            writer.add_scalar("training_loss", loss_all, epoch + 1)

            if hyp["network"] == "CSCNetTiedLS":
                # plot bias
                bias = torch.squeeze(net.b).clone().detach().cpu().numpy()
                fig = plt.figure()
                ax = plt.subplot(111)
                plt.plot(bias, ".")
                writer.add_figure("bias", fig, epoch + 1)

            if hyp["network"] == "CSCNetTiedLS":
                x_hat = out
            elif hyp["network"] == "CSCNetTiedHyp":
                x_hat = out
            elif hyp["network"] == "DenSaE":
                [x_hat, u_hat, Ax_hat, Bu_hat] = out

            plot_helpers.plot_img(img, img_hat, epoch, writer)

            if hyp["network"] == "DenSaE":
                plot_helpers.plot_axbu(Ax_hat, Bu_hat, epoch, writer)
                plot_helpers.plot_code_axbu(
                    x_hat, u_hat, net, epoch, writer, hyp["dense"], hyp["reshape"]
                )
                plot_helpers.plot_dict_axbu(net, epoch, writer)
            else:
                plot_helpers.plot_code(
                    x_hat, net, epoch, writer, hyp["dense"], hyp["reshape"]
                )
                plot_helpers.plot_dict(net, epoch, writer)

        if (epoch + 1) % loss_period == 0:
            torch.save(loss_all, os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))
        if (epoch + 1) % model_period == 0:
            torch.save(net, os.path.join(PATH, "model_epoch{}.pt".format(epoch)))
            torch.save(
                classifier, os.path.join(PATH, "classifier_epoch{}.pt".format(epoch))
            )

        if (epoch + 1) % info_period == 0:
            print(
                "epoch [{}/{}], loss:{:.10f}, train acc:{:.4f}, val acc:{:.4f}, test acc:{:.4f}".format(
                    epoch + 1,
                    hyp["num_epochs"],
                    loss.item(),
                    train_acc,
                    val_acc,
                    test_acc,
                )
            )

    writer.close()

    return net


def train_classifier(
    net,
    data_loader,
    hyp,
    criterion,
    optimizer,
    scheduler,
    writer,
    PATH,
    val_loader=None,
    test_loader=None,
):

    info_period = hyp["info_period"]
    loss_period = hyp["loss_period"]
    model_period = hyp["model_period"]
    device = hyp["device"]
    num_epochs = hyp["num_epochs"]

    for epoch in tqdm(range(num_epochs), disable=True):
        if epoch > 0:
            scheduler.step()
        loss_all = 0
        for idx, (x, c) in tqdm(enumerate(data_loader), disable=True):
            optimizer.zero_grad()

            x = x.to(device)
            c = c.to(device)

            output = net(x)

            loss = criterion(output, c)

            loss_all += float(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

        if (epoch + 1) % info_period == 0:
            train_acc = test_network(data_loader, net, hyp)
            val_acc = test_network(val_loader, net, hyp)
            test_acc = test_network(test_loader, net, hyp)

            writer.add_scalar("train-acc", train_acc, epoch + 1)
            writer.add_scalar("val-acc", val_acc, epoch + 1)
            writer.add_scalar("test-acc", test_acc, epoch + 1)
            writer.add_scalar("training_loss", loss_all, epoch + 1)

            print(
                "epoch [{}/{}], loss:{:.10f}, train acc:{:.4f}, val acc:{:.4f}, test acc:{:.4f}".format(
                    epoch + 1, hyp["num_epochs"], loss_all, train_acc, val_acc, test_acc
                )
            )

        if (epoch + 1) % loss_period == 0:
            torch.save(loss_all, os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))
        if (epoch + 1) % model_period == 0:
            torch.save(net, os.path.join(PATH, "model_epoch{}.pt".format(epoch)))

    writer.close()

    return


def test_network(data_loader, net, hyp):

    device = hyp["device"]

    with torch.no_grad():
        num_correct = 0
        num_total = 0
        correct_ex = []
        incorrect_ex = []
        examples = 300
        for idx, (x, c) in tqdm(enumerate(data_loader), disable=True):

            x = x.to(device)
            c = c.to(device)

            c_hat = net(x)

            correct_indicators = c_hat.max(1)[1].data == c
            num_correct += correct_indicators.sum().item()
            num_total += c.size()[0]

    acc = num_correct / num_total

    return acc


def test_network_joint(data_loader, net, classifier, hyp):

    device = hyp["device"]

    with torch.no_grad():
        num_correct = 0
        num_total = 0
        correct_ex = []
        incorrect_ex = []
        examples = 300
        for idx, (img, c) in tqdm(enumerate(data_loader), disable=True):

            img = img.to(device)
            c = c.to(device)

            _, out = net(img)

            if hyp["network"] == "DenSaE":
                x, u, _, _ = out

                u = u.reshape(-1, u.shape[1] * u.shape[2] * u.shape[3])
                x = x.reshape(-1, x.shape[1] * x.shape[2] * x.shape[3])
                code = torch.cat([x, u], dim=-1)
                code = F.normalize(code, dim=1)
            else:
                code = out
                code = code.reshape(-1, code.shape[1] * code.shape[2] * code.shape[3])
                code = F.normalize(code, dim=1)

            c_hat = classifier(code)

            correct_indicators = c_hat.max(1)[1].data == c
            num_correct += correct_indicators.sum().item()
            num_total += c.size()[0]

    acc = num_correct / num_total

    return acc
