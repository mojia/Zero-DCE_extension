import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def log_gradients_in_model(model, writer, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            writer.add_histogram(tag + "/grad", value.grad.cpu(), step)


def cal_psnr(im1, im2):
    im1_255 = im1 * 255
    im2_255 = im2 * 255

    mse = torch.mean((torch.abs(im1_255 - im2_255) ** 2))
    psnr = 10 * torch.log10(255 * 255 / mse)
    return psnr


def cal_ssim(im1, im2):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = torch.mean(im1)
    mu2 = torch.mean(im2)
    sigma1 = torch.sqrt(torch.mean((im1 - mu1) ** 2))
    sigma2 = torch.sqrt(torch.mean((im2 - mu2) ** 2))
    sigma12 = torch.mean(((im1 - mu1) * (im2 - mu2)))

    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2

    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim


def train(config):
    writer = SummaryWriter('./data/board/case_1600_100_5_0point1/')

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    scale_factor = config.scale_factor
    DCE_net = model.enhance_net_nopool(scale_factor).cpu()

    # DCE_net.apply(weights_init)
    if config.load_pretrain:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir, map_location=torch.device('cpu')))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16)
    L_TV = Myloss.L_TV()

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    DCE_net.train()
    loss_idx_value = 0
    for epoch in range(config.num_epochs):
        psns_mean = []
        for iteration, img_lowlight in enumerate(train_loader):

            img_lowlight = img_lowlight.cpu()

            E = 0.6

            enhanced_image, A = DCE_net(img_lowlight)

            # Illumination Smoothness Loss
            Loss_TV = 1600 * L_TV(A)

            # Spatial Consistency Loss
            loss_spa = 100 * torch.mean(L_spa(enhanced_image, img_lowlight))

            # Color Constancy Loss
            loss_col = 5 * torch.mean(L_color(enhanced_image))

            # Exposure Control Loss
            loss_exp = 0.1 * torch.mean(L_exp(enhanced_image, E))

            psns_mean.append(cal_psnr(img_lowlight, enhanced_image).item())
            # ssim = cal_ssim(img_lowlight, enhanced_image)

            # best_loss
            loss = Loss_TV + loss_spa + loss_col + loss_exp
            writer.add_scalar('loss/Loss_TV', Loss_TV.item(), loss_idx_value)
            writer.add_scalar('loss/loss_spa', loss_spa.item(), loss_idx_value)
            writer.add_scalar('loss/loss_col', loss_col.item(), loss_idx_value)
            writer.add_scalar('loss/loss_exp', loss_exp.item(), loss_idx_value)
            writer.add_scalar('loss/loss_ALL', loss.item(), loss_idx_value)
            loss_idx_value += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            log_gradients_in_model(DCE_net, writer, iteration)

            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, "epoch", str(epoch), ":", loss.item())

            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

        writer.add_scalar('evaluation/psns', np.array(psns_mean).mean(), epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    # parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lowlight_images_path', type=str, default="data/small_100_train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots_Zero_DCE++/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots_Zero_DCE++/Epoch99.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
