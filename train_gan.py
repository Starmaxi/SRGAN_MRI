 
"""
Super-resolution of CelebA using Generative Adversarial Networks with Mixed Precision.
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys
from tqdm import tqdm

from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# Argumente
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=30, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()
hr_shape = (opt.hr_height, opt.hr_width)

# Initialize models
generator = GeneratorDensNet(in_channels=1, out_channels=1)
#generator = GeneratorResNet(in_channels=1, out_channels=1)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor().eval()  # Inference mode

# Disable gradients for feature extractor
for param in feature_extractor.parameters():
    param.requires_grad = False

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator.cuda()
    discriminator.cuda()
    feature_extractor.cuda()
    criterion_GAN.cuda()
    criterion_content.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Mixed Precision
scaler = GradScaler(enabled=cuda)  # Nur aktivieren, wenn CUDA verfügbar ist
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# DataLoader
dataloader = DataLoader(
    ImageDataset("/mnt/1tb_ssd/Dokumente/Schule/THI/projekt_2/out/joint_data/", hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    pin_memory=True
)


# Training
for epoch in tqdm(range(opt.epoch, opt.n_epochs)):
    for i, imgs in tqdm(enumerate(dataloader)):
        imgs_lr = Variable(imgs["lr"].type(Tensor)).cuda(non_blocking=True)
        imgs_hr = Variable(imgs["hr"].type(Tensor)).cuda(non_blocking=True)
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generator (mit Mixed Precision)
        # ------------------
        optimizer_G.zero_grad()

        with autocast(enabled=cuda):  # Automatische Typumwandlung zu float16
            gen_hr = generator(imgs_lr[:, 0:1, :, :])
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

            # Content loss (Feature Extractor benötigt float32)
            gen_features = feature_extractor(gen_hr.expand(-1, 3, -1, -1))
            real_features = feature_extractor(imgs_hr.expand(-1, 3, -1, -1))
            loss_content = criterion_content(gen_features, real_features.detach())

            loss_G = loss_content + 1e-3 * loss_GAN

        # Gradient Scaling für Mixed Precision
        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()

        # ---------------------
        #  Train Discriminator (mit Mixed Precision)
        # ---------------------
        optimizer_D.zero_grad()

        with autocast(enabled=cuda):
            loss_real = criterion_GAN(discriminator(imgs_hr[:, 0:1, :, :]), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()[:, 0:1, :, :]), fake)
            loss_D = (loss_real + loss_fake) / 2

        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        scaler.update()

        # Logging
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    # Model speichern
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
