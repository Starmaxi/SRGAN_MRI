import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from datasets import *
#import cv2
import math
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import vgg19
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision
from models import CoordRegressionNet

# Beispiel: Tensor erstellen und Formen zeichnen
h, w = 256, 256
#tensor = torch.zeros((h, w))  # Leerer Tensor (Höhe x Breite)

cuda = torch.cuda.is_available()

# Mixed Precision
scaler = GradScaler(enabled=cuda)  # Nur aktivieren, wenn CUDA verfügbar ist
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss()

model = CoordRegressionNet()
model.train()
optimizer = optim.Adam(model.parameters(), 0.001)

criterion1.cuda()
criterion2.cuda()
model.cuda()

normalize = transforms.Normalize(mean=[0.5], std=[0.5])

dataloader = DataLoader(
    CircleDataset(256,50000),
    batch_size=6,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    #prefetch_factor=1000
)

loss_file = open("circle_loss_file.txt", "w")
# Training

for i, data in tqdm(enumerate(dataloader)):
    with autocast(enabled=cuda):
        input_data = data["img"].cuda(non_blocking=True)
        #input_data = data["img"]
        pred = model(input_data)
        y = data["y"].cuda(non_blocking=True)
        #print(y*256)
        #print(pred*256)
        loss = criterion1(pred, y) + 0.1 * criterion2(pred, y)
        #input()
        #plt.imshow(input_data[0][0].cpu().detach(), cmap="gray")
        #plt.show()
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        loss_file.write(str(loss.item())+"\n")

        #plt.imshow(input_data.squeeze(0).squeeze(0).squeeze(0).cpu(), cmap='gray')
        #plt.show()

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


loss_file.close()
torch.save(model.state_dict(), "model_outputs.pth")
