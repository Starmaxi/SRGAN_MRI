import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

HR_DIR = 'axial_volumes_npy/high_resolution'
GT_DIR = 'axial_volumes_npy/ground_truth'
SAVE_DIR = 'fused_volumes_npy'
os.makedirs(SAVE_DIR, exist_ok=True)

epochs = 5
batch_size = 1
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class VolumeDataset(Dataset):
    def __init__(self, hr_dir, gt_dir):
        self.filenames = [f for f in os.listdir(hr_dir) if f.endswith('.npy')]
        self.hr_dir = hr_dir
        self.gt_dir = gt_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        vol_hr = np.load(os.path.join(self.hr_dir, fname))  # shape: (D, H, W)
        vol_gt = np.load(os.path.join(self.gt_dir, fname))  # shape: (D, H, W)
        stacked = np.stack([vol_hr, vol_gt], axis=0)        # shape: (2, D, H, W)
        return torch.tensor(stacked, dtype=torch.float32), fname

class VolumeFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(8, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = VolumeFusionNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
dataset = VolumeDataset(HR_DIR, GT_DIR)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.train()
for epoch in range(epochs):
    total_loss = 0
    for x, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        x = x.to(device)
        target = x[:, 0:1]  # HR as pseudo-target
        output = model(x)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

print("Saving fused volumes...")
model.eval()
with torch.no_grad():
    for x, fname in tqdm(dataset):
        x = x.unsqueeze(0).to(device)
        output = model(x).squeeze().cpu().numpy()
        output = (output - output.min()) / (output.max() - output.min() + 1e-8)
        np.save(os.path.join(SAVE_DIR, fname.replace('.npy', '_fused.npy')), output)

