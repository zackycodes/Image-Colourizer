import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2

# ---------------------------
# 🔹 1. Load & Preprocess Dataset
# ---------------------------
common_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

grayscale_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

DATASET_SIZE = 10000
class GrayscaleColorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return min(DATASET_SIZE, len(self.dataset))

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        color = common_transform(img)
        grayscale = grayscale_transform(img)
        return grayscale, color

cifar_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
dataset = GrayscaleColorDataset(cifar_dataset)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

grayscale_sample, color_sample = next(iter(dataloader))

print(f"✅ Grayscale shape: {grayscale_sample.shape}")
print(f"✅ Color shape: {color_sample.shape}")

# ---------------------------
# 🔹 2. Define Model: Improved U-Net Style Encoder-Decoder
# ---------------------------

class UNetColorization(nn.Module):
    def __init__(self):
        super(UNetColorization, self).__init__()

        self.encoder = nn.Sequential( 
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):  # forward function
        x = self.encoder(x) 
        x = self.decoder(x)
        return x

# ---------------------------
# 🔹 3. Training Setup
# ---------------------------
device = torch.device("cuda") # force cuda usage
model = UNetColorization().to(device) # UNet model
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0005) # optimizr :D

# ---------------------------
# 🔹 4. Training Loop
# ---------------------------

num_epochs = 100 # ~30 m 100

for epoch in range(num_epochs):
    total_loss = 0 # Aim
    for grayscale, color in dataloader:
        grayscale, color = grayscale.to(device), color.to(device)

        optimizer.zero_grad() 
        output = model(grayscale) 
        assert output.shape == color.shape, f"❌ Shape Mismatch: {output.shape} vs {color.shape}"

        loss = criterion(output, color) # loss
        loss.backward() # go back
        optimizer.step() # step here
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}") # out of loop; epoch done

# ---------------------------
# 🔹 5. Save Model
# ---------------------------

torch.save(model.state_dict(), "unet_colorization.pth")
print("✅ Model saved!")


# ---------------------------
# 🔹 6. Test Model on Unseen Data
# ---------------------------

model.eval()

with torch.no_grad():
    grayscale, color = next(iter(dataloader))
    grayscale = grayscale.to(device)
    output = model(grayscale)

    fig, axes = plt.subplots(3, 8, figsize=(12, 6)) #     fig, axes = plt.subplots(3, 8, figsize=(12, 6)) n, m , L, K |k

# plot:
    for i in range(8):
        axes[0, i].imshow(grayscale[i].squeeze().cpu(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(output[i].permute(1, 2, 0).cpu())
        axes[1, i].axis("off")
        axes[2, i].imshow(color[i].permute(1, 2, 0).cpu())
        axes[2, i].axis("off")

    axes[0, 0].set_title("Grayscale Input")
    axes[1, 0].set_title("Predicted Colorization")
    axes[2, 0].set_title("Ground Truth")
    plt.show()