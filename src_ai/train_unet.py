import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# --- 1. CONFIGURATION ---
INPUT_DIR = '/content/drive/MyDrive/hpc_dataset/input_iter10'
TARGET_DIR = '/content/drive/MyDrive/hpc_dataset/target_iter50'
IMG_SIZE = 256
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 50

# --- 2. CUSTOM DATASET CLASS ---
class SirtDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_files = sorted(glob.glob(os.path.join(input_dir, "*.bin")))
        self.target_files = sorted(glob.glob(os.path.join(target_dir, "*.bin")))
        print(f"Found {len(self.input_files)} input files and {len(self.target_files)} target files.")
        
        if len(self.input_files) == 0:
            raise RuntimeError(f"No .bin files found in {input_dir}")
        if len(self.input_files) != len(self.target_files):
            raise RuntimeError("Mismatch between number of input and target files!")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load raw binary
        in_path = self.input_files[idx]
        tar_path = self.target_files[idx]

        input_img = np.fromfile(in_path, dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)
        target_img = np.fromfile(tar_path, dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)

        # Normalization (Min-Max)
        min_val = input_img.min()
        max_val = input_img.max()
        if max_val - min_val > 1e-5:
            input_img = (input_img - min_val) / (max_val - min_val)
            target_img = (target_img - min_val) / (max_val - min_val)
        else:
            input_img = np.zeros_like(input_img)
            target_img = np.zeros_like(target_img)

        # Add Channel Dimension 
        input_tensor = torch.tensor(input_img).unsqueeze(0).float()
        target_tensor = torch.tensor(target_img).unsqueeze(0).float()

        return input_tensor, target_tensor

# --- 3. MODEL: SIMPLE U-NET ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(96, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        d1 = self.up(e2)
        d1 = torch.cat([d1, e1], dim=1)
        out = self.dec1(d1)
        return self.final(out)

# --- 4. TRAINING & VALIDATION LOOP ---
def train():
    # A. Prepare Data
    full_dataset = SirtDataset(INPUT_DIR, TARGET_DIR)

    # Split: 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) 
    print(f"Dataset: {len(full_dataset)} images.")
    print(f"Training on {train_size}, Validating on {val_size}.")

    # B. Setup Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    # C. Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # --- 5. VISUALIZATION ---
    print("\nTraining Complete. Visualizing a random validation sample...")

    # Pick the first image from validation loader
    inputs, targets = next(iter(val_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.no_grad():
        prediction = model(inputs)

    # Move to CPU for plotting
    input_np = inputs.cpu().squeeze().numpy()
    target_np = targets.cpu().squeeze().numpy()
    pred_np = prediction.cpu().squeeze().numpy()

    plt.figure(figsize=(15, 5))

    # 1. Input (Blurry)
    plt.subplot(1, 4, 1)
    plt.title("Input (Iter 10)\n[Unseen Data]")
    plt.imshow(input_np, cmap='gray')
    plt.axis('off')

    # 2. AI Prediction
    plt.subplot(1, 4, 2)
    plt.title("AI Prediction\n[Accelerated]")
    plt.imshow(pred_np, cmap='gray')
    plt.axis('off')

    # 3. Target (Ground Truth)
    plt.subplot(1, 4, 3)
    plt.title("Target (Iter 50)\n[HPC Ground Truth]")
    plt.imshow(target_np, cmap='gray')
    plt.axis('off')

    # 4. Error Map
    plt.subplot(1, 4, 4)
    plt.title("Error Map")
    plt.imshow(np.abs(target_np - pred_np), cmap='hot')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save the model
    torch.save(model.state_dict(), "sirt_unet_model.pth")
    print("Model saved to sirt_unet_model.pth")

if __name__ == "__main__":
    train()