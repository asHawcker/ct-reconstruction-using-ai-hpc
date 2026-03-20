import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time
import os
from skimage.draw import ellipse
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# CONFIGURATION 
SOLVER_EXE = "/content/sirt_solver"
MODEL_PATH = "/content/sirt_unet_model_ssim.pth"
IMG_SIZE = 256

# 1. MODEL DEFINITION
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

# 2. PHANTOM GENERATORS
def get_shepp_logan():
    """Returns a 256x256 Shepp-Logan phantom as float32."""
    img = shepp_logan_phantom()
    img = resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    return img

def generate_random_phantom():
    """Generates a random anatomical-like phantom."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    
    # Outer shell with slight randomization
    center_r = IMG_SIZE // 2 + np.random.randint(-10, 10)
    center_c = IMG_SIZE // 2 + np.random.randint(-10, 10)
    radius_r = IMG_SIZE // 2.2 + np.random.randint(-5, 15)
    radius_c = IMG_SIZE // 3.0 + np.random.randint(-5, 15)
    rr, cc = ellipse(center_r, center_c, radius_r, radius_c, shape=img.shape)
    img[rr, cc] = 1.0 
    
    # Random internal features
    num_features = np.random.randint(6, 12)
    for _ in range(num_features):
        r = np.random.randint(IMG_SIZE // 4, 3 * IMG_SIZE // 4)
        c = np.random.randint(IMG_SIZE // 4, 3 * IMG_SIZE // 4)
        r_radius = np.random.randint(5, IMG_SIZE // 6)
        c_radius = np.random.randint(5, IMG_SIZE // 6)
        intensity = np.random.uniform(0.2, 0.8)
        
        rr, cc = ellipse(r, c, r_radius, c_radius, shape=img.shape)
        if np.random.rand() > 0.4:
            img[rr, cc] += intensity
        else:
            img[rr, cc] -= intensity
            
    img = np.clip(img, 0.0, 2.0)
    return img

# 3. BENCHMARKING 
def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading AI Model on {device}...")
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Define 4 test cases
    test_cases = [
        ("Shepp-Logan", get_shepp_logan()),
        ("Random Phantom 1", generate_random_phantom()),
        ("Random Phantom 2", generate_random_phantom()),
        ("Random Phantom 3", generate_random_phantom())
    ]

    test_phantom_file = "test_phantom.bin"
    out_iter50_file = "test_iter50.bin"
    out_iter10_file = "test_iter10.bin"
    
    results = [] 

    for name, phantom_data in test_cases:
        print(f"\n" + "="*50)
        print(f"Evaluating: {name}")
        print("="*50)
        
        phantom_data.tofile(test_phantom_file)

        # A. TRADITIONAL HPC (50 Iterations) 
        t0_trad = time.time()
        subprocess.run([SOLVER_EXE, test_phantom_file, out_iter50_file, "50"], check=True)
        t1_trad = time.time()
        time_traditional = t1_trad - t0_trad
        print(f"Traditional HPC (50 Iters) Time : {time_traditional:.4f} s")

        # B. HYBRID PIPELINE (10 Iters + AI) 
        # 1. HPC part
        t0_hybrid = time.time()
        subprocess.run([SOLVER_EXE, test_phantom_file, out_iter10_file, "10"], check=True)
        t1_hybrid_hpc = time.time()
        time_hpc_part = t1_hybrid_hpc - t0_hybrid
        
        # 2. PyTorch Data Prep
        iter10_np = np.fromfile(out_iter10_file, dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)
        min_val, max_val = iter10_np.min(), iter10_np.max()
        if max_val - min_val > 1e-5:
            iter10_norm = (iter10_np - min_val) / (max_val - min_val)
        else:
            iter10_norm = np.zeros_like(iter10_np)

        input_tensor = torch.tensor(iter10_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        
        # 3. AI Inference
        with torch.no_grad():
            prediction_tensor = model(input_tensor)
        t1_hybrid_ai = time.time()
        
        time_ai_part = t1_hybrid_ai - t1_hybrid_hpc
        time_hybrid_total = time_hpc_part + time_ai_part
        speedup = time_traditional / time_hybrid_total
        
        print(f"Hybrid HPC Part (10 Iters) Time : {time_hpc_part:.4f} s")
        print(f"Hybrid AI Inference Time        : {time_ai_part:.4f} s")
        print(f"Total Hybrid Time               : {time_hybrid_total:.4f} s")
        print(f"Speedup for {name}              : {speedup:.2f}x Faster")

        # Load final 50 iter for plotting
        iter50_np = np.fromfile(out_iter50_file, dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)
        prediction_np = prediction_tensor.cpu().squeeze().numpy()

        # Store data for plotting
        results.append({
            "name": name,
            "ground_truth": phantom_data,
            "iter50": iter50_np,
            "iter10": iter10_norm,
            "prediction": prediction_np,
            "time_trad": time_traditional,
            "time_hybrid": time_hybrid_total,
            "speedup": speedup
        })

    # --- 4. PLOT ---
    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    fig.suptitle("AI-Accelerated SIRT Benchmarks", fontsize=20, y=0.98)

    for row, res in enumerate(results):
        # Ground Truth
        ax = axes[row, 0]
        ax.imshow(res["ground_truth"], cmap='gray')
        ax.set_title(f"{res['name']}\nGround Truth", fontsize=12)
        ax.axis('off')

        # Traditional
        ax = axes[row, 1]
        ax.imshow(res["iter50"], cmap='gray')
        ax.set_title(f"HPC (50 Iters)\nTime: {res['time_trad']:.2f}s", fontsize=12)
        ax.axis('off')

        # Hybrid Input
        ax = axes[row, 2]
        ax.imshow(res["iter10"], cmap='gray')
        ax.set_title(f"AI Input (10 Iters)", fontsize=12)
        ax.axis('off')

        # Hybrid Output
        ax = axes[row, 3]
        ax.imshow(res["prediction"], cmap='gray')
        ax.set_title(f"AI Prediction\nTotal Time: {res['time_hybrid']:.2f}s | Speedup: {res['speedup']:.1f}x", fontsize=12, color='green')
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

if __name__ == "__main__":
    run_benchmark()