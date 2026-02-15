import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import matplotlib.pyplot as plt

def generate_and_save_phantom(size=256, filename="data/phantom.bin"):
    # 1. Generate 2D Phantom
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (size, size)).astype(np.float32)
    
    # 2. Save as raw binary (C++ friendly)
    phantom.tofile(filename)
    print(f"Phantom saved to {filename} (Size: {size}x{size})")

def visualize_binary(filename, size=256, title="Image"):
    data = np.fromfile(filename, dtype=np.float32).reshape((size, size))
    plt.imshow(data, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    import os
    if not os.path.exists('data'): os.makedirs('data')
    generate_and_save_phantom(256)
    # visualize_binary("data/phantom.bin", 256, "Ground Truth")