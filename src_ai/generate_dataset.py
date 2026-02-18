import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse
import os
import time
from datetime import timedelta

# CONFIG
NUM_SAMPLES = 10
IMG_SIZE = 256
OUTPUT_DIR = "data/raw_phantoms"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def create_random_phantom(size=256):
    img = np.zeros((size, size), dtype=np.float32)
    
    # 1. Main outer shell
    rr, cc = ellipse(size//2, size//2, size//2.2, size//3, shape=img.shape)
    img[rr, cc] = 1.0
    
    # 2. Add 5-10 random internal features (Tumors/Organs)
    num_features = np.random.randint(5, 10)
    for _ in range(num_features):
        r = np.random.randint(size//4, 3*size//4)
        c = np.random.randint(size//4, 3*size//4)
        r_radius = np.random.randint(5, size//5)
        c_radius = np.random.randint(5, size//5)
        intensity = np.random.uniform(0.2, 0.8)
        
        # Draw ellipse
        rr, cc = ellipse(r, c, r_radius, c_radius, shape=img.shape)
        
        # Randomly add or subtract intensity
        if np.random.rand() > 0.5:
            img[rr, cc] += intensity
        else:
            img[rr, cc] -= intensity
            
    # Clip values 
    img = np.clip(img, 0, 2.0)
    return img

start_time = time.perf_counter()

print(f"Generating {NUM_SAMPLES} random phantoms...")

for i in range(NUM_SAMPLES):
    phantom = create_random_phantom(IMG_SIZE)
    filename = os.path.join(OUTPUT_DIR, f"{i:03d}.bin")
    phantom.tofile(filename)

print("Done! Example:")
plt.imshow(create_random_phantom(), cmap='gray')
plt.title("Sample Random Phantom")
plt.show()

elapsed = time.perf_counter() - start_time
print(f"Total execution time: {elapsed:.3f} seconds ({timedelta(seconds=elapsed)})")