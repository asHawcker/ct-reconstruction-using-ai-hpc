import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def view_bin(filename, size=256):
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return

    # Load raw float32 data
    try:
        data = np.fromfile(filename, dtype=np.float32)
        
        # Check if the file size matches the expected dimensions
        expected_len = size * size
        if data.size != expected_len:
            print(f"Error: File size mismatch.")
            print(f"Expected {expected_len} floats ({size}x{size}), but got {data.size}.")
            print("Did you change the image size in C++ without changing it here?")
            return

        # Reshape to 2D image
        image = data.reshape((size, size))

        # Plot
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.title(f"Visualizing: {filename}")
        plt.colorbar(label='Intensity')
        plt.axis('off') # Hide axes ticks
        plt.show()

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 visualize.py <path_to_bin_file> [size]")
        print("Example: python3 visualize.py data/sirt_iter50.bin")
    else:
        file_path = sys.argv[1]
        img_size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        view_bin(file_path, img_size)