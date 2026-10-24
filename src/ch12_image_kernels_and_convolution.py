"""
Chapter 12: Computer Vision: Spatial Convolution Kernels from Scratch
Modern Machine Learning & Data Science Engineering
"""
import numpy as np

def convolve2d(image, kernel):
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    out_h = i_h - k_h + 1
    out_w = i_w - k_w + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(patch * kernel)
    return output

def max_pool2d(feature_map, pool_size=2, stride=2):
    h, w = feature_map.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            r_start = i * stride
            c_start = j * stride
            patch = feature_map[r_start:r_start+pool_size, c_start:c_start+pool_size]
            output[i, j] = np.max(patch)
    return output

def demonstrate_image_kernels():
    print("=====================================================================")
    print("   COMPUTER VISION: 2D CONVOLUTION & MAX POOLING MECHANICS           ")
    print("=====================================================================")

    # Create a 6x6 synthetic image with a sharp vertical edge in the center
    image = np.array([
        [10, 10, 10, 90, 90, 90],
        [10, 10, 10, 90, 90, 90],
        [10, 10, 10, 90, 90, 90],
        [10, 10, 10, 90, 90, 90],
        [10, 10, 10, 90, 90, 90],
        [10, 10, 10, 90, 90, 90],
    ], dtype=float)

    print("Original 6x6 Grayscale Image (Step Edge at Column 3):\n", image.astype(int))

    # Sobel Vertical Edge Detection Filter (3x3)
    sobel_vertical = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)

    # Gaussian Blur Smoothing Filter (3x3)
    gaussian_blur = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=float) / 16.0

    print("\n--- 1. Applying Vertical Sobel Edge Filter ---")
    edge_map = convolve2d(image, sobel_vertical)
    print("Edge Feature Map (Dimensions 4x4):\n", edge_map.astype(int))
    print("Note: Peak activations appear exactly where the vertical boundary sits!")

    print("\n--- 2. Applying 2x2 Max Pooling (Downsampling) ---")
    pooled = max_pool2d(edge_map, pool_size=2, stride=2)
    print("Max-Pooled Map (Dimensions 2x2):\n", pooled.astype(int))

if __name__ == "__main__":
    demonstrate_image_kernels()
