#!/usr/bin/env python3
"""
Download MNIST dataset and save a sample of 10 images as transparent PNGs.
Uses torchvision - no TensorFlow required.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import datasets
import torchvision.transforms as transforms

def download_mnist_transparent_sample(output_dir="mnist_transparent", num_samples=10):
    """
    Download MNIST dataset and save sample images as transparent PNGs.
    
    Args:
        output_dir: Directory to save the transparent images
        num_samples: Number of sample images to save
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Downloading MNIST dataset...")
    # Download MNIST dataset using torchvision
    mnist_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True, 
                                   transform=transforms.ToTensor())
    
    print(f"Converting and saving {num_samples} sample images as transparent PNGs...")
    
    # Save sample images from training set
    for i in range(num_samples):
        # Get image tensor and label
        img_tensor, label = mnist_dataset[i]
        
        # Convert tensor to numpy array and denormalize to 0-255
        img_array = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
        
        # Convert grayscale to RGBA
        img_rgba = Image.new('RGBA', (28, 28), (255, 255, 255, 0))  # Start with transparent
        pixels = img_rgba.load()
        
        # Set pixels: higher intensity (darker) = more opaque
        for y in range(28):
            for x in range(28):
                intensity = img_array[y, x]
                # intensity: 0=white (transparent), 255=black (opaque)
                # Alpha: 0=transparent, 255=opaque
                alpha = intensity
                pixels[x, y] = (0, 0, 0, alpha)  # Black digit with varying transparency
        
        # Save as PNG
        output_path = output_dir / f"mnist_{i:02d}_label_{label}.png"
        img_rgba.save(output_path)
        print(f"  ✓ Saved {output_path.name}")
    
    print(f"\n✓ Successfully saved {num_samples} transparent MNIST images to '{output_dir}'")

def add_noise_to_images(input_dir="mnist_transparent", output_dir="mnist_transparent_noisy", noise_amount=0.3):
    """
    Add Gaussian noise to transparent MNIST images and save to a new folder.
    
    Args:
        input_dir: Directory with original transparent images
        output_dir: Directory to save noisy images
        noise_amount: Standard deviation of Gaussian noise (0-1 scale)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' not found!")
        return
    
    print(f"Adding noise to images from '{input_dir}'...")
    
    # Get all PNG files
    png_files = sorted(input_dir.glob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in '{input_dir}'")
        return
    
    for img_path in png_files:
        # Open image
        img_rgba = Image.open(img_path).convert('RGBA')
        
        # Convert to numpy array
        img_array = np.array(img_rgba)
        
        # Extract alpha channel
        alpha_channel = img_array[:, :, 3]
        
        # Add Gaussian noise to the alpha channel (based on noise_amount)
        noise = np.random.normal(0, noise_amount * 255, alpha_channel.shape)
        noisy_alpha = np.clip(alpha_channel.astype(float) + noise, 0, 255).astype(np.uint8)
        
        # Create new RGBA image with noisy alpha
        noisy_array = img_array.copy()
        noisy_array[:, :, 3] = noisy_alpha
        
        # Convert back to image
        noisy_img = Image.fromarray(noisy_array, 'RGBA')
        
        # Save noisy image
        output_path = output_dir / img_path.name
        noisy_img.save(output_path)
        print(f"  ✓ Saved {output_path.name}")
    
    print(f"\n✓ Successfully saved {len(png_files)} noisy transparent MNIST images to '{output_dir}'")

if __name__ == "__main__":
    download_mnist_transparent_sample()
    add_noise_to_images()
