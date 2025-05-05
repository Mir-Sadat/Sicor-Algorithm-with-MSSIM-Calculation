# --- Install necessary packages ---
!apt-get install -y libgdal-dev
!pip install rasterio h5py scikit-learn pyproj numpy scipy scikit-image

# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Set your Google Drive paths here ---
safe_folder = '/content/drive/MyDrive/S2B.SAFE'

# --- Check if the folder exists ---
import os
if not os.path.exists(safe_folder):
    raise ValueError(f"SAFE folder not found at {safe_folder}!")

print(f"✅ Found SAFE folder: {safe_folder}")

# --- Locate the IMG_DATA folder inside GRANULE ---
import glob

granule_folders = glob.glob(os.path.join(safe_folder, 'GRANULE', '*'))
if len(granule_folders) == 0:
    raise ValueError("No GRANULE sub-folder found inside SAFE folder!")

granule_folder = granule_folders[0]
print(f"✅ Found GRANULE folder: {granule_folder}")

img_data_folder = os.path.join(granule_folder, 'IMG_DATA')
print(f"✅ Found IMG_DATA folder: {img_data_folder}")

# --- Import libraries ---
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import random

# --- Helper Function to Display Sentinel-2 image ---
def plot_rgb(img_array, bands=(3, 2, 1), title="", stretch=True):
    """
    img_array: (bands, height, width)
    bands: which bands to use for RGB display (default B04, B03, B02 -> R,G,B)
    """
    rgb = np.stack([img_array[b-1] for b in bands], axis=-1)
    if stretch:
        rgb = (rgb - np.percentile(rgb, 2)) / (np.percentile(rgb, 98) - np.percentile(rgb, 2))
        rgb = np.clip(rgb, 0, 1)
    plt.figure(figsize=(8,8))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- Load Bands B02 (Blue), B03 (Green), B04 (Red) ---
jp2_files = []
for band in ['B02', 'B03', 'B04']:
    files = glob.glob(os.path.join(img_data_folder, f"*{band}_10m.jp2"))
    if len(files) == 0:
        raise ValueError(f"No {band} band found in IMG_DATA folder!")
    jp2_files.append(files[0])

print("✅ Found bands:", jp2_files)

# --- Read bands ---
bands = []
for jp2 in jp2_files:
    with rasterio.open(jp2) as src:
        bands.append(src.read(1).astype(np.float32))

bands = np.stack(bands)
print(f"✅ Bands loaded with shape: {bands.shape}")

# --- Plot the original L1C (Top-of-Atmosphere reflectance) ---
plot_rgb(bands, bands=(3,2,1), title="Original L1C (Top-of-Atmosphere)")

# --- Atmospheric Correction (Simulated) ---
print("⚡ Simulating Sen2Cor atmospheric correction...")

corrected_bands = bands / 10000.0
corrected_bands = np.clip(corrected_bands, 0, 1)

# --- Plot the corrected L2A (Surface Reflectance) ---
plot_rgb(corrected_bands, bands=(3,2,1), title="Corrected L2A (Surface Reflectance - Simulated)")

print("✅ Atmospheric correction (simulated) done!")

# --- Normalize images for MSSIM ---
def normalize(img):
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)

original_rgb = np.stack([bands[b-1] for b in (3,2,1)], axis=-1)  # (H, W, 3)
corrected_rgb = np.stack([corrected_bands[b-1] for b in (3,2,1)], axis=-1)

original_rgb_norm = normalize(original_rgb)
corrected_rgb_norm = normalize(corrected_rgb)

# --- Patch-wise MSSIM computation ---

patch_size = 512
height, width, _ = original_rgb_norm.shape

# How many patches possible?
patch_rows = height // patch_size
patch_cols = width // patch_size
print(f"✅ Image can be divided into {patch_rows} x {patch_cols} patches")

# Randomly select 10 patch coordinates
random.seed(42)  # reproducibility
patch_indices = [(i,j) for i in range(patch_rows) for j in range(patch_cols)]
selected_patches = random.sample(patch_indices, 10)

mssim_values = []

for idx, (i, j) in enumerate(selected_patches):
    y_start = i * patch_size
    x_start = j * patch_size

    orig_patch = original_rgb_norm[y_start:y_start+patch_size, x_start:x_start+patch_size, :]
    corr_patch = corrected_rgb_norm[y_start:y_start+patch_size, x_start:x_start+patch_size, :]

    # Compute MSSIM for each channel and average
    ssim_per_channel = []
    for k in range(3):  # RGB channels
        ssim_val, _ = ssim(orig_patch[:,:,k], corr_patch[:,:,k], data_range=1.0, full=True)
        ssim_per_channel.append(ssim_val)
    patch_mssim = np.mean(ssim_per_channel)
    mssim_values.append(patch_mssim)

    # Plot side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].imshow(orig_patch)
    axs[0].set_title('Original Patch')
    axs[0].axis('off')
    axs[1].imshow(corr_patch)
    axs[1].set_title(f'Corrected Patch\nMSSIM: {patch_mssim:.4f}')
    axs[1].axis('off')
    plt.suptitle(f"Patch {idx+1}")
    plt.show()

# --- Summary ---
mean_mssim = np.mean(mssim_values)
print(f"🔍 Average MSSIM over 10 patches: {mean_mssim:.4f}")
