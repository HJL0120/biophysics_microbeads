
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift
from scipy.stats import pearsonr
from scipy.ndimage import shift as ndi_shift

def phase_correlation(ref, target):
    F1 = fft2(ref)
    F2 = fft2(target)
    R = F1 * F2.conj()
    R /= np.abs(R) + 1e-8
    r = fftshift(ifft2(R).real)
    max_idx = np.unravel_index(np.argmax(r), r.shape)
    shifts = np.array(max_idx) - np.array(ref.shape) // 2
    return shifts

def high_pass_filter(img_fft, threshold_ratio=0.005):
    h, w = img_fft.shape
    cx, cy = w // 2, h // 2
    radius = int(min(cx, cy) * threshold_ratio)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = dist_from_center >= radius
    return img_fft * mask

def extract_distance(fname):
    match = re.search(r'-(\d+)nm', fname)
    return int(match.group(1)) if match else None

# Load images
base_path = "./"
files = sorted([f for f in os.listdir(base_path) if f.endswith(".png")])
ref_filename = "MovingBead-0_-35nm.png"
ref_distance = -35

shapes = [Image.open(os.path.join(base_path, f)).convert("L").size[::-1] for f in files]
min_shape = (min(s[0] for s in shapes), min(s[1] for s in shapes))

ref_img = Image.open(os.path.join(base_path, ref_filename)).convert("L")
ref_img_cropped = np.array(ref_img)[:min_shape[0], :min_shape[1]]
ref_fft_filtered = high_pass_filter(fftshift(fft2(ref_img_cropped)))

fft_corr_results = []
real_corr_results = []

for fname in files:
    distance_from_focal = extract_distance(fname)
    if distance_from_focal is None:
        continue

    img = Image.open(os.path.join(base_path, fname)).convert("L")
    img_cropped = np.array(img)[:min_shape[0], :min_shape[1]]

    img_fft_filtered = high_pass_filter(fftshift(fft2(img_cropped)))
    r_fft, _ = pearsonr(ref_fft_filtered.flatten().real, img_fft_filtered.flatten().real)

    shift_vec = phase_correlation(ref_img_cropped, img_cropped)
    aligned_img = ndi_shift(img_cropped, shift_vec, mode='nearest')
    aligned_fft_filtered = high_pass_filter(fftshift(fft2(aligned_img)))
    filtered_real_ref = ifft2(ref_fft_filtered).real
    filtered_real_aligned = ifft2(aligned_fft_filtered).real
    r_real_filtered, _ = pearsonr(filtered_real_ref.flatten(), filtered_real_aligned.flatten())

    dist_from_ref = abs(distance_from_focal - ref_distance)
    fft_corr_results.append((dist_from_ref, r_fft))
    real_corr_results.append((dist_from_ref, r_real_filtered))

df_fft = pd.DataFrame(sorted(fft_corr_results), columns=["Distance from Reference (nm)", "FFT Pearson r"])
df_real = pd.DataFrame(sorted(real_corr_results), columns=["Distance from Reference (nm)", "Filtered Real Pearson r"])

plt.figure(figsize=(10, 6))
plt.plot(df_fft["Distance from Reference (nm)"], df_fft["FFT Pearson r"], marker='o', label="FFT Domain")
plt.plot(df_real["Distance from Reference (nm)"], df_real["Filtered Real Pearson r"], marker='s', label="Filtered Real (Aligned)", color='green')
plt.title("Correlation vs. Distance from Reference Bead")
plt.xlabel("Distance from Reference (nm)")
plt.ylabel("Pearson Correlation Coefficient")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
