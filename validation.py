import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original: np.ndarray, compared: np.ndarray, max_pixel_value: float = 255.0) -> float:
    """
    Calculates the PSNR (Peak Signal-to-Noise Ratio) between two NumPy arrays.

    :param original: The original image/array.
    :param compared: The image/array to compare (e.g., reconstructed or compressed).
    :param max_pixel_value: Maximum possible pixel value (e.g., 255 for 8-bit images).
    :return: PSNR value in dB.
    """

    mse = np.mean((original - compared) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

def show_slices_side_by_side(vol1, vol2, slice_idx=0, title1="Original", title2="Vergleich"):
    """Zeigt zwei Slices aus 3D-Volumes nebeneinander an."""
    
    # Optional: log1p anwenden, wenn es FFT-Amplituden sind
    v1 = np.log1p(vol1[slice_idx])
    v2 = np.log1p(vol2[slice_idx])
    magnitude = np.log1p(vol1[slice_idx])  # log(1 + x) verhindert log(0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(v1, cmap='gray')
    axes[0].set_title(f"{title1} (Slice {slice_idx})")
    axes[0].axis('off')

    axes[1].imshow(v2, cmap='gray')
    axes[1].set_title(f"{title2} (Slice {slice_idx})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude, cmap='gray')
    plt.title(f"Slice {slice_idx} (log-skaliert)")
    plt.axis('off')
    plt.colorbar()
    plt.show()


def calculate_ssim(original: np.ndarray, compared: np.ndarray, slice_idx: int = 80) -> float:
    """
    Berechnet den SSIM-Wert (Structural Similarity Index) für zwei MRI-Volumes auf einem bestimmten Slice.
    
    :param original: Originalbild-Volume (z. B. shape: (Slices, H, W) oder (Slices, H, W, 2))
    :param compared: Vergleichsbild-Volume
    :param slice_idx: Index des Slices, das verglichen werden soll
    :return: SSIM-Wert zwischen 0 (komplett verschieden) und 1 (identisch)
    """
    
    # Falls die Eingabe komplexe Daten (Real + Imag) sind, umwandeln
    if original.ndim == 4 and original.shape[-1] == 2:
        original_complex = original[slice_idx, ..., 0] + 1j * original[slice_idx, ..., 1]
        original_img = np.abs(original_complex)
    else:
        original_img = original[slice_idx]

    if compared.ndim == 4 and compared.shape[-1] == 2:
        compared_complex = compared[slice_idx, ..., 0] + 1j * compared[slice_idx, ..., 1]
        compared_img = np.abs(compared_complex)
    else:
        compared_img = compared[slice_idx]

    # Optional: Normalisierung (SSIM erwartet gleichen Wertebereich)
    original_img = original_img / np.max(original_img)
    compared_img = compared_img / np.max(compared_img)

    # SSIM-Berechnung
    score = ssim(original_img, compared_img, data_range=1.0)
    return score



def plot_grayscale_histogram(vol):
    vol_log = np.log1p(vol)  # log(1 + x), um auch 0-Werte abzubilden
    plt.hist(vol_log.ravel(), bins=256, color='gray')
    plt.title("Log-scaled histogram of the fft-amplitudes")
    plt.xlabel("log(1 + amplitude)")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.show()



def get_image_volume_fft2(data):
    """
    Construct a 3D image volume from 4D complex data using 2D FFT per slice.
    This assumes the input data is 170 sagittal slices in image domain.
    """
    data_complex = data[..., 0] + 1j * data[..., 1]  # Shape: (170, 256, 256)
    vol = np.zeros_like(data_complex, dtype=np.float32)

    for i in range(data_complex.shape[0]):  # Loop over sagittal slices
        vol[i] = np.abs(np.fft.fft2(data_complex[i]))
        # plot_grayscale_histogram(vol[i])              <- necessary for HISTOGRAMS over single slices instead of whole volume

    # print(f'Data dimensions: {vol.shape}')
    return vol  # Shape: (170, 256, 256)




image_vol = np.load(r"C:\Users\Sarah\OneDrive\Documents\8.Semester\Projekt\experiments\data\Single-channel\Train_part1\Train\e13991s3_P01536.7.npy")
image_vol_2 = np.load(r"C:\Users\Sarah\OneDrive\Documents\8.Semester\Projekt\experiments\data\Single-channel\Train_part1\Train\e14078s3_P02048.7.npy")
image_vol_ffttransformed = get_image_volume_fft2(image_vol)                                 # Bilddaten im k-space / FFT Raum
image_vol_2_ffttransformed = get_image_volume_fft2(image_vol_2)                             # Bilddaten im k-space / FFT Raum

print("PSNR:", calculate_psnr(image_vol, image_vol_2, max_pixel_value=np.max(image_vol)))
show_slices_side_by_side(image_vol_ffttransformed, image_vol_2_ffttransformed, slice_idx=80)

print("SSIM:", calculate_ssim(image_vol, image_vol_2))


def plot_every_vol_as_histogram(dateien):
    for datei in dateien:
        dateipfad = os.path.join(ordner, datei)
        data = np.load(dateipfad)  # Shape: (170, 256, 256, 2)
        print(f"Verarbeite: {datei}")
        vol = get_image_volume_fft2(data)
        # print(f"{datei}: vol min={vol.min()}, max={vol.max()}, mean={vol.mean()}, std={vol.std()}")
        # print("NaNs vorhanden?", np.isnan(vol).any())

        plot_grayscale_histogram(vol)                   # <- HISTOGRAM over entire volume instead of single slices (see code line 104)


ordner = r"C:\Users\Sarah\OneDrive\Documents\8.Semester\Projekt\experiments\data\Single-channel\Train_part1\Train"
dateien = sorted(os.listdir(ordner))
# plot_every_vol_as_histogram(dateien)




# Beispiel: Zwei Bilder als .npy laden
img1_np = np.load(r"C:\Users\Sarah\OneDrive\Documents\8.Semester\Projekt\experiments\data\Single-channel\Train_part1\Train\e13991s3_P01536.7.npy")  # dein erstes Bild
img2_np = np.load(r"C:\Users\Sarah\OneDrive\Documents\8.Semester\Projekt\experiments\data\Single-channel\Train_part1\Train\e14078s3_P02048.7.npy")       # dein zweites Bild (hier musst du dein 2. File angeben!)

# Sicherstellen, dass sie float-Werte haben
img1_np = img1_np.astype(np.float32)
img2_np = img2_np.astype(np.float32)

# Falls sie unterschiedliche Größen haben → zuschneiden oder resizen
# Hier: Zuschneiden auf die kleinste gemeinsame Größe
min_height = min(img1_np.shape[0], img2_np.shape[0])
min_width  = min(img1_np.shape[1], img2_np.shape[1])

img1_np = img1_np[:min_height, :min_width]
img2_np = img2_np[:min_height, :min_width]

# NCC Funktion
def normalized_cross_correlation(img1, img2):
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    
    numerator = np.sum((img1 - mean1) * (img2 - mean2))
    denominator = np.sqrt(np.sum((img1 - mean1)**2) * np.sum((img2 - mean2)**2))
    
    ncc_value = numerator / denominator
    return ncc_value

# Berechnen und ausgeben
ncc_result = normalized_cross_correlation(img1_np, img2_np)
print(f"NCC: {ncc_result:.4f}")
