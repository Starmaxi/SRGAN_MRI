import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from PIL import Image

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

# print("PSNR:", calculate_psnr(image_vol, image_vol_2, max_pixel_value=np.max(image_vol)))
# show_slices_side_by_side(image_vol_ffttransformed, image_vol_2_ffttransformed, slice_idx=80)

# print("SSIM:", calculate_ssim(image_vol, image_vol_2))


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
img1_np = np.load(r"C:\Users\Sarah\OneDrive\Documents\8.Semester\Projekt\experiments\data\Single-channel\Train_part1\Train\e13991s3_P01536.7.npy") 
img2_np = np.load(r"C:\Users\Sarah\OneDrive\Documents\8.Semester\Projekt\experiments\data\Single-channel\Train_part1\Train\e14078s3_P02048.7.npy") 

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
#ncc_result = normalized_cross_correlation(img1_np, img2_np)
#print(f"NCC: {ncc_result:.4f}")



######### PSNR ###############################################
# Pfade zu den Verzeichnissen
original_dir = r"C:\Users\Sarah\OneDrive\Documents\Maschinelles Lernen\ML2\Aufgabenblatt 3\archiv(1).tar\orig_srgan_val.tar\val_ground_truth\sagittal"
compared_dir = r"C:\Users\Sarah\OneDrive\Documents\Maschinelles Lernen\ML2\Aufgabenblatt 3\archiv(1).tar\orig_srgan_val.tar\val_processed\sagittal"

# Anzahl der Bilder
num_images = 1292  # von 0 bis 27111 -> 27112 Bilder

# Schleife über alle Bilder
# psnr_values = []

# for i in range(num_images):
    
#     filename = f"{i}.png"
    
#     # Volle Pfade zusammensetzen
#     original_path = os.path.join(original_dir, filename)
#     compared_path = os.path.join(compared_dir, filename)
    
#     # Bilder laden
#     original = np.array(Image.open(original_path)).astype(np.float32)
#     compared = np.array(Image.open(compared_path)).astype(np.float32)
    
#     # PSNR berechnen
#     psnr = calculate_psnr(original, compared)
#     psnr_values.append(psnr)
    
#     # Optional: Fortschritt ausgeben
#     # if i % 1000 == 0:
#     #     print(f"Bild {i}/{num_images-1}: PSNR = {psnr:.2f} dB")

# # Optional: Durchschnittliches PSNR ausgeben
# average_psnr = np.mean(psnr_values)
# print(f"\nDurchschnittliches PSNR über alle axial-Bilder: {average_psnr:.2f} dB")



########### SSIM ####################################################################
# Pfade zu den Verzeichnissen
# original_dir = r"C:\Users\Sarah\OneDrive\Documents\Maschinelles Lernen\ML2\Aufgabenblatt 3\archiv(1).tar\val_data_partly_trained.tar\ground_truth\sagittal"
# compared_dir = r"C:\Users\Sarah\OneDrive\Documents\Maschinelles Lernen\ML2\Aufgabenblatt 3\archiv(1).tar\val_data_partly_trained.tar\processed\sagittal"

# Anzahl der Bilder
# num_images = 1292  # von 0 bis 27111 -> 27112 Bilder

# # # Schleife über alle Bilder
# ssim_values = []

# for i in range(num_images):
#     filename = f"{i}.png"
    
#     # Volle Pfade zusammensetzen
#     original_path = os.path.join(original_dir, filename)
#     compared_path = os.path.join(compared_dir, filename)
    
#     # Bilder laden
#     original = np.array(Image.open(original_path)).astype(np.float32)
#     compared = np.array(Image.open(compared_path)).astype(np.float32)
    
#     # Falls Bilder farbig sind, in Graustufen umwandeln
#     if original.ndim == 3:
#         original = np.mean(original, axis=2)
#     if compared.ndim == 3:
#         compared = np.mean(compared, axis=2)
    
#     # SSIM berechnen
#     ssim_score = ssim(original, compared, data_range=original.max() - original.min())
#     ssim_values.append(ssim_score)
    
#     # # Optional: Fortschritt ausgeben
#     # if i % 1000 == 0:
#     #     print(f"Bild {i}/{num_images-1}: SSIM = {ssim_score:.4f}")

# # Optional: Durchschnittliches SSIM ausgeben
# average_ssim = np.mean(ssim_values)
# print(f"\nDurchschnittliches SSIM über alle sagittal Bilder: {average_ssim:.4f}")


############ NCC ################################################
# Liste zum Speichern der NCC-Werte
# ncc_values = []

# for i in range(num_images):
#     filename = f"{i}.png"
    
#     original_path = os.path.join(original_dir, filename)
#     compared_path = os.path.join(compared_dir, filename)
    
#     # Bilder laden und in float32 umwandeln
#     original = np.array(Image.open(original_path)).astype(np.float32)
#     compared = np.array(Image.open(compared_path)).astype(np.float32)
    
#     # In Graustufen umwandeln, falls Bild RGB ist
#     if original.ndim == 3:
#         original = np.mean(original, axis=2)
#     if compared.ndim == 3:
#         compared = np.mean(compared, axis=2)

#     # NCC berechnen
#     ncc = normalized_cross_correlation(original, compared)
#     ncc_values.append(ncc)

    # Optional: Fortschritt anzeigen
    # if i % 1000 == 0:
    #     print(f"Bild {i}/{num_images-1}: NCC = {ncc:.4f}")

# Durchschnittlichen NCC ausgeben
# average_ncc = np.mean(ncc_values)
# print(f"\nDurchschnittlicher NCC über alle sagittal-Bilder: {average_ncc:.4f}")



############# Histograms ###################################################
def plot_image_histograms(image_path1, image_path2, title1="Image 1", title2="Image 2"):
    """
    Loads two images and plots their grayscale histograms side by side.

    Parameters:
    - image_path1: str, path to the first image
    - image_path2: str, path to the second image
    - title1: str, optional, title for the first image histogram
    - title2: str, optional, title for the second image histogram
    """

    # Load and convert images to grayscale
    img1 = Image.open(image_path1).convert("L")
    img2 = Image.open(image_path2).convert("L")

    # Convert to numpy arrays
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(img1_array.flatten(), bins=256, range=(0, 255), color='gray')
    axes[0].set_title(f"{title1} Histogram")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(img2_array.flatten(), bins=256, range=(0, 255), color='gray')
    axes[1].set_title(f"{title2} Histogram")
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# plot_image_histograms(r"C:\Users\Sarah\OneDrive\Documents\Maschinelles Lernen\ML2\Aufgabenblatt 3\archiv(1).tar\val_data_partly_trained.tar\ground_truth\axial\0.png", r"C:\Users\Sarah\OneDrive\Documents\Maschinelles Lernen\ML2\Aufgabenblatt 3\archiv(1).tar\val_data_partly_trained.tar\processed\axial\0.png", "Ground Truth", "Upscaled")


# lr_img = Image.open(r"C:\Users\Sarah\OneDrive\Documents\Maschinelles Lernen\ML2\Aufgabenblatt 3\archiv(1).tar\val_data_partly_trained.tar\processed\axial\0.png").convert("L")
# hr_img = Image.open(r"C:\Users\Sarah\OneDrive\Documents\Maschinelles Lernen\ML2\Aufgabenblatt 3\archiv(1).tar\val_data_partly_trained.tar\ground_truth\axial\0.png").convert("L")

# target_size = (256,256)
# lr_img_resized = lr_img.resize(target_size, Image.Resampling.LANCZOS)  # LANCZOS für bessere Qualität
# hr_img_resized = hr_img.resize(target_size, Image.Resampling.LANCZOS)

# # Bilder als NumPy-Arrays konvertieren
# lr_np = np.array(lr_img_resized)  # Array-Form: (H, W)
# hr_np = np.array(hr_img_resized)  # Array-Form: (H, W)

# error_map = np.abs(hr_np - lr_np)

# def plot_histogram(image, title):
#     # Bild in Grauwert umwandeln
#     grayscale_img = image.convert("L")
#     img_array = np.array(grayscale_img).flatten()  # Bild in 1D Array umwandeln
#     plt.hist(img_array, bins=256, color='gray', alpha=0.7)
#     plt.title(title)
#     plt.xlabel('Pixel Wert')
#     plt.ylabel('Häufigkeit')

# plt.figure(figsize=(18, 12))

# plt.subplot(2, 3, 1)
# plt.imshow(lr_img, cmap='gray')
# plt.title("Low Resolution")

# plt.subplot(2, 3, 2)
# plt.imshow(hr_img, cmap='gray')
# plt.title("High Resolution")

# plt.subplot(2, 3, 3)
# plt.imshow(error_map, cmap='hot')
# plt.title("Error Map (Abs. Fehler)")

# # plt.subplot(2, 3, 4)
# # plot_histogram(lr_img, "Low Resolution Histogram")

# # plt.subplot(2, 3, 5)
# # plot_histogram(hr_img, "High Resolution Histogram")

# # plt.subplot(2, 3, 6)
# # plt.hist(error_map.flatten(), bins=50, color='red', alpha=0.7)
# # plt.title("Error Map Histogram")
# # plt.xlabel('Fehler Wert')
# # plt.ylabel('Häufigkeit')

# plt.tight_layout()
# plt.show()
