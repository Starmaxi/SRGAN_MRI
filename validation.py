import numpy as np
import matplotlib.pyplot as plt


def calculate_psnr(original: np.ndarray, compared: np.ndarray, max_pixel_value: float = 255.0) -> float:
    """
    Berechnet den PSNR (Peak Signal-to-Noise Ratio) zwischen zwei NumPy-Arrays.

    :param original: Das Originalbild/-array.
    :param compared: Das zu vergleichende Bild/Array (z. B. rekonstruiert oder komprimiert).
    :param max_pixel_value: Maximaler möglicher Pixelwert (z. B. 255 für 8-bit Bilder).
    :return: PSNR-Wert in dB.
    """
    mse = np.mean((original - compared) ** 2)
    if mse == 0:
        return float('inf')  # Kein Fehler = unendlich guter PSNR
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr


def plot_grayscale_histogram(image: np.ndarray):
    """
    Erstellt ein Histogramm für ein Graustufenbild.

    :param image: 2D-NumPy-Array mit Grauwerten (z. B. dtype=np.uint8)
    """
    plt.figure(figsize=(8, 5))
    plt.hist(image.ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
    plt.title("Histogramm (Graustufen)")
    plt.xlabel("Pixelwert")
    plt.ylabel("Normierte Häufigkeit")
    plt.grid(True)
    plt.show()


image = np.load(r"data\Single-channel\Train_part1\Train\e13991s3_P01536.7.npy")
plot_grayscale_histogram(image)
