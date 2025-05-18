import cv2
import numpy as np
import os
from tqdm import tqdm


def create_brain_mask(image):
    """
    Create a mask to isolate the brain region in the MRI image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create empty mask
    mask = np.zeros_like(gray)

    if contours:
        # Find the largest contour (assumed to be the brain)
        largest_contour = max(contours, key=cv2.contourArea)

        # Fill the largest contour
        cv2.drawContours(mask, [largest_contour], -1, (255), cv2.FILLED)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def process_image(input_path, output_path):
    """
    Process a single image by applying the brain mask.
    """
    # Read image
    image = cv2.imread(input_path)

    if image is None:
        print(f"Could not read image: {input_path}")
        return

    # Create brain mask
    mask = create_brain_mask(image)

    # Apply mask to original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Save the masked image
    cv2.imwrite(output_path, masked_image)


def main():
    # Input directory containing the original images
    input_dir = "out"
    # Output directory for masked images
    output_dir = "masked_out"

    # Create output directory structure
    subdirs = ["sagittal", "axial", "coronal", "joint_data"]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # Process all images in each subdirectory
    for subdir in subdirs:
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_dir, subdir)

        if not os.path.exists(input_subdir):
            print(f"Directory not found: {input_subdir}")
            continue

        # Get all JPEG files
        jpeg_files = [f for f in os.listdir(input_subdir) if f.lower().endswith('.jpg')]

        print(f"Processing images in {subdir}...")
        for jpeg_file in tqdm(jpeg_files):
            input_path = os.path.join(input_subdir, jpeg_file)
            output_path = os.path.join(output_subdir, jpeg_file)
            process_image(input_path, output_path)


if __name__ == "__main__":
    main()