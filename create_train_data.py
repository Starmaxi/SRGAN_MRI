import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


# path to folder with subfolders "Train_part1", "Train_part2" etc.
path = "Single-channel/Single-channel/"

# path to output folder
out_path="out"

if not os.path.exists(out_path):
    os.makedirs(out_path)
    os.makedirs(out_path + "/sagittal")
    os.makedirs(out_path + "/axial")
    os.makedirs(out_path + "/coronal")
    os.makedirs(out_path + "/joint_data")

def get_image_volume_fft2(data):
    """
    Construct a 3D image volume from 4D complex data using 2D FFT per slice.
    This assumes the input data is 170 sagittal slices in image domain.
    """
    data_complex = data[..., 0] + 1j * data[..., 1]  # Shape: (170, 256, 256)
    vol = np.zeros_like(data_complex, dtype=np.float32)

    for i in range(data_complex.shape[0]):  # Loop over sagittal slices
        vol[i] = np.abs(np.fft.fft2(data_complex[i]))

    print(f'Data dimensions: {vol.shape}')
    return vol  # Shape: (170, 256, 256)
    
def rotate_and_save(img, out_path):

    # save image
    plt.imsave(out_path + ".jpg", img, cmap='gray')
    
    # save image rotated 90°
    img = np.rot90(img, k=-1)
    plt.imsave(out_path + "_90.jpg", img, cmap='gray')
    
    # save image rotated 180°
    img = np.rot90(img, k=-1)
    plt.imsave(out_path + "_180.jpg", img, cmap='gray')
    
    # save image rotated 270°
    img = np.rot90(img, k=-1)
    plt.imsave(out_path + "_270.jpg", img, cmap='gray')
    
    # mirror image
    img = np.rot90(img, k=-1)
    img = np.fliplr(img)
    plt.imsave(out_path + "_m.jpg", img, cmap='gray')
    
    # save image rotated 90°
    img = np.rot90(img, k=-1)
    plt.imsave(out_path + "_m90.jpg", img, cmap='gray')
    
    # save image rotated 180°
    img = np.rot90(img, k=-1)
    plt.imsave(out_path + "_m180.jpg", img, cmap='gray')

    # save image rotated 270°
    img = np.rot90(img, k=-1)
    plt.imsave(out_path + "_m270.jpg", img, cmap='gray')
    

for sub_path in ["Train_part1","Train_part2","Train_part3"]:
    for npy_file in [f for f in os.listdir(path + sub_path + "/Train/") if f.endswith(".npy")]:
        print("processing %s file" % npy_file)
        data = np.load(path + sub_path + "/Train/" + npy_file)
        img_vol = get_image_volume_fft2(data)
        
        vol_name = npy_file[:-4]
        
        for img in range(img_vol.shape[0]):
            slice_img = np.rot90(img_vol[img, :, :], k=2)
            rotate_and_save(slice_img, out_path + "/sagittal/" + vol_name + "_sagittal_" + str(img))
            rotate_and_save(slice_img, out_path + "/joint_data/" + vol_name + "_sagittal_" + str(img))

        for img in range(img_vol.shape[1]):
            slice_img = np.rot90(img_vol[:, img, :], k=1)
            rotate_and_save(slice_img, out_path + "/axial/" + vol_name + "_axial_" + str(img))
            rotate_and_save(slice_img, out_path + "/joint_data/" + vol_name + "_axial_" + str(img))
            slice_img = np.rot90(img_vol[:, :, img], k=1)
            rotate_and_save(slice_img, out_path + "/coronal/" + vol_name + "_coronal_" + str(img))
            rotate_and_save(slice_img, out_path + "/joint_data/" + vol_name + "_coronal_" + str(img))

