import SimpleITK as sitk
import numpy as np
import os
# This file is used to norm the nii.gz file
def normalize_data_2d(img_np):
    # preprocessing
    t,y,x = img_np.shape
    for i in range(t):
        img = img_np[i,::]
        cmin = np.min(img)
        cmax = np.percentile(img,98)
        img[img > cmax] = cmax #norm, to avoid the part too light
        img_array = (img - cmin) / (cmax- cmin + 0.0001)
        img_np[i, ::] = img_array

    return img_np

def normalize_data_3d(img_np):
    # preprocessing
    cmin = np.min(img_np)
    cmax = np.max(img_np)
    img_np = (img_np - cmin) / (cmax- cmin + 0.0001)
    return img_np

def normalize_data(img_np):
    # preprocessing
    cm = np.median(img_np)
    img_np = img_np / (2*cm + 0.0001)
    img_np[img_np < 0] = 0.0
    img_np[img_np >1.0] = 1.0
    return img_np

current_script_dir = os.path.dirname(__file__)

# Go up one level to the 'xx' directory
parent_dir = os.path.dirname(current_script_dir)

#image_path = parent_dir + "/weights_and_datasets/ACDC/normed/testing/image"

training_origin_path = "/research/cbim/medical/bg654/MYOcardial Segmentation with Automated Infarct Quantification/database/training/D8_postMI"
save_training_image_path = "/research/cbim/medical/bg654/MYOcardial Segmentation with Automated Infarct Quantification/database/norm_training/D8_postMI/images"
save_training_gt_path = "/research/cbim/medical/bg654/MYOcardial Segmentation with Automated Infarct Quantification/database/norm_training/D8_postMI/labels"
os.makedirs(save_training_image_path, exist_ok=True)
os.makedirs(save_training_gt_path, exist_ok=True)
print(len(os.listdir(training_origin_path)))

for subroot, dirs, files in os.walk(training_origin_path):
    for i in dirs:
        training_patient_path = os.path.join(subroot,i)
        file_list = os.listdir(training_patient_path)
        if i == "images":
            for file in file_list:
                print(file)
                # if file.endswith('nii.gz') and 'frame' in file and 'gt' not in file:
                if file.endswith('nii.gz') and 'gt' not in file:
                    cine_file = os.path.join(training_patient_path, file)
                    cine_image = sitk.ReadImage(cine_file)
                    cine_image = sitk.Cast(cine_image, sitk.sitkFloat64)
                    spacing2 = cine_image.GetSpacing()
                    origin2 = cine_image.GetOrigin()
                    direction2 = cine_image.GetDirection()
                    np_img = sitk.GetArrayFromImage(cine_image)
                    np_img_normed = normalize_data_2d(np_img)
                    mask2 = sitk.GetImageFromArray(np_img_normed)
                    mask2.SetSpacing(spacing2)
                    mask2.SetOrigin(origin2)
                    mask2.SetDirection(direction2)
                    sitk.WriteImage(mask2, os.path.join(save_training_image_path,file))
                # if file.endswith('nii.gz') and 'frame' in file and 'gt' in file:
                if file.endswith('nii.gz') and 'gt' in file:
                    cine_file = os.path.join(training_patient_path, file)
                    cine_image = sitk.ReadImage(cine_file)
                    cine_image = sitk.Cast(cine_image, sitk.sitkFloat64)
                    spacing2 = cine_image.GetSpacing()
                    origin2 = cine_image.GetOrigin()
                    direction2 = cine_image.GetDirection()
                    np_img = sitk.GetArrayFromImage(cine_image)
    #                 np_img_normed = normalize_data_2d(np_img)
                    mask2 = sitk.GetImageFromArray(np_img)
                    mask2.SetSpacing(spacing2)
                    mask2.SetOrigin(origin2)
                    mask2.SetDirection(direction2)
                    sitk.WriteImage(mask2, os.path.join(save_training_gt_path,file))