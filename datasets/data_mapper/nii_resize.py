import os

import SimpleITK as sitk
import numpy as np

def resample_sitkImage_by_size(sitkImage, newSize, vol_default_value='min'):
    """
    :param sitkImage:
    :param newSize:
    :return:
    """
    if sitkImage == None: return None
    if newSize is None: return None
    dim = sitkImage.GetDimension()
    print(dim)
    if len(newSize) != dim: return None

    # determine the default value
    vol_value = 0.0
    if vol_default_value == 'min':
        vol_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitkImage)))
    elif vol_default_value == 'zero':
        vol_value = 0.0
    elif str(vol_default_value).isnumeric():
        vol_value = float(vol_default_value)

    # calculate new size
    np_oldSize = np.array(sitkImage.GetSize())
    # print(np_oldSize )
    np_oldSpacing = np.array(sitkImage.GetSpacing())
    # print(np_oldSpacing)
    np_newSize = np.array(newSize)
    # print(np_newSize)
    np_newSpacing = np.divide(np.multiply(np_oldSize, np_oldSpacing), np_newSize)
    # print(np_newSpacing)
    newSpacing = tuple(np_newSpacing.astype(float).tolist())

    # resample sitkImage into new specs
    transform = sitk.Transform()
    return sitk.Resample(sitkImage, newSize, transform, sitk.sitkLinear, sitkImage.GetOrigin(),
                         newSpacing, sitkImage.GetDirection(), vol_value, sitkImage.GetPixelID())



def resample_sitkMask_by_size(sitkImage, newSize, vol_default_value='min'):
    """
    :param sitkImage:
    :param newSize:
    :return:
    """
    if sitkImage == None: return None
    if newSize is None: return None
    dim = sitkImage.GetDimension()
    if len(newSize) != dim: return None

    # determine the default value
    vol_value = 0.0
    if vol_default_value == 'min':
        vol_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitkImage)))
    elif vol_default_value == 'zero':
        vol_value = 0.0
    elif str(vol_default_value).isnumeric():
        vol_value = float(vol_default_value)

    # calculate new size
    np_oldSize = np.array(sitkImage.GetSize())
    np_oldSpacing = np.array(sitkImage.GetSpacing())
    np_newSize = np.array(newSize)
    np_newSpacing = np.divide(np.multiply(np_oldSize, np_oldSpacing), np_newSize)
    newSpacing = tuple(np_newSpacing.astype(float).tolist())

    # resample sitkImage into new specs
    transform = sitk.Transform()
    return sitk.Resample(sitkImage, newSize, transform, sitk.sitkNearestNeighbor, sitkImage.GetOrigin(),
                         newSpacing, sitkImage.GetDirection(), vol_value, sitkImage.GetPixelID())


def resample_smooth_sitkMask_by_size(sitkImage, newSize, vol_default_value='min', interpolation_method=sitk.sitkLinear):
    """
    Resample a SimpleITK image to a new size with optional smoothing.

    :param sitkImage: Input SimpleITK image.
    :param newSize: New size for the resampled image (tuple).
    :param vol_default_value: Default voxel value for regions outside the original image.
    :param interpolation_method: Interpolation method to use (default is sitk.sitkLinear for smoother borders).
    :return: Resampled SimpleITK image.
    """
    if sitkImage is None or newSize is None:
        return None
    dim = sitkImage.GetDimension()
    if len(newSize) != dim:
        return None

    # Determine the default value for outside pixels
    vol_value = 0.0
    if vol_default_value == 'min':
        vol_value = float(np.min(sitk.GetArrayFromImage(sitkImage)))
    elif vol_default_value == 'zero':
        vol_value = 0.0
    elif str(vol_default_value).isnumeric():
        vol_value = float(vol_default_value)

    # Calculate new spacing
    np_oldSize = np.array(sitkImage.GetSize())
    np_oldSpacing = np.array(sitkImage.GetSpacing())
    np_newSize = np.array(newSize)
    np_newSpacing = np.divide(np.multiply(np_oldSize, np_oldSpacing), np_newSize)
    newSpacing = tuple(np_newSpacing.astype(float).tolist())

    # Resample the image with the specified interpolation method
    transform = sitk.Transform()
    resampled_image = sitk.Resample(
        sitkImage,
        newSize,
        transform,
        interpolation_method,  # Use smoother interpolation method
        sitkImage.GetOrigin(),
        newSpacing,
        sitkImage.GetDirection(),
        vol_value,
        sitkImage.GetPixelID()
    )

    return resampled_image


# current_script_dir = os.path.dirname(__file__)
current_script_dir = "/research/cbim/medical/bg654/MM"
# Go up one level to the 'xx' directory
# parent_dir = os.path.dirname(current_script_dir)
parent_dir = "/research/cbim/medical/bg654/MM"
print(parent_dir)

image_path = parent_dir + "/norm_testing/image"
mask_path = parent_dir + "/norm_testing/gt"
save_image_path = parent_dir + "/testing/image"
save_mask_path = parent_dir + "/testing/gt"
os.makedirs(save_image_path, exist_ok=True)
os.makedirs(save_mask_path, exist_ok=True)

for image in os.listdir(image_path):
    single_image_path = os.path.join(image_path,image)
    nii_image = sitk.ReadImage(single_image_path)
    slice_num = np.array(nii_image.GetSize())[2]
    resized_image = resample_sitkImage_by_size(nii_image, [256, 256, int(slice_num)])
    sitk.WriteImage(resized_image, os.path.join(save_image_path, image))


for mask in os.listdir(mask_path):
    single_mask_path = os.path.join(mask_path, mask)
    nii_mask = sitk.ReadImage(single_mask_path)
    slice_num = np.array(nii_mask.GetSize())[2]
    resized_mask = resample_sitkMask_by_size(nii_mask, [256, 256, int(slice_num)])
    sitk.WriteImage(resized_mask, os.path.join(save_mask_path, mask))



for mask in os.listdir(mask_path):
    single_mask_path = os.path.join(mask_path, mask)
    nii_mask = sitk.ReadImage(single_mask_path)
    slice_num = np.array(nii_mask.GetSize())[2]

    # Load the corresponding image
    single_image_path = os.path.join(image_path, mask)  # Assuming the image has the same name as the mask
    nii_image = sitk.ReadImage(single_image_path)

    # Initialize lists to store valid slices for the mask and image
    filtered_mask_slices = []
    filtered_image_slices = []

    # Iterate through each slice
    for slice_idx in range(slice_num):
        mask_array = sitk.GetArrayFromImage(nii_mask)
        image_array = sitk.GetArrayFromImage(nii_image)
        current_mask_slice = mask_array[slice_idx, :, :]
        current_image_slice = image_array[slice_idx, :, :]

        # Check if the mask slice contains any non-zero label
        if np.any(current_mask_slice):
            filtered_mask_slices.append(current_mask_slice)
            filtered_image_slices.append(current_image_slice)

    # Only save the mask and image if there are valid slices
    if filtered_mask_slices:
        # Rebuild new 3D volumes for the mask and image
        new_mask_array = np.array(filtered_mask_slices)
        new_image_array = np.array(filtered_image_slices)

        new_nii_mask = sitk.GetImageFromArray(new_mask_array)
        new_nii_image = sitk.GetImageFromArray(new_image_array)

        # Keep the same metadata for both
        new_nii_mask.SetSpacing(nii_mask.GetSpacing())
        new_nii_mask.SetOrigin(nii_mask.GetOrigin())
        new_nii_mask.SetDirection(nii_mask.GetDirection())

        new_nii_image.SetSpacing(nii_image.GetSpacing())
        new_nii_image.SetOrigin(nii_image.GetOrigin())
        new_nii_image.SetDirection(nii_image.GetDirection())

        # Resample and save the filtered mask and image
        resized_mask = resample_sitkMask_by_size(new_nii_mask, [256, 256, len(filtered_mask_slices)])
        resized_image = resample_sitkImage_by_size(new_nii_image, [256, 256, len(filtered_image_slices)])

        sitk.WriteImage(resized_mask, os.path.join(save_mask_path, mask))
        sitk.WriteImage(resized_image, os.path.join(save_image_path, mask))  # Save image with the same name as the mask