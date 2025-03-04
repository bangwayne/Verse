import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import math
import random
import pdb
import logging
import copy
from pathlib import Path

class CardiacDataset(Dataset):
    def __init__(self, cfg, data_dir, data_name_list, images_dir_name="image", masks_dir_name="annotations", mode='train',
                 transform=None, three_chanel=True):

        self.mode = mode
        self.data_name_list = ["ACDC", "MM", "MnM2", "MYO_C0", "MYO_T2", "MYO_LGE", "LASCARQS"]
        self.dataset_weight = {
            "ACDC": 0.2,
            "MM": 0.2,
            "MnM2": 1.0,
            "MYO_C0": 0.2,
            "MYO_T2": 1.2,
            "MYO_LGE": 1.0,
            "LASCARQS": 0.2,
        }
        self.data_config = {
            "ACDC": {"unique_labels": [1, 2, 3]},
            "MM": {"unique_labels": [1, 2, 3]},
            "MnM2": {"unique_labels": [1, 2, 3]},
            "MYO_C0": {"unique_labels": [1, 2, 3]},
            "MYO_T2": {"unique_labels": [4]},
            "MYO_LGE": {"unique_labels": [5]},
            "LASCARQS": {"unique_labels": [6]}
        }
        self.dataset_path_list = [Path(data_dir + "/" + data_name) for data_name in self.data_name_list]
        self.images_path_list = [dataset_path / mode / images_dir_name for dataset_path in self.dataset_path_list]
        self.masks_path_list = [dataset_path / mode / masks_dir_name for dataset_path in self.dataset_path_list]
        self.three_channel = three_chanel
        self.transform = transform
        self.dataset_samples = []
        self.metadata, self.sample_weight_list = self.load_metadata()
        for i in range(len(self.data_name_list)):
            images_path = self.images_path_list[i]
            for x in sorted(images_path.glob('*.nii.gz')):
                self.dataset_samples.append(x.name)
        print(f'len(self.dataset_samples) = {len(self.dataset_samples)}')

    def load_metadata(self):
        metadata = []
        weight_list = []
        dataset_slice_counts = {name: 0 for name in self.data_config}

        for idx, dataname in enumerate(self.data_config):
            images_path = self.images_path_list[idx]
            masks_path = self.masks_path_list[idx]
            img_name_list = [x.name for x in sorted(images_path.glob('*.nii.gz'))]

            print(f"Start loading {dataname} {self.mode} metadata")
            for filename in img_name_list:
                img_path = os.path.join(images_path, filename)
                mask_name = filename.split('.')[0] + "_gt.nii.gz"
                mask_path = os.path.join(masks_path, mask_name)

                itk_mask = sitk.ReadImage(mask_path)
                array_mask = sitk.GetArrayFromImage(itk_mask)

                unique_labels = self.data_config[dataname]["unique_labels"]
                slices_num = array_mask.shape[0]

                for slice_index in range(slices_num):
                    slice_mask = array_mask[slice_index]
                    real_unique_labels = np.unique(slice_mask[slice_mask != 0])
                    if len(real_unique_labels) > 0:
                        weight_list.append(self.dataset_weight[dataname])
                        dataset_slice_counts[dataname] += 1
                        metadata.append((img_path, mask_path, slice_index, unique_labels, real_unique_labels, filename))

                del itk_mask, array_mask

        print(f"Load done, length of dataset: {len(metadata)}")
        print("Slice counts per dataset:")
        for dataset, count in dataset_slice_counts.items():
            print(f"  {dataset}: {count} slices")

        assert len(metadata) == len(weight_list), (
            f"Length mismatch: metadata ({len(metadata)}) and weight_list ({len(weight_list)})"
        )
        return metadata, weight_list

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path, mask_path, slice_index, labels, real_unique_labels, filename = self.metadata[idx]
        if self.mode == "train":
            unique_label = np.random.choice(real_unique_labels)
        itk_img = sitk.ReadImage(img_path)
        itk_mask = sitk.ReadImage(mask_path)

        tensor_img, tensor_mask = self.preprocess(itk_img, itk_mask)

        height, width = tensor_img.shape[1], tensor_img.shape[2]

        gray_image_tensor = tensor_img[slice_index]

        if self.three_channel:
            gray_image_tensor = gray_image_tensor.unsqueeze(0)
            three_channel_image = gray_image_tensor.repeat(3, 1, 1)
            image = three_channel_image
        else:
            image = gray_image_tensor

        sample = {'image': image,
                  'sem_seg': tensor_mask[slice_index]}
        if self.transform:
            sample = self.transform(sample)

        sample['file_name'] = img_path
        sample['slice_index'] = slice_index
        sample['width'] = width
        sample['height'] = height
        sample['q_index'] = int(unique_label)


        transformed_mask = sample['sem_seg']

        single_mask = torch.stack([(transformed_mask == unique_label).bool()])

        target = {
            "labels": torch.tensor(labels).long(),
            'unique_labels': self.get_label([unique_label]).long(),
            # this place is very important to keep the label dtype long, else it will report some error
            "masks": single_mask
        }
        sample['target'] = target

        return sample

    @staticmethod
    def preprocess(itk_img, itk_mask):
        img = sitk.GetArrayFromImage(itk_img)
        mask = sitk.GetArrayFromImage(itk_mask)
        tensor_img = torch.from_numpy(img).float()
        tensor_mask = torch.from_numpy(mask).long()
        return tensor_img, tensor_mask

    @staticmethod
    def get_label(unique_labels, label_set=[1, 2, 3, 4, 5, 6]):
        label_mask_list = []

        for label in unique_labels:
            label_mask = [0] * (len(label_set) + 1)
            if label in label_set:
                index = label_set.index(label)
                label_mask[index] = 1
            else:
                label_mask[-1] = 1
            label_mask_list.append(label_mask)

        return torch.tensor(label_mask_list)

    




