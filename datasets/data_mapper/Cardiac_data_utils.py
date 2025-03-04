import logging
import math
import numpy as np
import torch
import copy
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, ConcatDataset
import os
from torchvision.io import read_image
import json
from albumentations import *
from albumentations.pytorch import ToTensorV2
from monai import data, transforms
import torch.nn.functional as F
from datasets.data_mapper.datasampler import LabelBatchSampler, DistributedLabelBatchSampler
from datasets.data_mapper.Cardiac_dataloader import CardiacDataset

import torch
import numpy as np
from torch.utils.data import Sampler
import math


class WeightedDistributedSampler(Sampler):
    """
    A sampler that supports both weighted sampling and distributed training.

    Args:
        dataset (Dataset): The dataset to sample from.
        weights (torch.Tensor): A tensor of weights for each sample.
        num_samples (int): The number of samples to draw.
        replacement (bool): Whether to sample with replacement.
        shuffle (bool): Whether to shuffle the dataset between epochs.
        num_replicas (int): Number of processes in distributed training.
        rank (int): Rank of the current process.
        seed (int): Random seed for shuffling.
    """

    def __init__(self, dataset, weights, num_samples=None, replacement=True,
                 shuffle=True, num_replicas=None, rank=None, seed=42):
        if not torch.distributed.is_available():
            raise RuntimeError("Requires distributed package to be available.")
        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed process group is not initialized.")

        self.dataset = dataset
        self.weights = weights
        self.replacement = replacement
        self.shuffle = shuffle
        self.seed = seed

        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank or torch.distributed.get_rank()

        # Calculate the number of samples per process
        self.num_samples = num_samples or len(dataset)
        self.num_samples_per_replica = math.ceil(self.num_samples / self.num_replicas)

        # Total number of samples for all replicas
        self.total_size = self.num_samples_per_replica * self.num_replicas

    def __iter__(self):
        # Generate the weighted sampling indices
        g = torch.Generator()
        g.manual_seed(self.seed + torch.distributed.get_rank())

        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=g
        ).tolist()

        # Shuffle if needed
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)

        # Subsample for this replica
        start = self.rank * self.num_samples_per_replica
        end = start + self.num_samples_per_replica
        indices = indices[start:end]

        return iter(indices)

    def __len__(self):
        return self.num_samples_per_replica

def ACDC_collate_fn(batch):
    # Return the batch as is, without trying to combine the dictionaries
    # This makes each element of the batch a separate dictionary
    return batch


def get_loader(cfg):
    train_transform = transforms.Compose(
        [
            transforms.AddChanneld(keys=["sem_seg"]),
            transforms.RandFlipd(keys=["image", "sem_seg"], prob=0.3, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "sem_seg"], prob=0.3, spatial_axis=1),
            transforms.Resized(keys=["image", "sem_seg"], spatial_size=(256, 256)),
            transforms.RandAdjustContrastd(keys=["image"], gamma=(0, 0.8), prob=0.2),
            transforms.ToTensord(keys=["image", "sem_seg"]),
            transforms.SqueezeDimd(keys=["sem_seg"], dim=0),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.AddChanneld(keys=["sem_seg"]),
            transforms.Resized(keys=["image", "sem_seg"], spatial_size=(256, 256)),
            transforms.ToTensord(keys=["image", "sem_seg"]),
            transforms.SqueezeDimd(keys=["sem_seg"], dim=0),
        ]
    )

    print(f'----- {cfg.DATASETS.TEST_MODEL} on combination dataset -----')

    # data_dir = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.NAME)
    data_dir = cfg.DATASETS.PATH
    data_name_list = cfg.DATASETS.NAME_LIST
    print(data_dir)
    if cfg.DATASETS.ITER_MODEL:
        if cfg.DATASETS.TEST_MODEL == "train":
            combination_train_ds = CardiacDataset(cfg=cfg, data_dir=data_dir, data_name_list=data_name_list, images_dir_name="image",
                                                  masks_dir_name="annotations", mode='train', transform=train_transform)
        else:
            combination_train_ds = CardiacDataset(cfg=cfg, data_dir=data_dir, data_name_list=data_name_list, images_dir_name="image",
                                                  masks_dir_name="annotations", mode='test', transform=val_transform)
    else:
        if cfg.DATASETS.TEST_MODEL == "train":
            combination_train_ds = CardiacDataset(cfg=cfg, data_dir=data_dir, transform=train_transform)
        else:
            combination_train_ds = CardiacDataset(cfg=cfg, data_dir=data_dir, transform=val_transform)

    if cfg.DATASETS.ITER_MODEL:
    # default to be true
        samples_weight = torch.from_numpy(np.array(combination_train_ds.sample_weight_list))
        train_sampler = WeightedDistributedSampler(combination_train_ds, samples_weight) if cfg.TRAINING.DIST else None

        train_loader = data.DataLoader(
            combination_train_ds,
            batch_size=cfg.TRAINING.BATCH_SIZE,
            shuffle=(train_sampler is None),
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=ACDC_collate_fn,
        )

    else:
        train_sampler = DistributedSampler(combination_train_ds) if cfg.TRAINING.DIST else None

        train_loader = data.DataLoader(
            combination_train_ds,
            batch_size=cfg.TRAINING.BATCH_SIZE,
            shuffle=(train_sampler is None),
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=ACDC_collate_fn,
        )
    return train_loader
