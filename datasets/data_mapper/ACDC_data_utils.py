from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, ConcatDataset
import os
from torchvision.io import read_image
import json
from albumentations import *
from albumentations.pytorch import ToTensorV2
from monai import data, transforms
from datasets.data_mapper.ACDC_dataloader import ACDCDataset


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

    # train_transform = None
    # val_transform = None
    print(f'----- {cfg.DATASETS.TEST_MODEL} on combination dataset -----')

    data_dir = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.NAME)
    # print(data_dir)

    #data_dir = cfg.DATASETS.PATH
    print(data_dir)
    if cfg.DATASETS.ITER_MODEL:
        if cfg.DATASETS.TEST_MODEL == "train":
            combination_train_ds = ACDCDataset(cfg=cfg, data_dir=data_dir, images_dir_name="image",
                                               masks_dir_name="annotations", mode='train', transform=train_transform)
        else:
            combination_train_ds = ACDCDataset(cfg=cfg, data_dir=data_dir, images_dir_name="image",
                                               masks_dir_name="annotations", mode='test', transform=val_transform)
    else:
        if cfg.DATASETS.TEST_MODEL == "train":
            combination_train_ds = ACDCDataset(cfg=cfg, data_dir=data_dir, transform=train_transform)
        else:
            combination_train_ds = ACDCDataset(cfg=cfg, data_dir=data_dir, transform=val_transform)

    if cfg.DATASETS.ITER_MODEL:
        train_sampler = DistributedSampler(combination_train_ds) if cfg.TRAINING.DIST else None

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


