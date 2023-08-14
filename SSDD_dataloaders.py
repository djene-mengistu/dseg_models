#Importing the required libraries

import os
import cv2
import warnings
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, 
                            Resize, Compose, GaussNoise, Blur, RandomSizedCrop, 
                            ElasticTransform, GridDistortion, OpticalDistortion,
                            RandomRotate90, RandomBrightnessContrast, OneOf)
from albumentations.pytorch import ToTensorV2
from SSDD_utilities import*
warnings.filterwarnings("ignore")


#Data pipeline for semisupervised training:Duo-SegNet

class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "trainval_images",  image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] # 1x256x1600x4
        mask = mask.permute(2, 0, 1) # 4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5), # only horizontal flip as of now
                Blur(p=0.3),
                GaussNoise(p=0.5),
                # CLAHE(clip_limit = 4.0, title_grid_size = (8,8), always_apply=False, p=0.5),
                
                # RandomCrop(256, 400, p=1),
                # RandomResizedCrop(height=256, width=256, scale=(0.08,1.0), ratio=(0.75,1.333),interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0 ),
                # RandomSizedCrop(min_max_height = 256, height=256, width=256, w2h_ratio = 1.0, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0 ),
                # CropNonEmptyMaskIfExists(height=256, width=256, ignore_values = None, ignore_channels = None, always_apply=True, p=1.0 )
            ]
        )
    list_transforms.extend(
        [
            # RandomCrop(256, 400, p=1),
            # Resize(256, 256, p=1),
            Normalize(mean=mean, std=std, p=1),
            # RandomCrop(256, 400, p=1),
            # Resize(256, 256, p=1),
            ToTensorV2(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms

def dataloaders(
    data_folder,
    df_path,
    phase,
    mean=None,
    std=None,
    batch_size=4,
    num_workers=4,
):
    '''Returns dataloader for the model training'''
    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    
    train_df, val_df = train_test_split(df, test_size=0.20, stratify=df["defects"], random_state=69)
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, mean, std, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last = True,
        pin_memory=True,
        shuffle=True,   
    )

    return dataloader
