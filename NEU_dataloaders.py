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
from NEU_utilities import*
warnings.filterwarnings("ignore")


#Data pipeline:

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
        image_path = os.path.join(self.root, "train_images",  image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] # 1x200x200x3
        mask = mask.permute(2,0,1) # 3x200x200
        return img, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                Blur(p = 0.3),
                GaussNoise(p=0.5),
                # GaussNoise(p=0.5),
                # ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                # Transpose(p=0.5),
                # GridDistortion(p=1),
                HorizontalFlip(p=0.5), # only horizontal flip as of now
                # RandomSizedCrop(min_max_height=(50, 100), height=224, width=224, p=0.4),
                RandomBrightnessContrast(p=0.4),
                # GridDistortion(p=0.5),
                RandomRotate90(p=0.5),
                OneOf([
                       ElasticTransform(p=0.4, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                       GridDistortion(p=0.3),
                       OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),], p=0.4)
                
            ]

        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            Resize(height=256, width=256, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1),
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
    batch_size=16,
    num_workers=4,
):
    '''Returns dataloader for the model training'''
    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69) #proportion of labeled training images
    # train_df, val_df = train_test_split(ntrain_df, test_size=0.2, random_state=45)
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, mean, std, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )

    return dataloader