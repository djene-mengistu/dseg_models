import sys
sys.path.append('./')
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from segmentation_models_pytorch import Unet #Install the modules of segmentation model

model = Unet("efficientnet-b4", encoder_weights="imagenet", classes=3, activation=None) #Change accordingly