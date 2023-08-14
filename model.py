#The model for dual-segmetnation netwrok
#Import the required libraries
import torch
import torch.nn as nn
import functools
import numpy as np
import sys
sys.path.append('./')

from models.FPN import model as m #import all the required models accordingly ({ENEt, DANet, ENCNet, and etc})
from models.proposed_models.segformer import*
from models.proposed_models.Transformer_based import*
from models.proposed_models.CNN_based import*

# net = m # Use this for CNN-based models
#model = net
net = Transformer_based('ResT-S') #Proposed hybrid netwrok of trasnformer-encoder (ResT) and CNN-decoder (UperNet)
# net = CNN_based('ConvNeXt-S') #CNN-based hybrid Networks
net.init_pretrained('.../pretrained_weights/cpt/rest_small.pth') #Dowload the required pretrained weights for the backbone netwrok from official implementaiton cite
model = net
