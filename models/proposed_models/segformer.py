from email.policy import strict
import torch
import os
import sys
sys.path.append('./')
from torch import Tensor
from torch.nn import functional as F
from proposed_models.base import BaseModel
from proposed_models.segformer_head import SegFormerHead

#The number of classes is NEU = 3 and SSDD = 4
class SegFormer(BaseModel):
    def __init__(self, backbone: str = 'MiT-B4', num_classes: int = 3) -> None:
        super().__init__(backbone, num_classes)
        # self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels, 256, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


# if __name__ == '__main__':
net = SegFormer('MiT-B4')
net.init_pretrained('.../pretrained_weights/cpt/mit_b4.pth') 
model = net
# x = torch.zeros(10, 3, 256, 256)
# y = model(x)
# print(y.shape)
print(model)