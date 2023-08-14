import torch
from torch import Tensor
from torch.nn import functional as F
import sys
sys.path.append('./')
from proposed_models.base import BaseModel
from proposed_models.fpn_head import FPNHead
from proposed_models.upernet_head import UPerHead
from proposed_models.segformer_head import SegFormerHead
from proposed_models.lawin_head import LawinHead


class CNN_based(BaseModel):
    def __init__(self, backbone: str = 'ConvNeXt-S', num_classes: int = 3): #Change number of classes and backbone netwrok accordingly
        super().__init__(backbone, num_classes)
        self.decode_head = UPerHead(self.backbone.channels, 256, num_classes)
        # self.decode_head = SegFormerHead(self.backbone.channels, 256, num_classes)
        # self.decode_head = LawinHead(self.backbone.channels, 256, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y

    
# if __name__ == '__main__':
# model = CNN_based('ConvNeXt-S', 3)
# model.init_pretrained('/media/disk2t_/Dejene/DD/SegFormer_New/cpt/convnext_small.pth')
# x = torch.randn(2, 3, 224, 224)
# y = model(x)
# print(model)
# print(y.shape)