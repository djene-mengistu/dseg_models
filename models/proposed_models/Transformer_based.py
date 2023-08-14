import torch
from torch import Tensor
from torch.nn import functional as F
import sys
sys.path.append('./')
from proposed_models.base import BaseModel
from proposed_models.fpn_head import FPNHead
from proposed_models.upernet_head import UPerHead 

'''number of classes is 3 for NEU and 4 for SSDD'''

class Transformer_based(BaseModel):
    def __init__(self, backbone: str = 'ResT-S', num_classes: int = 3) -> None: #Change the decoder type accordingly {PVT, MIT, ResT, and others}
        super().__init__(backbone, num_classes)
        self.decode_head = UPerHead(self.backbone.channels, 128, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


# if __name__ == '__main__':
# net = Transformer_based('ResT-S', 4)
# net.init_pretrained('.../pretrained_weights/cpt/rest_small.pth')
# model = net
# x = torch.zeros(2, 3, 256, 256)
# y = model(x)
# print(y.shape)
# print(model)
        

