import torch
from torch import Tensor
import sys
sys.path.append('./')
from torch.nn import functional as F
from proposed_models.base import BaseModel
from proposed_models.lawin_head import LawinHead


class Lawin(BaseModel):
    """
    Notes:::::.
    """
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 4) -> None:
        super().__init__(backbone, num_classes)
        # self.decode_head = LawinHead(self.backbone.channels, 256 if 'B0' in backbone else 512, num_classes)
        self.decode_head = LawinHead(self.backbone.channels, 256, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


# if __name__ == '__main__':
#     model = Lawin('MiT-B0')
#     model.eval()
#     x = torch.zeros(1, 3, 256, 256)
#     y = model(x)
#     print(y.shape)
#     # from fvcore.nn import flop_count_table, FlopCountAnalysis
#     # print(flop_count_table(FlopCountAnalysis(model, x)))
# net = Lawin('ResNet-50')
# net.init_pretrained('.../pretrained_weights/cpt/resnet_50.pth')
# # model.load_state_dict(torch.load('.../pretrained_weights/cpt/segformer.b4.512x512.ade.160k.pth', map_location='cpu'), strict = False)
# model = net
# x = torch.zeros(2, 3, 256, 256)
# y = model(x)
# print(y.shape)
# print(model)