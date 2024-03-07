import torch
import torch.nn as nn
from transformers import Swinv2Model
from Swin import SwinTransformerV2


class CustomSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = SwinTransformerV2(img_size=256, window_size=8)
        self.swin.train()
        self.head = nn.Linear(in_features=768, out_features=512, bias=True)
        self.pool_128 = nn.AvgPool3d((3,2,2), stride=2)
        self.pool_64 = nn.AvgPool3d((96,1,1), stride=1)
        self.pool_32 = nn.AvgPool3d((192,1,1), stride=1)
        self.pool_16 = nn.AvgPool3d((384,1,1), stride=1)
        self.pool_8 = nn.AvgPool3d((768,1,1), stride=1)
        self.pool_4 = nn.AvgPool3d((768,2,2), stride=2)

    def forward(self, x):
        pooler_output, stage1_output, stage2_output, stage3_output, stage4_output = self.swin(x)
        size128_output, size64_output, size32_output, size16_output, size8_output, size4_output = self.pool_128(x), self.pool_64(stage1_output), self.pool_32(stage2_output), self.pool_16(stage3_output), self.pool_8(stage4_output), self.pool_4(stage4_output)
        x = self.head(pooler_output)
        return x, size128_output, size64_output, size32_output, size16_output, size8_output, size4_output

    