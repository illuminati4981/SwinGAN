import torch
import torch.nn as nn
from transformers import SwinModel


class CustomSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.shallow_extractor = nn.Sequential(
          nn.Conv2d(3, 3, 9),
          nn.Conv2d(3, 3, 9),
          nn.Conv2d(3, 3, 9),
          nn.Conv2d(3, 3, 9)
        )
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.swin.train()
        self.head = nn.Linear(in_features=768, out_features=512, bias=True)

    def forward(self, x):
        x = self.shallow_extractor(x)
        x = self.swin(x).pooler_output
        x = self.head(x)
        return x

    