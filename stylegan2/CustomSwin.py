import torch.nn as nn
from transformers import SwinModel


class CustomSwin(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_features = 768
    self.num_classes = 512
    self.shallow_extractor = nn.Sequential(
      nn.Conv2d(3, 3, 9),
      nn.Conv2d(3, 3, 9),
      nn.Conv2d(3, 3, 9),
      nn.Conv2d(3, 3, 9)
    )
    self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    self.head = nn.Linear(self.num_features, self.num_classes)

  def forward(self, x):
    x = self.shallow_extractor(x)
    x = self.swin(x)
    x = self.head(x)
    return x

    