import torch.nn as nn
from Swin import SwinTransformer
from stylegan2.training.networks import Generator


class SwinGan(nn.Module):
    def __init__(self, swin_transformer: SwinTransformer, gan_generator: Generator):
        self.swin_transformer = swin_transformer
        self.gan_generator = gan_generator

    def forward(self, x):
        x = self.swin_transformer(x)
        x = self.gan_generator(x)
        return x
