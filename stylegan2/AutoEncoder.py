import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
from Swin import SwinV2Encoder, SwinV2Decoder


class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, 3, padding=1), # (8, 8)
            nn.BatchNorm2d(out_features, affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features=512, out_features=768, bias=True),
            nn.LayerNorm(768),
            nn.ReLU()
        )
        self.conv1to4 = nn.ConvTranspose2d(768, 768*2, 4)
        self.conv4x4 = ConvBlock(768*2+1, 768*2)
        self.conv4to8 = nn.ConvTranspose2d(768*2, 768, 3, padding=1, 
                            stride=2, output_padding=1)
        self.conv8x8 = ConvBlock(768+1, 768)
        self.conv8to16 = nn.ConvTranspose2d(768, 384, 3, padding=1, 
                            stride=2, output_padding=1)
        self.conv16x16 = ConvBlock(384+1, 384)
        self.conv16to32 = nn.ConvTranspose2d(384, 192, 3, padding=1, 
                            stride=2, output_padding=1)
        self.conv32x32 = ConvBlock(192+1, 192)
        self.conv32to64 = nn.ConvTranspose2d(192, 96, 3, padding=1, 
                            stride=2, output_padding=1)
        self.conv64x64 = ConvBlock(96+1, 96)
        self.conv64to128 = nn.ConvTranspose2d(96, 48, 3, padding=1, 
                            stride=2, output_padding=1)
        self.conv128x128 = ConvBlock(48+1, 48)
        self.conv128to256 = nn.ConvTranspose2d(48, 24, 3, padding=1, 
                            stride=2, output_padding=1)
        self.conv256x256 = ConvBlock(24, 24)
        
        self.output_conv = nn.ConvTranspose2d(24, 3, 3, padding=1)

    def forward(self, x, noises):
        size128_output, size64_output, size32_output, size16_output, size8_output, size4_output = noises
        x = self.linear(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.conv1to4(x)
        x = torch.concat((x,size4_output), dim=1)
        x = self.conv4x4(x)
        x = self.conv4to8(x)
        x = torch.concat((x,size8_output), dim=1)
        x = self.conv8x8(x)
        x = self.conv8to16(x)
        x = torch.concat((x,size16_output), dim=1)
        x = self.conv16x16(x)
        x = self.conv16to32(x)
        x = torch.concat((x,size32_output), dim=1)
        x = self.conv32x32(x)
        x = self.conv32to64(x)
        x = torch.concat((x,size64_output), dim=1)
        x = self.conv64x64(x)
        x = self.conv64to128(x)
        x = torch.concat((x,size128_output), dim=1)
        x = self.conv128x128(x)
        x = self.conv128to256(x)
        x = self.conv256x256(x)
        x = self.output_conv(x)
        return x
            

class SwinAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SwinV2Encoder(img_size=256, window_size=8)
        self.decoder = SwinV2Decoder(img_size=256, window_size=8, num_classes=3)

    def forward(self, x):
        x, hidden_states = self.encoder(x)
        x = self.decoder(x, hidden_states)
        
        return x