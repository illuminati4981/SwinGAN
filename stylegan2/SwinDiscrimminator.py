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
from transformers import Swinv2Model
            

class SwinDiscrimminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        for param in self.encoder.parameters():
            param.requires_grad = False
        #self.encoder = SwinV2Encoder(img_size=256, window_size=8)
        self.head = nn.Linear(768, 1, bias=True)
        self.flatten = nn.Flatten()
        #NO ACTIVATION FUNCTION BECAUSE WE USE BCEWITHLOGITLOSS
    def forward(self, x):
        outputs = self.encoder(x, return_dict=True, output_hidden_states=True)
        x = outputs.pooler_output
        x = self.flatten(x)
        x = self.head(x)
        
        return x