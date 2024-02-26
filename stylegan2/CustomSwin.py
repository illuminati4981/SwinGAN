import torch
import torch.nn as nn
from transformers import Swinv2Model


class CustomSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.swin.train()
        self.head = nn.Linear(in_features=768, out_features=512, bias=True)
        self.pool_stage1 = nn.AvgPool3d((96,1,1), stride=1)
        self.pool_stage2 = nn.AvgPool3d((192,1,1), stride=1)
        self.pool_stage3 = nn.AvgPool3d((384,1,1), stride=1)
        self.pool_stage4 = nn.AvgPool3d((768,1,1), stride=1)

    def forward(self, x):
        return_dict = self.swin(x, output_hidden_states=True, return_dict=True)
        pooler_output = return_dict["pooler_output"]
        stage1_output, stage2_output, stage3_output, stage4_output = return_dict["reshaped_hidden_states"][:4]
        stage1_output, stage2_output, stage3_output, stage4_output = self.pool_stage1(stage1_output), self.pool_stage2(stage2_output), self.pool_stage3(stage3_output), self.pool_stage4(stage4_output)
        x = self.head(pooler_output)
        return x, stage1_output, stage2_output, stage3_output, stage4_output

    