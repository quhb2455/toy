import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import timm

class NN(nn.Module):
    def __init__(self, model_name, num_classes, hid_bs=144, bs=32):
        super().__init__()
        self.model = timm.create_model(model_name=model_name, num_classes=0, pretrained=True, global_pool='')
        self.avgpool1d = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(hid_bs, num_classes)
        self.bs = bs

    def forward(self, x):
        outputs = []
        for i in range(int(x.shape[0] / self.bs)) :
            output = self.model(x[i * self.bs: (i + 1) * self.bs]).detach().cpu()
            output = self.avgpool1d(output)
            outputs.append(output.flatten(1))

        batch_output = torch.cat(outputs, dim=0).cuda()
        x = self.linear(batch_output)
        return x, batch_output

class _NN(nn.Module):
    def __init__(self, model_name, num_classes, hid_bs=128, bs=32):
        super().__init__()
        self.model = timm.create_model(model_name=model_name, num_classes=num_classes, pretrained=True)
        self.linear = nn.Linear(hid_bs, num_classes)
        self.bs = bs

    def forward(self, x):
        outputs = []
        for i in range(int(x.shape[0] / self.bs)) :
            outputs.append(self.model(x[i*self.bs : (i+1)*self.bs]).detach().cpu())

        batch_output = torch.cat(outputs, dim=1).cuda().flatten()
        x = self.linear(batch_output)
        return x, batch_output

class simple_NN(nn.Module):
    def __init__(self, model_name, num_classes, in_chans=3):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name, 
            num_classes=num_classes, 
            pretrained=True, 
            in_chans=in_chans)

    def forward(self, x):
        x = self.model(x)
        return x

