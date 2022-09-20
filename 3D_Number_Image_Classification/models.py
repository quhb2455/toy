import torch
from torch import nn
import torch.nn.functional as F
import timm

class CNN(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name=model_name, num_classes=10, pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x
