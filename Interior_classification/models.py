import torch
import torch.nn as nn
import timm 

class BaseModel(nn.Module) :
    def __init__(self, **cfg) -> None:
        super(BaseModel, self).__init__()
        self.model = timm.create_model(model_name=cfg["model_name"], 
                                       num_classes=cfg["num_classes"], 
                                       pretrained=True)
    def forward(self, x) :
        return self.model(x)
    