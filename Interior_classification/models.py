import torch
import torch.nn as nn
import timm 

class _BaseModel(nn.Module) :
    def __init__(self, **cfg) -> None:
        super(BaseModel, self).__init__()
        self.model = timm.create_model(model_name=cfg["model_name"], 
                                       num_classes=cfg["num_classes"], 
                                       pretrained=True)
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        
    def forward(self, x, div=False) :
        fm = self.model.forward_features(x)
        output = self.model.forward_head(fm)
        if div :
            return self.flatten(fm), output
        else :
            return output
        

class BaseModel(nn.Module) :
    def __init__(self, **cfg) -> None:
        super(BaseModel, self).__init__()
        self.model = timm.create_model(model_name=cfg["model_name"], 
                                       num_classes=cfg["num_classes"], 
                                       pretrained=True)
        
    def forward(self, x) :
        return self.model(x)

            