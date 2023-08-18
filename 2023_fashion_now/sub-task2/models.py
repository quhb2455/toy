import torch
import torch.nn as nn
import timm 

class DivBaseModel(nn.Module) :
    def __init__(self, **cfg) -> None:
        super(DivBaseModel, self).__init__()
        self.model = timm.create_model(model_name=cfg["model_name"], 
                                       num_classes=cfg["num_classes"], 
                                       pretrained=False)
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
                                       pretrained=False)#True)
        
    def forward(self, x) :
        return self.model(x)


class MultiHeadBaseModel(nn.Module) :
    def __init__(self, **cfg) -> None:
        super(MultiHeadBaseModel, self).__init__()
        
        self.backbone = timm.create_model(model_name=cfg["model_name"], 
                                       num_classes=cfg['num_classes'], 
                                       global_pool='avg',
                                       pretrained=True)
        
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        
    def forward(self, x): 
        return self.flatten(self.backbone.forward_features(x))

class FullyConnectedLayer(nn.Module) :
    def __init__(self, 
                 base_dim, 
                 mid_dim,
                 num_classes,
                 drop_rate=0.2,
                 ) -> None:
        super(FullyConnectedLayer, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(base_dim, mid_dim),
            nn.Dropout(drop_rate),
            nn.GELU(),
            nn.Linear(mid_dim, num_classes)
        )
    
    def forward(self, x) :
        return self.layers(x)
        
class ClassifierHead(nn.Module) :
    def __init__(self, **cfg) -> None:
        super(ClassifierHead, self).__init__()
        
        self.head = FullyConnectedLayer(
            base_dim=3, 
            mid_dim=64, 
            num_classes=18)
    
    def forward(self, x):
        x = self.head(x)
        # x.masked_fill_(mask, -10000.)
        return x 
    
class ClassifierHead_2(nn.Module) :
    def __init__(self, **cfg) -> None:
        super(ClassifierHead_2, self).__init__()
        
        self.head = FullyConnectedLayer(
            base_dim=cfg["base_dim"], 
            mid_dim=cfg["mid_dim"], 
            num_classes=5)
    
    def forward(self, x, mask):
        x = self.head(x)
        # x.masked_fill_(mask, -10000.)
        return x 

class ClassifierHead_3(nn.Module) :
    def __init__(self, **cfg) -> None:
        super(ClassifierHead_3, self).__init__()
        
        self.head = FullyConnectedLayer(
            base_dim=cfg["base_dim"], 
            mid_dim=cfg["mid_dim"], 
            num_classes=5)
    
    def forward(self, x, mask):
        x = self.head(x)
        # x.masked_fill_(mask, -10000.)
        return x 

class ClassifierHead_4(nn.Module) :
    def __init__(self, **cfg) -> None:
        super(ClassifierHead_4, self).__init__()
        
        self.head = FullyConnectedLayer(
            base_dim=cfg["base_dim"], 
            mid_dim=cfg["mid_dim"], 
            num_classes=2)
    
    def forward(self, x, mask):
        x = self.head(x)
        # x.masked_fill_(mask, -10000.)
        return x 

class ClassifierHead_5(nn.Module) :
    def __init__(self, **cfg) -> None:
        super(ClassifierHead_5, self).__init__()
        
        self.head = FullyConnectedLayer(
            base_dim=cfg["base_dim"], 
            mid_dim=cfg["mid_dim"], 
            num_classes=2)
    
    def forward(self, x, mask):
        x = self.head(x)
        # x.masked_fill_(mask, -10000.)
        return x 

class ClassifierHead_6(nn.Module) :
    def __init__(self, **cfg) -> None:
        super(ClassifierHead_6, self).__init__()
        
        self.head = FullyConnectedLayer(
            base_dim=cfg["base_dim"], 
            mid_dim=cfg["mid_dim"], 
            num_classes=1)
    
    def forward(self, x, mask):
        x = self.head(x)
        # x.masked_fill_(mask.unsqueeze(1), -10000.)
        return x 