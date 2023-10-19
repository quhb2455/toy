import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 

class ChannelAttention(nn.Module) :
    def __init__(self, in_ch, r=16) :
        super(ChannelAttention, self).__init__()
        # channel 단위로 pooling을 진행하기 위해 Maxpooling에는 input channel을 kernel_size로 넣어줌
        # self.maxpool = nn.MaxPool2d(kernel_size=size)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch//r),
            nn.ReLU(),
            nn.Linear(in_ch//r, in_ch)
        )
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
    def forward(self, x) :
        # max_x = self.maxpool(x)
        # avg_x = self.avgpool(x)
        
        # init에서 선언하면 feature map의 H, W 를 알아야하는데
        # torch.nn.functional 을 사용하면 바로 forward에서 사용 가능하기 때문에
        # feature map 사이즈가 필요 없음.
        max_x = F.max_pool2d(x, kernel_size=x.shape[2]).view(x.shape[0], -1)
        avg_x = F.adaptive_avg_pool2d(x, output_size=(1,1)).view(x.shape[0], -1)
        
        x = x * self.sigmoid(self.fc(max_x).unsqueeze(2).unsqueeze(3) + self.fc(avg_x).unsqueeze(2).unsqueeze(3))
        x = self.flatten(x)
        return x
    
class DivBaseModel(nn.Module) :
    def __init__(self, in_chans=3, **cfg) -> None:
        super(DivBaseModel, self).__init__()
        self.model = timm.create_model(model_name=cfg["model_name"], 
                                       num_classes=cfg["num_classes"], 
                                       in_chans=in_chans,
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
                                       pretrained=False)
        
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

class ColorClassifierHead(nn.Module) :
    def __init__(self, base_dim, mid_dim, num_classes) -> None:
        super(ColorClassifierHead, self).__init__()
        
        self.head = FullyConnectedLayer(
            base_dim=base_dim, 
            mid_dim=mid_dim, 
            num_classes=num_classes)
        self.num_classes = num_classes
        
    def forward(self, x):#, masking, mode):
        # if mode == 'upper' :
        #     mask = masking.ge(0.5)
        # else :
        #     mask = masking.le(0.5)
        out = self.head(x)
        # out.masked_fill_(mask, -100.0)
        return out

class BinaryClassifierHead(nn.Module) :
    def __init__(self, base_dim, mid_dim) -> None:
        super(BinaryClassifierHead, self).__init__()

        self.head = FullyConnectedLayer(
            base_dim=base_dim, 
            mid_dim=mid_dim, 
            num_classes=1)
        
    def forward(self, x) :
        x = self.head(x)
        x = F.sigmoid(x)
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
        x.masked_fill_(mask, -10000.)
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