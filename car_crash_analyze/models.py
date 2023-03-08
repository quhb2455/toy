import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import timm

import timm
from timm.models.beit import _create_beit
from timm.models.vision_transformer import _create_vision_transformer, eva_large_patch14_336
from ML_Decoder.src_files.ml_decoder.ml_decoder import MLDecoder

    
class EVA(nn.Module) :
    def __init__(self, num_classes, enc_dim=1024, pretrained_path="./ckpt/eva_large_patch14_336.in22k_ft_in22k_in1k.bin") -> None:
        super(EVA, self).__init__()
        
        self.model = _create_vision_transformer(
            'eva_large_patch14_336', 
            **dict(patch_size=14, embed_dim=enc_dim, depth=24, num_heads=16, global_pool='avg'))
        self.model.load_state_dict(torch.load(pretrained_path))
        self.head = nn.Linear(self.model.head.out_features, num_classes, bias=True)
        # self.head = MLDecoder(num_classes = num_classes, 
        #                         initial_num_features = enc_dim, 
        #                         num_of_groups = -1,
        #                         decoder_embedding = 768, 
        #                         zsl = 0)
        
    def forward(self, x) :
        x = self.model(x)
        x = self.head(x)
        return x
    
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
    
class SlowFast(nn.Module) :
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        self.head = nn.Linear(self.model.blocks[-1].proj.out_features, num_classes)
    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x

    
     
class _WEATHER_MODEL(nn.Module) :
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.backbone = timm.models.efficientnet_b0(pretrained=True, features_only=True, out_indices=(2,3,4))
        self.conv1 = nn.Sequential(
            nn.Conv2d(40, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(112, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        self.conv3 = nn.Sequential(
            # nn.Conv2d(320, 256, kernel_size=5, stride=2, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(576, 128),
            nn.ELU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x) :
        outpus = self.backbone(x)
        
        o1 = self.conv1(outpus[0])
        o2 = self.conv2(outpus[1])
        o3 = self.conv3(outpus[2])
        
        output = self.cls_head(torch.cat([o1, o2, o3], dim=1))
        return output 

class WEATHER_MODEL(nn.Module) :
    def __init__(self, num_classes) -> None:
        super().__init__()
        # self.backbone = timm.models.efficientnet_b0(pretrained=True, features_only=True, out_indices=(2,3,4))
        self.backbone = timm.models.efficientnetv2_rw_s(pretrained=True, features_only=True, out_indices=(2,3,4))
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(160, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        self.conv3 = nn.Sequential(
            # nn.Conv2d(320, 256, kernel_size=5, stride=2, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(528, 128),
            nn.ELU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x) :
        outpus = self.backbone(x)
        
        o1 = self.conv1(outpus[0])
        o2 = self.conv2(outpus[1])
        o3 = self.conv3(outpus[2])
        
        output = self.cls_head(torch.cat([o1, o2, o3], dim=1))
        return output 

class __WEATHER_MODEL(nn.Module) :
    def __init__(self, num_classes) -> None:
        super().__init__()
        # self.backbone = timm.models.efficientnet_b0(pretrained=True, features_only=True, out_indices=(2,3,4))
        self.backbone = timm.models.convnext_base_384_in22ft1k(pretrained=True, features_only=True, out_indices=(1,2,3))
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        self.conv3 = nn.Sequential(
            # nn.Conv2d(320, 256, kernel_size=5, stride=2, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ELU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x) :
        outpus = self.backbone(x)
        
        o1 = self.conv1(outpus[0])
        o2 = self.conv2(outpus[1])
        o3 = self.conv3(outpus[2])
        
        output = self.cls_head(torch.cat([o1, o2, o3], dim=1))
        return output 