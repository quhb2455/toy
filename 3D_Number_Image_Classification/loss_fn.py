import torch
import torch.nn as nn

class FocalLoss(nn.Module) :
    def __init__(self, alpha=2, gamma=2, logits=False, reduction='none') :
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets) :
        ce_loss = nn.CrossEntropyLoss(reduction=self.reduction)(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction :
            return torch.mean(F_loss)
        else :
            return F_loss
