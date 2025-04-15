import torch
import torch.nn as nn
import torchvision.models as models
from src import config

class TrashNetClassifier(nn.Module):
    def __init__(self):
        super(TrashNetClassifier, self).__init__()
        
        self.backbone = models.mobilenet_v2(pretrained=True)
        
        if config.FREEZE_BACKBONE:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
                
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.backbone.last_channel, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.Linear(256, config.NUM_CLASSES)
        )
        
    def forward(self, x):
        return self.backbone(x)