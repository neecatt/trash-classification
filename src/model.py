import torch
import torch.nn as nn
import torchvision.models as models
from src import config
from torchvision.models import mobilenet_v2


class TrashNetClassifier(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(TrashNetClassifier, self).__init__()
        self.backbone = mobilenet_v2(pretrained=True)
        

        if config.FREEZE_BACKBONE:
            for param in list(self.backbone.parameters())[:-8]:
                param.requires_grad = False
            

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE),  
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x