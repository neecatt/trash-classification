import torch
import torch.nn as nn
from src.model import TrashNetClassifier
from torchvision.models import resnet18, efficientnet_b0

class EnsembleModel(nn.Module):
    def __init__(self, num_classes=6):
        super(EnsembleModel, self).__init__()
        

        self.model1 = TrashNetClassifier(num_classes=num_classes)
        

        self.model2 = resnet18(pretrained=True)
        self.model2.fc = nn.Linear(self.model2.fc.in_features, num_classes)
        

        self.model3 = efficientnet_b0(pretrained=True)
        self.model3.classifier[1] = nn.Linear(self.model3.classifier[1].in_features, num_classes)
        
    def forward(self, x):

        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        

        return (out1 + out2 + out3) / 3