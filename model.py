import torch.nn as nn
import torchvision.models as models

class MovieposterNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Charger ResNet50 pré-entraîné
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Geler tous les poids
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Remplacer uniquement la dernière couche
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)