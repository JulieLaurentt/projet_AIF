import torch
import torch.nn as nn
import torch.nn.functional as F

class MovieposterNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MovieposterNet, self).__init__()
        # Entrée : 3 canaux (RGB), 8 filtres, noyau 5x5
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calcul des dimensions après convolutions et pooling :
        # Input: (3, 224, 224)
        # Conv1: 224 - 5 + 1 = 220 -> Pool: 110
        # Conv2: 110 - 5 + 1 = 106 -> Pool: 53
        # Taille aplatie : 16 * 53 * 53 = 44944
        self.fc1 = nn.Linear(16 * 53 * 53, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_features(self, x):
        # Pour TensorBoard : extraction des caractéristiques avant les couches denses
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return x