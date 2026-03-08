import torch
import torch.nn as nn
import torch.nn.functional as F

class MovieposterNet(nn.Module):
    def __init__(self):
        super(MovieposterNet, self).__init__()
        self.conv1 = nn.Conv2d(1,8,5)
        self.conv2 = nn.Conv2d(8,16, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # First convolution followed by
        x = self.pool(x)                # a relu activation and a max pooling#
        x = F.relu(self.conv2(x))
        x = self.pool(x)      
        x=torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return x