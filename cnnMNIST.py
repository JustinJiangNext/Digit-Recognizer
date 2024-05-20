import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class cnnMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, 5) 
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(12, 24, 4)
        self.conv2_drop = nn.Dropout2d()
        self.lin1 = nn.Linear(600, 100)
        self.lin2 = nn.Linear(100, 10)

    def forward(self, x):
        #1 x 28 x 28
        x = F.leaky_relu(self.conv1(x))
        #12 x 24 x 24
        x = self.pool(x) 
        #12 x 8 x 8
        x = self.conv2_drop(x)
        #12 x 8 x 8
        x = F.relu(self.conv2(x))
        #24 x 5 x 5
        x = torch.flatten(x, 1)
        #600
        x = F.relu(self.lin1(x))
        #50
        x = self.lin2(x)
        #10
        #return F.softmax(x)
        return x


