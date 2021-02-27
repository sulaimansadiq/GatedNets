import torch
from torch import nn


class VanillaCNN(nn.Module):

    def __init__(self):
        super(VanillaCNN, self).__init__()

        self.bottle_sz = 10
        self.num_classes = 10

        self.conv1 = nn.Conv2d(4, 3, (3, 3), padding=0)
        self.bn1 = nn.BatchNorm2d(3)
        self.activ1 = nn.ReLU()

        self.conv2 = nn.Conv2d(3, 10, (3, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(10)
        self.activ2 = nn.ReLU()

        self.fc1 = nn.Linear(2560, self.bottle_sz)
        self.activ3 = nn.ReLU()

        self.fc2 = nn.Linear(self.bottle_sz, self.num_classes)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activ2(out)

        out = out.view(out.shape[0], -1)      # flatten to in_chs dim vector
        out = self.fc1(out)
        out = self.activ3(out)

        out = self.fc2(out)

        return out