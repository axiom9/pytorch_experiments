import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(num_features=10)
        self.pooling = nn.AvgPool2d(2,2)
        self.conv_2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=20)
        self.linear_1 = nn.Linear(in_features=20*5*5, out_features=312)
        self.bn_l1 = nn.BatchNorm1d(num_features=312)
        self.linear_2 = nn.Linear(in_features=312, out_features=128)
        self.bn_l2 = nn.BatchNorm1d(num_features=128)
        self.linear_3 = nn.Linear(128, 64)
        self.bn_l3 = nn.BatchNorm1d(num_features=64)
        self.linear_4 = nn.Linear(64, config.num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.pooling(self.conv_1(x))
        x = F.relu(self.bn1(input=x))
        # x = F.relu(x)
        x = self.pooling(self.conv_2(x))
        x = F.relu(self.bn2(input=x))
        x = F.relu(x)
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.bn_l1(input=self.linear_1(x)))
        # x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = F.relu(self.bn_l2(input=self.linear_2(x)))
        # x = F.relu(self.linear_2(x))
        x = F.relu(self.bn_l3(input=self.linear_3(x)))
        # x = F.relu(self.linear_3(x))
        x = self.dropout(x)
        x = F.relu(input=self.linear_4(x))
        return x

    @torch.no_grad()
    def weight_reset(self, m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()
    
