from torch import nn
import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, 3, padding=0),  # 126
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.03),

        )

        
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0, stride=2),  # 124
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.03),
            nn.MaxPool2d(2,2), # 62
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 10, 3, padding=0, stride=1),  # 60
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.03)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0),  # 58
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.03),
            nn.MaxPool2d(2,2),  # 27
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 10, 3),  # 56
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.03)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(10, 10, 3),  # 4
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.03)
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=9)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(10, 2, 1),  #1
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1,2)
        return F.log_softmax(x, dim=-1)