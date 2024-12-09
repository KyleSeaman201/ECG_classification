import torch
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self, num_classes):
        super(CNN2D, self).__init__()

        self.device=None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.num_classes = num_classes

        self.dropout = nn.Dropout(p=0.2)
        self.n = 8 # What should n be?

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(self.n)

        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=self.n * 2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(self.n * 2)

        self.conv3 = nn.Conv2d(in_channels=self.n * 2, out_channels=self.n * 4, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(self.n * 4)

        self.conv4 = nn.Conv2d(in_channels=self.n * 4, out_channels=self.n * 8, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(self.n * 8)

        self.leaky_relu = nn.LeakyReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.bn1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.bn2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.bn3(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.bn4(x)
        x = self.pool(x)

        H_TFR = x # feature representation needed for Fusion layer

        num_features = x.size(1) * x.size(2) * x.size(3)
        if self.fc1 is None:
            self.fc1 = nn.Linear(num_features, self.num_classes)
            self.fc1.to(self.device)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.dropout(x)

        return x, H_TFR