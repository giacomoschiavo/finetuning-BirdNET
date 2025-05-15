import torch.nn as nn

class MoreDeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super(MoreDeeperCNN, self).__init__()
        self.out_features = 512
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, self.out_features, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # output shape: [batch, 64, 4, 4]

        self.fc1 = nn.Linear(self.out_features * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        if x.ndim == 3:  # [batch, 256, 256] → aggiungi il canale
            x = x.unsqueeze(1)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))

        x = self.adaptive_pool(x)  # [B, 64, 4, 4]
        x = x.view(x.size(0), -1)  # Flatten → [B, 1024]

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
