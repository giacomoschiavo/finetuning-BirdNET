import torch.nn as nn

class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # Adaptive pooling: forza le feature map a 4x4
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # output shape: [batch, 64, 4, 4]

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.ndim == 3:  # [batch, 256, 256] → aggiungi il canale
            x = x.unsqueeze(1)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))


        x = self.adaptive_pool(x)  # [B, 64, 4, 4]
        x = x.view(x.size(0), -1)  # Flatten → [B, 1024]

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
