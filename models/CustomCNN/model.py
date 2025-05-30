import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, input_shape, config, num_classes=20):
        super(CustomCNN, self).__init__()
        self.convs = nn.ModuleList()
        in_channels = 1  # input è mono
        H, W = input_shape

        # Creo i layer convolutional dinamicamente
        for out_channels, kernel_size in zip(config['channels'], config['kernel_sizes']):
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            in_channels = out_channels
            H, W = H // 2, W // 2  # dopo MaxPool 2x2

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Calcolo flatten size
        flatten_dim = in_channels * H * W

        # Dense
        self.dropout = nn.Dropout(config['dropout'])
        self.fc1 = nn.Linear(flatten_dim, config['dense_hidden'])
        self.fc_out = nn.Linear(config['dense_hidden'], num_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        for conv in self.convs:
            x = self.pool(self.relu(conv(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc_out(x)
        return x
