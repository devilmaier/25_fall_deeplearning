import torch
import torch.nn as nn

class TimeSeries1DCNN(nn.Module):
    def __init__(self, input_dim, output_dim=1, mean=None, std=None):
        super(TimeSeries1DCNN, self).__init__()
        
        if mean is not None and std is not None:
            self.register_buffer('input_mean', mean)
            self.register_buffer('input_std', std)
        else:
            self.register_buffer('input_mean', torch.zeros(1, 1, input_dim))
            self.register_buffer('input_std', torch.ones(1, 1, input_dim))

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.gap = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = (x - self.input_mean) / (self.input_std + 1e-9)
        x = x.permute(0, 2, 1) 
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.gap(x).squeeze(-1)
        out = self.fc(x)
        return out.squeeze()

