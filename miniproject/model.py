import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # ResNet core ideas: Skip Connection
        return F.relu(out)

class GomokuResNet(nn.Module):
    def __init__(self, num_blocks=5, num_filters=64):
        super(GomokuResNet, self).__init__()
        # Initial convolutional layer: maps 3 channels (black, white, empty) to 64 hidden feature channels
        self.conv_initial = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(num_filters)
        
        # Stack residual blocks
        self.res_blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_blocks)])
        
        # Policy Head: outputs the probability of playing at 225 positions
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 15 * 15, 225)

    def forward(self, x):
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        for block in self.res_blocks:
            x = block(x)
            
        # Policy head output
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * 15 * 15)  # Flatten
        p = self.policy_fc(p)
        
        # Return log probabilities (used for calculating cross-entropy loss and prior probabilities in MCTS)
        return F.log_softmax(p, dim=1)