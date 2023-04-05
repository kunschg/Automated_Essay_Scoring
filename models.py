import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from tqdm import tqdm
import os

#################################
# Simple 1D convolutional network
#################################

class ConvNet1D(nn.Module):
    def __init__(self, input_channels, depth, kernel_size, max_essay_length, skip_connections):
        super().__init__() 
        self.conv_channels = [input_channels] + [input_channels]*depth + [1]
        self.kernel_size = kernel_size
        self.max_essay_length = max_essay_length
        self.skip_connections = skip_connections
        
        # Convolutions
        self.convs = nn.ModuleList([nn.Conv1d(
            in_channels = self.conv_channels[i], 
            out_channels = self.conv_channels[i+1],
            kernel_size = self.kernel_size,
            padding = 'same') for i in range(len(self.conv_channels)-1)])

        # Full connections
        self.fc_essay = nn.Linear(self.max_essay_length*self.conv_channels[-1], 1)
        self.fc_essay_set = nn.Linear(8, 1)
        self.fc_final = nn.Linear(2,1)
        
    def forward(self, essays, essay_sets):
        # Essay branch
        x1 = essays

        if self.skip_connections:
            for i, conv in enumerate(self.convs): 
                if self.conv_channels[i] == self.conv_channels[i+1]:
                    x1 = F.relu(conv(x1)) + x1
                else:
                    x1 = F.relu(conv(x1))

        else:   
            for conv in self.convs:
                x1 = F.relu(conv(x1))

        x1 = F.relu(self.fc_essay(x1))
        x1 = x1.view(x1.shape[0],1)
        
        # Essay set branch
        x2 = F.one_hot(essay_sets-1, num_classes=8).view(essay_sets.shape[0],1,8).float()
        x2 = F.relu(self.fc_essay_set(x2))
        x2 = x2.view(x2.shape[0],1)
        
        # Concatenate and predict
        x = torch.cat((x1, x2), 1)
        x = self.fc_final(x)
        x = torch.sigmoid(x)
        
        return x