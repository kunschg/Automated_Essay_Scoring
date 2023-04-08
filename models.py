import numpy as np
import math
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
    def __init__(self, input_channels, depth, kernel_size, max_essay_length, skip_connections, batch_norm):
        super().__init__()
        self.input_channels = input_channels
        self.depth = depth
        self.conv_channels = [self.input_channels] + [self.input_channels]*self.depth + [1]
        self.kernel_size = kernel_size
        self.max_essay_length = max_essay_length
        self.skip_connections = skip_connections
        self.batch_norm = batch_norm
        
        # Convolutions
        self.convs = nn.ModuleList([nn.Conv1d(
            in_channels = self.conv_channels[i], 
            out_channels = self.conv_channels[i+1],
            kernel_size = self.kernel_size,
            padding = 'same') for i in range(len(self.conv_channels)-1)])
        self.bnorm1d = nn.BatchNorm1d(self.input_channels)

        # Full connections
        self.fc_essay = nn.Linear(self.max_essay_length*self.conv_channels[-1], 1)
        self.fc_essay_set = nn.Linear(8, 1)
        self.fc_final = nn.Linear(2,1)
        
    def forward(self, essays, essay_sets):
        # Essay branch
        x1 = essays

        if self.skip_connections:
            if self.batch_norm:
                for i, conv in enumerate(self.convs): 
                    if self.conv_channels[i] == self.conv_channels[i+1]:
                        x1 = F.relu(self.bnorm1d(conv(x1))) + x1
                    else:
                        x1 = F.relu(conv(x1))                
            else:
                for i, conv in enumerate(self.convs): 
                    if self.conv_channels[i] == self.conv_channels[i+1]:
                        x1 = F.relu(conv(x1)) + x1
                    else:
                        x1 = F.relu(conv(x1))
        else:   
            if self.batch_norm:
                for i, conv in enumerate(self.convs):
                    if self.conv_channels[i] == self.conv_channels[i+1]:
                        x1 = F.relu(self.bnorm1d(conv(x1)))
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
        x = torch.sigmoid(self.fc_final(x))
        x = x.view(x.shape[0])
        
        return x

################
# ConvNet1D (v2)
################

"""
Modification: essay_set information embedded and concatenated before the convolutions
"""

class ConvNet1Dv2(nn.Module):
    def __init__(self, input_channels, depth, kernel_size, max_essay_length, skip_connections, batch_norm):
        super().__init__()
        self.input_channels = input_channels
        self.depth = depth
        self.conv_channels = [self.input_channels+1]*(self.depth+1) + [1]
        self.kernel_size = kernel_size
        self.max_essay_length = max_essay_length
        self.skip_connections = skip_connections
        self.batch_norm = batch_norm
        
        # Convolutions
        self.convs = nn.ModuleList([nn.Conv1d(
            in_channels = self.conv_channels[i], 
            out_channels = self.conv_channels[i+1],
            kernel_size = self.kernel_size,
            padding = 'same') for i in range(len(self.conv_channels)-1)])
        self.bnorm1d = nn.BatchNorm1d(self.input_channels+1)

        # Full connections
        self.fc_emb = nn.Linear(8, self.max_essay_length)
        self.fc_final = nn.Linear(self.max_essay_length,1)
        
    def forward(self, essays, essay_sets):
        x1 = essays
        x2 = F.one_hot(essay_sets-1, num_classes=8).view(essay_sets.shape[0],1,8).float()
        x2 = F.relu(self.fc_emb(x2))
        x = torch.cat((x1, x2), 1)

        if self.skip_connections:
            if self.batch_norm:
                for i, conv in enumerate(self.convs): 
                    if self.conv_channels[i] == self.conv_channels[i+1]:
                        x = F.relu(self.bnorm1d(conv(x))) + x
                    else:
                        x = F.relu(conv(x))                
            else:
                for i, conv in enumerate(self.convs): 
                    if self.conv_channels[i] == self.conv_channels[i+1]:
                        x = F.relu(conv(x)) + x
                    else:
                        x = F.relu(conv(x))
        else:   
            if self.batch_norm:
                for i, conv in enumerate(self.convs):
                    if self.conv_channels[i] == self.conv_channels[i+1]:
                        x = F.relu(self.bnorm1d(conv(x)))
                    else:
                        x = F.relu(conv(x))
            else:            
                for conv in self.convs:
                    x = F.relu(conv(x))

        x = x.view(x.shape[0],-1)
        x = torch.sigmoid(self.fc_final(x))
        x = x.view(x.shape[0])
        
        return x

################
# ConvNet1D (v3)
################

"""
Modification: essay_set information embedded and concatenated right after the last convolution
"""

class ConvNet1Dv3(nn.Module):
    def __init__(self, input_channels, depth, kernel_size, max_essay_length, skip_connections, batch_norm):
        super().__init__()
        self.input_channels = input_channels
        self.depth = depth
        self.conv_channels = [self.input_channels] + [self.input_channels]*self.depth + [1]
        self.kernel_size = kernel_size
        self.max_essay_length = max_essay_length
        self.skip_connections = skip_connections
        self.batch_norm = batch_norm
        
        # Convolutions
        self.convs = nn.ModuleList([nn.Conv1d(
            in_channels = self.conv_channels[i], 
            out_channels = self.conv_channels[i+1],
            kernel_size = self.kernel_size,
            padding = 'same') for i in range(len(self.conv_channels)-1)])
        self.final_conv = nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = self.kernel_size, padding = 'same')
        self.bnorm1d = nn.BatchNorm1d(self.input_channels)

        # Full connections
        self.fc_essay_set = nn.Linear(8, self.max_essay_length)
        self.fc_final = nn.Linear(self.max_essay_length,1)
        
    def forward(self, essays, essay_sets):
        # Essay branch
        x1 = essays

        if self.skip_connections:
            if self.batch_norm:
                for i, conv in enumerate(self.convs): 
                    if self.conv_channels[i] == self.conv_channels[i+1]:
                        x1 = F.relu(self.bnorm1d(conv(x1))) + x1
                    else:
                        x1 = F.relu(conv(x1))                
            else:
                for i, conv in enumerate(self.convs): 
                    if self.conv_channels[i] == self.conv_channels[i+1]:
                        x1 = F.relu(conv(x1)) + x1
                    else:
                        x1 = F.relu(conv(x1))
        else:   
            if self.batch_norm:
                for i, conv in enumerate(self.convs):
                    if self.conv_channels[i] == self.conv_channels[i+1]:
                        x1 = F.relu(self.bnorm1d(conv(x1)))
                    else:
                        x1 = F.relu(conv(x1))
            else:            
                for conv in self.convs:
                    x1 = F.relu(conv(x1))

        # Essay set branch
        x2 = F.one_hot(essay_sets-1, num_classes=8).view(essay_sets.shape[0],1,8).float()
        x2 = F.relu(self.fc_essay_set(x2))
        
        # Concatenation
        x = torch.cat((x1,x2), 1)
        x = F.relu(self.final_conv(x))
        x = x.view(x.shape[0],-1)
        x = torch.sigmoid(self.fc_final(x))
        x = x.view(x.shape[0])
        
        return x
    
###################
# Transformer model
###################

class PositionalEncoding(nn.Module):
    def __init__(self, input_channels, max_essay_length, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_essay_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_channels, 2) * (-math.log(10000.0) / input_channels))
        pe = torch.zeros(max_essay_length, input_channels)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(1,0)        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_channels, max_essay_length, num_heads, hidden_dim, num_layers, dropout):
        super().__init__()

        self.embedding_dim = input_channels
        self.max_essay_length = max_essay_length
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pos_encoder = PositionalEncoding(input_channels, max_essay_length, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_channels, num_heads, hidden_dim, dropout), num_layers)
        self.fc_essay = nn.Linear(input_channels * max_essay_length, 1)
        self.fc_essay_set = nn.Linear(8,1)
        self.fc_final = nn.Linear(2,1)

    def forward(self, essays, essay_sets):
        # Essay branch
        x1 = self.pos_encoder(essays).transpose(1,2)
        x1 = self.transformer_encoder(x1)
        x1 = x1.view(-1, self.embedding_dim * self.max_essay_length)
        x1 = self.fc_essay(x1)  

        # Essay_set branch
        x2 = F.one_hot(essay_sets-1, num_classes=8).view(essay_sets.shape[0],8).float()
        x2 = self.fc_essay_set(x2)

        # Concatenation and prediction
        x = torch.cat((x1, x2), axis = 1)
        x = torch.sigmoid(self.fc_final(x).view(-1))

        return x