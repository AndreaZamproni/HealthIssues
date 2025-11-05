import torch
from torch import nn




class FeedForwardNet(nn.Module):
def __init__(self, in_features, hidden_layers, hidden_size, dropout_rate, num_classes):
super().__init__()
modules = [nn.Linear(in_features, hidden_size)]
if dropout_rate > 0:
modules.append(nn.Dropout(dropout_rate))
modules.append(nn.ReLU())
for _ in range(hidden_layers):
modules.append(nn.Linear(hidden_size, hidden_size))
if dropout_rate > 0:
modules.append(nn.Dropout(dropout_rate))
modules.append(nn.ReLU())
modules.append(nn.Linear(hidden_size, num_classes))
self.net = nn.Sequential(*modules)


def forward(self, x):
return self.net(x)