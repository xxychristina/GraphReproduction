'''
  Fully connected two layer neural network
'''

import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
  def __init__(self, n_h):
    super(FeedForwardNetwork, self).__init__()
    self.full1 = nn.Linear(66, 6)
    # self.full1 = nn.Linear(66, 6)
    self.full2 = nn.Linear(6, 1)
  
  def forward(self, x):
    t = F.relu(self.full1(x))
    t = F.sigmoid(self.full2(t))
    return t