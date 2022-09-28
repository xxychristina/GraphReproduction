'''
  Fully connected two layer neural network
'''

import torch.nn as nn
import torch.nn.functional as F

class ProjectHead(nn.Module):
  def __init__(self, n_h):
    super(ProjectHead, self).__init__()
    self.layer1 = nn.Linear(n_h, n_h)
    self.batchNorm = nn.BatchNorm1d(n_h)
    self.layer2 = nn.Linear(n_h, n_h)

  
  def forward(self, x):
    x = self.layer1(x)
    x = self.batchNorm(x)
    x = self.layer2(x)
    return x