import torch.nn as nn
from layers import GCN, Discriminator
import numpy as np

class STDGI(nn.Module):
  def __init__(self, n_in, n_h):
    super(STDGI, self).__init__()
    self.gcn = GCN(n_in, n_h, 'prelu')
    self.disc = Discriminator(n_h)

  def forward(self, input, adj, msk):
    ''''
      x: features at time t
      xk1: features at time t + 1
      xk3: features at time t + 3
      xk6: features at time t + 6
    '''
    #graph embedding

    ret = 0
    #input: 12 * 207 * 2
    for i in range(11):
      x = input[i]
      xk1 = input[i+1]

      idx = np.random.permutation(len(xk1))
      ck1 = xk1[:, idx, :]   
    
      embed = self.gcn(x, adj)
  
      # time step k = 1, 3, 6
      ret += self.disc(embed, x, xk1, ck1)

    return ret, self.gcn

