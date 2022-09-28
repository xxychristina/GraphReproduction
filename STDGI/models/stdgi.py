import torch.nn as nn
from layers import GCN, Discriminator
import numpy as np

class STDGI(nn.Module):
  def __init__(self, n_in, n_h):
    super(STDGI, self).__init__()
    self.gcn = GCN(n_in, n_h)
    self.disc = Discriminator(n_h)

  def forward(self, input, adj):
    ''''
      x: features at time t
      xk1: features at time t + 1
      xk3: features at time t + 3
      xk6: features at time t + 6
    '''
    #graph embedding

    logits = 0
    #input: [64, 12, 207, 2]
    input = input.transpose(0, 1)
    #input: [12, 64, 207, 2]
    for i in range(11):
      # print(input.size())
      x = input[i]
      # print(x)
      xk1 = input[i+1]

      idx = np.random.permutation(len(xk1[1]))
      ck1 = xk1[:, idx, :]
    
      embed = self.gcn(x, adj)
  
      # time step k = 1, 3, 6
      # loss.shape = [64, 414])
      logit = self.disc(embed, xk1, ck1)
      # print(loss.size())
      # print(loss)
      logits += logit

    logits = logits / 11
    return logits

  def embed(self, seq, adj):
    h_1 = self.gcn(seq, adj)
    return h_1.detach()