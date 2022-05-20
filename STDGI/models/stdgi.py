import torch.nn as nn
from layers import GCN, Discriminator

class STDGI(nn.Module):
  def __init__(self, n_in, n_h, activation):
    super(STDGI, self).__init__()
    self.gcn = GCN(n_in, n_h, activation)
    self.disc = Discriminator(n_h)
    
  def forward(self, x, xk1, xk3, xk6, adj, msk):
    h = self.gcn(x, adj)
    c_1 = self.gcn(xk1, adj)
    c_2 = self.gcn(xk3, adj)
    c_3 = self.gcn(xk6, adj)

    ret = self.disc()

