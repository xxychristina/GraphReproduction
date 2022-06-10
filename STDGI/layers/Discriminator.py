'''two layer fully connected neural network,
which concatenates the embedding and two features of each pair
and outputs whether the pair is a positive or negative sample

Train three separate discriminators which compares the embedding to the raw feature
of the same node k steps in the future, k = 1, 3, 6
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from STDGI.layers.FeedForwardNetwork import FeedForwardNetwork

class Discriminator(nn.Module):
  def __init__(self, n_h):
    super(Discriminator, self).__init__()
    self.raw_d1 = FeedForwardNetwork(2)
    # self.raw_d3 = FeedForwardNetwork(2)
    # self.raw_d6 = FeedForwardNetwork(2)

    # self.node_d = FeedForwardNetwork(n_h)
  
  def forward(self, h, x, c):
    
    #(sc_1 => 1) : postivie sample
    #(sc_2 => 0) : negative sample
    sc_1 = self.raw_d1(torch.concat(h, x))
    sc_2 = self.raw_d1(torch.concat(h, c))

    logits = torch.cat((sc_1, sc_2), 1)
        
    return logits
