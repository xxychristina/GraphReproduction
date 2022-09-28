'''two layer fully connected neural network,
which concatenates the embedding and two features of each pair
and outputs whether the pair is a positive or negative sample

Train three separate discriminators which compares the embedding to the raw feature
of the same node k steps in the future, k = 1, 3, 6
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.feedForwardNetwork import FeedForwardNetwork

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
    # print(h.size())
    # print(h)
    # print(c.size())
    # print(c)
    cat1 = torch.concat((h, x), 2)
    cat2 = torch.concat((h, c), 2)
    sc_1 = self.raw_d1(cat1)
    sc_2 = self.raw_d1(cat2)

    #sc_1 shape: [64, 207, 1]
    sc_1 = torch.squeeze(sc_1, 2)
    sc_2 = torch.squeeze(sc_2, 2)
    logits = torch.cat((sc_1, sc_2), 1)
    # print(logits.size())
    # print(logits)
        
    # 64, 414
    return logits
