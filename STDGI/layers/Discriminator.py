'''two layer fully connected neural network,
which concatenates the embedding and rwo features of each pair
and outputs whether the pair is a positive or negative sample

Train three separate discriminators which compares the embedding to the raw feature
of the same node k steps in the future, k = 1, 3, 6
'''
import torch
import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self, n_h):
    super(Discriminator, self).__init__