from tracemalloc import start
import torch
import torch.nn as nn
import numpy as np
import augmentation as ag

class STGCL(nn.Module):
  def __init__(self, encoder, decoder):
    super(STGCL, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.projection_head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, 256))

  def forward(self, input):
    '''encode the original'''
    encoded = self.encoder(input)
    #shape: [64, 256, 325, 1]
    '''Predictive Learning:
        Generate the predict result with decoder
    '''
    output = self.decoder(encoded)
    #shape: [64, 1, 325, 12]

    '''Constrastive learning
        1. augmentation
        2. Readout: Summation funciton
        3, Project Head: linear + batch normalizationy + relu + linear
    '''
    corrupt_x = ag.temporalShift(input, 0.5)
    corrupt_encoded = self.encoder(corrupt_x)

    summaries_x = torch.squeeze(torch.sum(encoded, 2))
    summaries_c = torch.squeeze(torch.sum(corrupt_encoded, 2))

    return output, self.projection_head(summaries_x), self.projection_head(summaries_c)