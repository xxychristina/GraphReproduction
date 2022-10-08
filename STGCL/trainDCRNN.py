from tracemalloc import start
import torch
import torch.nn as nn
import numpy as np
import augmentation as ag

class STGCL(nn.Module):
  def __init__(self, device, encoder, decoder):
    super(STGCL, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.projection_head = nn.Sequential(nn.Linear(20800, 256), nn.ReLU(inplace=True), nn.Linear(256, 256))

  def forward(self, input, target, teacher_forcing_ratio):

    input = torch.transpose(input, dim0=0, dim1=1)
    target = torch.transpose(target[..., :self._output_dim], dim0=0, dim1=1)
    target = torch.cat([self.GO_Symbol, target], dim=0)

    '''encode the original'''
    init_hidden_state = self.encoder.init_hidden(64)
    encoded = self.encoder(input, init_hidden_state)
    
    '''Predictive Learning:
        Generate the predict result with decoder
    '''
    outputs = self.decoder(target, encoded, teacher_forcing_ratio)
    #shape: [64, 1, 325, 12]

    '''Constrastive learning
        1. augmentation
        2. Readout: Summation funciton
        3, Project Head: linear + batch normalizationy + relu + linear
    '''
    corrupt_x = ag.temporalShift(input, 0.5)
    corrupt_encoded = self.encoder(corrupt_x, init_hidden_state)

    summaries_x = torch.squeeze(torch.sum(encoded, 2))
    summaries_c = torch.squeeze(torch.sum(corrupt_encoded, 2))

    return outputs[1:, :, :], self.projection_head(summaries_x), self.projection_head(summaries_c)