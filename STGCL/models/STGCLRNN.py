from tracemalloc import start
import torch
import torch.nn as nn
import numpy as np
import augmentation as ag

class STGCL(nn.Module):
  def __init__(self, device, encoder, decoder, num_nodes, output_dim):
    super(STGCL, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.num_nodes = num_nodes
    self.output_dim = output_dim
    self.projection_head = nn.Sequential(nn.Linear(13248, 256), nn.ReLU(inplace=True), nn.Linear(256, 256))
    self.GO_Symbol = torch.zeros(1, 64, self.num_nodes * self.output_dim, 1)

  def forward(self, input, target, teacher_forcing_ratio):
    input = torch.transpose(input, dim0=0, dim1=1)
    target = torch.transpose(target[..., :self.output_dim], dim0=0, dim1=1)
    target = torch.cat([self.GO_Symbol, target], dim=0).cuda()

    '''encode the original'''
    init_hidden_state = self.encoder.init_hidden(64)
    encoded, _ = self.encoder(input, init_hidden_state)
    
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
    corrupt_encoded, _ = self.encoder(corrupt_x, init_hidden_state)

    layers = torch.stack((encoded[0], encoded[1]))
    corrupt_layers = torch.stack((corrupt_encoded[0], corrupt_encoded[1]))

    summaries_x = torch.squeeze(torch.sum(layers, 0))
    summaries_c = torch.squeeze(torch.sum(corrupt_layers, 0))

    return outputs, self.projection_head(summaries_x), self.projection_head(summaries_c)