from ctypes import util
import numpy as np
from models.STGCLRNN import STGCL
import torch
import torch.nn as nn
import torch.optim as optim
import util_dcrnn
import metric_dcrnn

class trainer():
  def __init__(self, scaler, device, adj, lr, encoder, decoder, num_nodes, r_f=30, c_rate=0.1, output_dim=1, max_grad_norm = 5):
    self.model = STGCL(device, encoder, decoder, num_nodes, output_dim).to(device)
    self.device = device
    self.adj = adj
    self.scaler = scaler
    self.r_f = r_f
    self.c_rate = c_rate
    self.max_grad_norm = max_grad_norm
    self.ploss = metric_dcrnn.masked_mae_loss(scaler, 0.0)
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
  
  def train(self, input, y, teacher_forcing_ratio, start_times):
    self.model.train()
    output, sum_x, sum_c = self.model(input, y, teacher_forcing_ratio)
    
    output = torch.transpose(output.view(12, 64, self.model.num_nodes,
                                                     self.model.output_dim), 0, 1)  # back to (50, 12, 207, 1)
    realy =  y[..., :1]
    self.optimizer.zero_grad()

    ploss = self.ploss(output, realy)
    closs = util_dcrnn.c_loss(self.device, sum_x, sum_c, start_times, self.r_f)
    loss = (1 - self.c_rate) * ploss + self.c_rate * closs
    loss.backward()

    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
    self.optimizer.step()
    return loss.item()
  
  def eval(self, input, y, teacher_forcing_ratio):
    self.model.eval()
    output, sum_x, sum_c = self.model(input, y, teacher_forcing_ratio)
    predict = torch.transpose(output.view(12, 64, self.model.num_nodes,
                                                     self.model.output_dim), 0, 1)  # back to (50, 12, 207, 1)
    realy =  y[..., :1]
    loss = self.ploss(predict, realy)
    return loss.item()

