from models import STDGI
import torch.optim as optim
import torch.nn as nn
import torch


class trainer():
  def __init__(self, ft_size, hid_units, adj, lr):
    # n_in, n_h, activation
    self.model = STDGI(ft_size, hid_units)
    self.adj = adj
    self.lr = lr
    self.b_xent = nn.BCEWithLogitsLoss()

  
  def train(self, intput):
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    self.optimizer.zero_grad()

    self.model.train()
    loss, embed = self.model(intput)

    loss = loss/11

    batch_size = len(input[0])
    nb_nodes = len(input[1])

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    loss = self.b_xent(loss, lbl)
    loss.backward()
    self.optimizer.step()

    return loss, embed




