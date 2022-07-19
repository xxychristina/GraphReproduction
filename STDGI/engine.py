from models import STDGI
import torch.optim as optim
import torch.nn as nn
import torch
import util


class trainer():
  def __init__(self, ft_size, hid_units, adj, lr):
    # n_in, n_h, activation
    self.model = STDGI(ft_size, hid_units)
    self.adj = adj
    self.lr = lr
    self.b_xent = nn.BCELoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr) #first 20
    self.optimizer1 = optim.Adam(self.model.parameters(), lr=self.lr * 0.9) #first 50
    self.optimizer2 = optim.Adam(self.model.parameters(), lr=self.lr * 0.8) #first 80
    self.optimizer3 = optim.Adam(self.model.parameters(), lr=self.lr * 0.7) #first 100


  def train(self, input, epoch):
    optimizer = None
    if epoch < 20:
      optimizer = self.optimizer
    elif epoch > 20 and epoch < 50:
      optimizer = self.optimizer1
    elif epoch < 80:
      optimizer = self.optimizer2
    else:
      optimizer = self.optimizer3

    optimizer.zero_grad()

    self.model.train()
    # input = nn.functional.pad(input,(1,0,0,0))
    # print(input.shape)
    # shape: [64, 12, 207, 2]
    logits = self.model(input, self.adj)

    # loss = loss/11

    # batch_size = len(input[0])
    # nb_nodes = len(input[1])

    lbl_1 = torch.ones(64, 207)
    lbl_2 = torch.zeros(64, 207)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    # print(logits)
    # print(lbl)
    loss = self.b_xent(logits, lbl)
    # print(loss)

    loss.backward()
    optimizer.step()

    return loss.item()

  def embed(self, seq):
    h_1 = self.model.embed(seq)
    return h_1

  def save(self):
    torch.save(self.model.state_dict(), 'best_dgi.pkl')

  def load(self):
    return self.model.load_state_dict(torch.load('best_dgi.pkl'))

  def eval(self, input, real_val):
    self.model.eval()
    input = nn.functional.pad(input,(1,0,0,0))
    output = self.model(input)
    output = output.transpose(1,3)
    #output = [batch_size,12,num_nodes,1]
    real = torch.unsqueeze(real_val,dim=1)
    predict = self.scaler.inverse_transform(output)
    loss = self.loss(predict, real, 0.0)
    mape = util.masked_mape(predict,real,0.0).item()
    rmse = util.masked_rmse(predict,real,0.0).item()
    return loss.item(),mape,rmse





