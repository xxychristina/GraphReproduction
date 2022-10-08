from models.STGCLRNN import STGCL
import torch
import torch.nn as nn
import torch.optim as optim
import util

class trainer():
  def __init__(self, scaler, device, adj, lr, encoder, decoder, r_f=30, c_rate=0.1, num_nodes=325, output_dim=1):
    self.model = STGCL(device, encoder, decoder, num_nodes, output_dim).to(device)
    self.device = device
    self.adj = adj
    self.scaler = scaler
    self.r_f = r_f
    self.c_rate = c_rate
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
  
  def train(self, input, y, teacher_forcing_ratio, start_times):
    self.model.train()
    output, sum_x, sum_c = self.model(input, y, teacher_forcing_ratio)
    
    output = torch.transpose(output.view(12, 64, self.model.num_nodes,
                                                     self.model.output_dim), 0, 1)  # back to (50, 12, 207, 1)
    realy =  y[..., :1]

    print(output.shape)
    print(y.shape)
    print(output)
    print(y)

    ploss = util.masked_mae(output, realy, 0.0)
    closs = util.c_loss(self.device, sum_x, sum_c, start_times, self.r_f)
    # print("ploss" + str(ploss))
    # print("closs" + str(closs))
    print(ploss)
    loss = (1 - self.c_rate) * ploss + self.c_rate * closs
    loss.backward()
    self.optimizer.step()

    return loss.item()
  
  def eval(self, input, y):
    self.model.eval()
    input = nn.functional.pad(input, (1, 0, 0, 0))
    output, sum_x, sum_c = self.model(input)
    output = output.transpose(1, 3)
    realy = y[:, 0, :, :]
    real = torch.unsqueeze(realy, dim=1)
    predict = self.scaler.inverse_transform(output)
    loss = util.masked_mae(predict, real, 0.0)
    mape = util.masked_mape(predict,real,0.0).item()
    rmse = util.masked_rmse(predict,real,0.0).item()
    return loss.item(), mape, rmse

