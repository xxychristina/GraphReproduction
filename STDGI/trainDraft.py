import math
import torch
import util

import torch.nn as nn
from models.regressor import *
import torch.optim as optim

from layers import GCN
from engine import trainer

# file path
data_path = "data/METR-LA"
adj_data = "data/sensor_graph/adj_mx.pkl"

# training params
adj_type = 'symnadj'
batch_size = 64
nb_epochs = 120
lr = math.exp(-3)
# encoder
hid_units = 64
em_size = 128
ft_size = 2


# data loading
sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(adj_data, adj_type)
dataloader = util.load_dataset(data_path, batch_size, batch_size, batch_size)

engine = trainer(ft_size, hid_units, adj_mx, lr)

# training epoches
train_loss = []
trainedEmbed = GCN()
for epoch in range(nb_epochs):
  for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
      # shape: [64, 12, 207, 2]
      trainx = torch.Tensor(x)
      trainx = trainx.transpose(1,3)
      # trainy = torch.Tensor(y)
      loss, embed = engine.train(trainx)
      train_loss.append(loss)
      trainedEmbed = embed

stdgiEmbed = []
for i in range(12):
  stdgiEmbed.append(torch.cat(trainedEmbed(trainx[i]), trainx), 1)


#regressor
INPUT_DIM = 13248
OUTPUT_DIM = 2
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 64
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters())
model.train()
epoch_loss = 0


