#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


import torch
import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import util
# from layers import GCN, AvgReadout, Discriminator
# from engine import trainer


# In[6]:


# file path
data_path = "data/METR-LA"
adj_data = "data/sensor_graph/adj_mx.pkl"

# training params
adj_type = 'symnadj'
batch_size = 64
nb_epochs = 120
lr = 1e-4
# encoder
hid_units = 64
em_size = 128
ft_size = 2


# In[4]:


# data loading
sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(adj_data, adj_type)
dataloader = util.load_dataset(data_path, batch_size, batch_size, batch_size)

# adj_mx = torch.Tensor(adj_mx)

#https://stackoverflow.com/questions/67814465/convert-list-of-tensors-into-tensor-pytorch
adj_mx = torch.stack([torch.tensor(i) for i in adj_mx], dim=0)
adj_mx = torch.squeeze(adj_mx)


# In[5]:


from engine import trainer
engine = trainer(ft_size, hid_units, adj_mx, lr)


# In[6]:


from layers import GCN
print("start training...",flush=True)
best = 1e9
best_t = 0

for epoch in range(5):
    train_loss = torch.tensor([])
    for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        # shape: [64, 12, 207, 2]
        trainx = torch.Tensor(x)
#         trainx = trainx.transpose(1,3)
        # trainy = torch.Tensor(y)
        loss = engine.train(trainx)
        train_loss = torch.cat((train_loss, torch.tensor([loss])), 0)
#         print(train_loss)
#         log = 'Iter: {:03d}, Train Loss: {:.4f}'
#         print(log.format(iter, train_loss[-1]),flush=True)
    
    log = 'Epoch: {:03d}, Train Loss: {:.4f}'
#     mtrain_loss = np.mean(train_loss)
    m_loss = np.mean(train_loss.tolist())
    if m_loss < best:
        best = loss
        engine.save()
        best_t = epoch
    print(log.format(epoch, m_loss),flush=True)


# In[ ]:





# In[1]:


import torch
import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import util


# In[2]:


# file path
data_path = "data/METR-LA"
adj_data = "data/sensor_graph/adj_mx.pkl"

# training params
adj_type = 'symnadj'
batch_size = 64
nb_epochs = 120
lr = 1e-4
# encoder
hid_units = 64
em_size = 128
ft_size = 2


# In[3]:


from models import STDGI 
# print('Loading {}th epoch'.format(best_t))
model = STDGI(ft_size, hid_units)
model.load_state_dict(torch.load('best_dgi.pkl'))


# In[4]:


sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(adj_data, adj_type)
dataloader = util.load_dataset(data_path, batch_size, batch_size, batch_size)

# adj_mx = torch.Tensor(adj_mx)

#https://stackoverflow.com/questions/67814465/convert-list-of-tensors-into-tensor-pytorch
adj_mx = torch.stack([torch.tensor(i) for i in adj_mx], dim=0)
adj_mx = torch.squeeze(adj_mx)


# In[5]:


#regressor
from models.regressor import *

INPUT_DIM = 64
OUTPUT_DIM = 2
EMB_DIM = 128
HID_DIM = 64
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
output_length = 12
seq_length = 12
n_features = 2

regressor = Seq2Seq(seq_length, n_features, EMB_DIM,OUTPUT_DIM,INPUT_DIM)


# In[6]:


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
regressor.apply(init_weights)


# In[ ]:


optimizer = torch.optim.Adam(regressor.parameters(), lr=4e-3,weight_decay=1e-5)

for epoch in range(5):
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        regressor = regressor.train()
        
        testx = torch.Tensor(x)
        #trainx = trainx.transpose(1,3)
        testy = torch.Tensor(y)
        testx = testx.transpose(0,1)
        testy = testy.transpose(0,1)

    #     embed = model.embed(testx, adj_mx)
        src = torch.stack([model.embed(feature, adj_mx) for feature in testx], dim=0)
        trg = torch.stack([model.embed(feature, adj_mx) for feature in testy], dim=0)

        src = src.transpose(2, 0)
        trg = trg.transpose(2, 0)
        #(node, batch, sequence, features)
        print(src.shape)

        
        #for each node
        for i in range(src.size()[0]):
            optimizer.zero_grad()
            
            seq_inp = src[i, :, :]
            seq_true = trg[i, :, :]
            
            for t in range(trg[i].size()[1]):
                seq_pred = regressor(seq_inp,testy[0, :, 1, :])
                print(seq_pred)
        


# In[ ]:


src[0, :, :].shape


# In[17]:


testy[0, :, 1, :].shape


# In[31]:


src[0, :, :]


# In[ ]:




