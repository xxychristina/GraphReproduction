import torch
import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import util_dcrnn
import metric_dcrnn
import augmentation

from layers.Encoder import dcrnn as dcrnnEncoder
from layers.Decoder import dcrnn as dcrnnDecoder

# file path
data_path = "data/METR-LA"
adj_data = "data/sensor_graph/adj_mx_unix.pkl"
adj_type = "dual_random_walk"
batch_size=64

sensor_ids, sensor_id_to_ind, adj_mx = util_dcrnn.load_graph_data(adj_data)
dataloader = util_dcrnn.load_dataset(data_path, batch_size, batch_size)

'''
  Parameters setting
'''
device = torch.device('cuda:0')
# device = torch.device('cpu')
batch_size = 64
enc_input_dim = 2
dec_input_dim = 1
max_diffusion_step = 2
num_nodes = 207
num_rnn_layers = 2
rnn_units = 64
seq_len = 12
output_dim = 1
filter_type = "dual_random_walk"
scaler = dataloader['scaler']
lr = 0.0001
len_epoch = 570
cl_decay_steps = 2000
c_rate = 0.1

projection_head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, 256))
encoder = dcrnnEncoder(input_dim=enc_input_dim, adj_mat=adj_mx, max_diffusion_step=max_diffusion_step, hid_dim=rnn_units, num_nodes=num_nodes, num_rnn_layers=num_rnn_layers, filter_type=filter_type)
decoder = dcrnnDecoder(input_dim=dec_input_dim, adj_mat=adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=num_nodes, hid_dim=rnn_units, output_dim=output_dim, num_rnn_layers=num_rnn_layers, filter_type=filter_type)

from engineRNN import trainer
engine = trainer(scaler, device, adj_mx, lr, encoder, decoder, num_nodes)
his_loss =[]

def _compute_sampling_threshold(global_step, k):
    """
    Computes the sampling probability for scheduled sampling using inverse sigmoid.
    :param global_step:
    :param k:
    :return:
    """
    return k / (k + math.exp(global_step / k))

for i in range(1, 101):
    train_loss = []
    for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        tempx = torch.Tensor(x)
        trainx = torch.Tensor(x).to(device)
        trainy = torch.Tensor(y).to(device)

        '''Timestamp for negative filter'''
        start_times = []
        for t in range(trainx.shape[0]):
            curr = np.round(tempx[t, 1, 0, 0], 4)
            start_times.append(curr)

        start_times = torch.Tensor(start_times).to(device)

        global_step = (i - 1) * len_epoch + iter
        teacher_forcing_ratio = _compute_sampling_threshold(global_step, cl_decay_steps)

        loss = engine.train(trainx, trainy, teacher_forcing_ratio, start_times)
        train_loss.append(loss)
        if iter % 50 == 0 :
            log = 'Iter: {:03d}, Train Loss: {:.4f}'
            print(log.format(iter, train_loss[-1]), flush=True)
    
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):

      global_step = (i - 1) * len_epoch + iter
      teacher_forcing_ratio = _compute_sampling_threshold(global_step, cl_decay_steps)
      
      testx = torch.Tensor(x).to(device)
      testy = torch.Tensor(y).to(device)
      loss = engine.eval(testx, testy, teacher_forcing_ratio)
      valid_loss.append(loss)
      # valid_loss.append(metrics[0])
      # valid_mape.append(metrics[1])
      # valid_rmse.append(metrics[2])
    
    mvalid_loss = np.mean(valid_loss)
    # mvalid_mape = np.mean(valid_mape)
    # mvalid_rmse = np.mean(valid_rmse)
    his_loss.append(mvalid_loss)
        
    print(f'Epoch: {i}, TrainLoss: {np.mean(train_loss)}, ValidLoss: {mvalid_loss}')
    torch.save(engine.model.state_dict(), "_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")

#testing
bestid = np.argmin(his_loss)
engine.model.load_state_dict(torch.load("_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
model = engine.model.to(device)
model.eval()

realy = dataloader['y_test']
realy = scaler.inverse_transform(realy)

y_preds = torch.FloatTensor([])
predictions = []
groundtruth = list()
with torch.no_grad():
  for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    testx = torch.Tensor(x).cuda()
    testy = torch.Tensor(y).cuda()
    outputs, _, _ = model(testx, testy, 0)
    y_preds = torch.cat([y_preds, outputs], dim=1)

y_preds = torch.transpose(y_preds, 0, 1)
y_preds = y_preds.detach().numpy()  # cast to numpy array

print("--------test results--------") 

amae = []
amape = []
armse = []
for horizon_i in range(12):
  y_truth = np.squeeze(realy[:, horizon_i, :, 0])

  y_pred = scaler.inverse_transform(y_preds[:, horizon_i, :])
  predictions.append(y_pred)
  groundtruth.append(y_truth)
  
  mae = metric_dcrnn.masked_mae_np(y_pred[:y_truth.shape[0]], y_truth, null_val=0)
  mape = metric_dcrnn.masked_mape_np(y_pred[:y_truth.shape[0]], y_truth, null_val=0)
  rmse = metric_dcrnn.masked_rmse_np(y_pred[:y_truth.shape[0]], y_truth, null_val=0)
  print(
      "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
          horizon_i + 1, mae, mape, rmse
      )
  )
  amae.append(mae)
  amape.append(mape)
  armse.append(rmse)

log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
torch.save(engine.model.state_dict(), "_dcrnn_exp"+str(1)+"_best_"+str(round(his_loss[bestid],2))+ "_c_rate_" + str(c_rate) + ".pth")

