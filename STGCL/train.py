import torch
import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import util
import augmentation

# file path
data_path = "data/PEMS-BAY"
adj_data = "data/sensor_graph/adj_mx_bay.pkl"
adj_type = "doubletransition"
batch_size=64


sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(adj_data, adj_type)
dataloader = util.load_dataset(data_path, batch_size, batch_size, batch_size)

from layers.Encoder import gwnet as gwnetEncoder
from layers.Decoder import gwnet as gwnetDecoder


device = torch.device('cuda:0')
# device = torch.device('cpu')
num_nodes = 325
dropout = 0.3
supports = [torch.Tensor(i).to(device) for i in adj_mx]
gcn_bool=True
addaptadj=True
in_dim = 2
seq_length= 12
nhid = 32
scaler = dataloader['scaler']
lr = 0.001
wdecay = 0.0001
c_rate = 0.1

projection_head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, 256))
encoder = gwnetEncoder(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
decoder = gwnetDecoder(out_dim=seq_length,skip_channels=nhid * 8,end_channels=nhid * 16)

from engine import trainer
engine = trainer(scaler, device, adj_mx, lr, wdecay,encoder, decoder, c_rate=c_rate)
his_loss =[]

for i in range(1, 101):
    train_loss = []
    dataloader['train_loader'].shuffle()
    for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        tempx = torch.Tensor(x)
        trainx = torch.Tensor(x).to(device)
        trainx = trainx.transpose(1, 3)
        #[64, 2, 325, 12]
        trainy = torch.Tensor(y).to(device)
        trainy = trainy.transpose(1, 3)
        #torch.Size([64, 2, 325, 12])

        start_times = []
        for t in range(trainx.shape[0]):
            curr = np.round(tempx[t, 1, 0, 0], 4)
            start_times.append(curr)
        start_times = torch.Tensor(start_times).to(device)
        loss = engine.train(trainx, trainy, start_times)
        train_loss.append(loss)
        if iter % 50 == 0 :
            log = 'Iter: {:03d}, Train Loss: {:.4f}'
            print(log.format(iter, train_loss[-1]), flush=True)
    
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x)
        testx = testx.transpose(1, 3).to(device)
        testy = torch.Tensor(y)
        testy = testy.transpose(1, 3).to(device)
        metrics = engine.eval(testx, testy)
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
    
    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    his_loss.append(mvalid_loss)
        
    print(f'Epoch: {i}, TrainLoss: {np.mean(train_loss)}, ValidLoss: {mvalid_loss}')
    torch.save(engine.model.state_dict(), "_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load("_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds, _, _ = engine.model(testx)
            preds = preds.transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(), "_exp"+str(1)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    
