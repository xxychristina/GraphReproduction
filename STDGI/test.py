import torch
import numpy as np
import util


def main():
  data_path = "data/METR-LA"
  batch_size = 64
  dataloader = util.load_dataset(data_path, batch_size, batch_size, batch_size)
  print(type(dataloader['train_loader']))
  for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
    trainx = torch.Tensor(x)
    
  return


if __name__=="__main__":
  main()