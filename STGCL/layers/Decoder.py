import torch
import torch.nn as nn
import torch.nn.functional as F

class gwnet(nn.Module):
  def __init__(self,out_dim=12,skip_channels=256,end_channels=512):
        super(gwnet, self).__init__()

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

  def forward(self, x):
      #decoder
      x = F.relu(self.end_conv_1(x))
      x = self.end_conv_2(x)
      return x
