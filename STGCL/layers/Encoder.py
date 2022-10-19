import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from models.DCRNN.model.dcrnn_cell import DCGRUCell
from models.DCRNN.base import BaseModel

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
  def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



  def forward(self, input):
      in_len = input.size(3)
      if in_len<self.receptive_field:
          x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
      else:
          x = input
      x = self.start_conv(x)
      skip = 0

      # calculate the current adaptive adj matrix once per iteration
      new_supports = None
      if self.gcn_bool and self.addaptadj and self.supports is not None:
          adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
          new_supports = self.supports + [adp]

      # WaveNet layers
      for i in range(self.blocks * self.layers):

          #            |----------------------------------------|     *residual*
          #            |                                        |
          #            |    |-- conv -- tanh --|                |
          # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
          #                 |-- conv -- sigm --|     |
          #                                         1x1
          #                                          |
          # ---------------------------------------> + ------------->	*skip*

          #(dilation, init_dilation) = self.dilations[i]

          #residual = dilation_func(x, dilation, init_dilation, i)
          residual = x
          # dilated convolution
          filter = self.filter_convs[i](residual)
          filter = torch.tanh(filter)
          gate = self.gate_convs[i](residual)
          gate = torch.sigmoid(gate)
          x = filter * gate

          # parametrized skip connection

          s = x
          s = self.skip_convs[i](s)
          try:
              skip = skip[:, :, :,  -s.size(3):]
          except:
              skip = 0
          skip = s + skip


          if self.gcn_bool and self.supports is not None:
              if self.addaptadj:
                  x = self.gconv[i](x, new_supports)
              else:
                  x = self.gconv[i](x,self.supports)
          else:
              x = self.residual_convs[i](x)

          x = x + residual[:, :, :, -x.size(3):]


          x = self.bn[i](x)

      #decoder
      x = F.relu(skip)
      return x


class dcrnn(nn.Module):
    def __init__(self, input_dim, adj_mat, max_diffusion_step, hid_dim, num_nodes,
                num_rnn_layers, filter_type):
        super(dcrnn, self).__init__()
        self.hid_dim = hid_dim
        self._num_rnn_layers = num_rnn_layers

        # encoding_cells = []
        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(DCGRUCell(input_dim=input_dim, num_units=hid_dim, adj_mat=adj_mat,
                                        max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes, filter_type=filter_type))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(DCGRUCell(input_dim=hid_dim, num_units=hid_dim, adj_mat=adj_mat,
                                            max_diffusion_step=max_diffusion_step,
                                            num_nodes=num_nodes, filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state):
        # inputs shape is (seq_length, batch, num_nodes, input_dim) (12, 64, 207, 2)
        # inputs to cell is (batch, num_nodes * input_dim)
        # init_hidden_state should be (num_layers, batch_size, num_nodes*num_units) (2, 64, 207*64)
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))  # (12, 64, 207*2)

        current_inputs = inputs
        output_hidden = []  # the output hidden states, shape (num_layers, batch, outdim)
        for i_layer in range(self._num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](current_inputs[t, ...], hidden_state)  # (50, 207*64)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).cuda()  # seq_len, B, ...
            # current_inputs = torch.stack(output_inner, dim=0)  # seq_len, B, ...

        # output_hidden: the hidden state of each layer at last time step, shape (num_layers, batch, outdim)
        # current_inputs: the hidden state of the top layer (seq_len, B, outdim)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples
        for i in range(self._num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # init_states shape (num_layers, batch_size, num_nodes*num_units)
        return torch.stack(init_states, dim=0)