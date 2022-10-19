import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from models.DCRNN.model.dcrnn_cell import DCGRUCell
from models.DCRNN.base import BaseModel

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

class dcrnn(nn.Module):
	def __init__(self, input_dim, adj_mat, max_diffusion_step, num_nodes,
                 hid_dim, output_dim, num_rnn_layers, filter_type):
		super(dcrnn, self).__init__()
		self.hid_dim = hid_dim
		self._num_nodes = num_nodes  # 207
		self._output_dim = output_dim  # should be 1
		self._num_rnn_layers = num_rnn_layers

		cell = DCGRUCell(input_dim=hid_dim, num_units=hid_dim,
							adj_mat=adj_mat, max_diffusion_step=max_diffusion_step,
							num_nodes=num_nodes, filter_type=filter_type)
		cell_with_projection = DCGRUCell(input_dim=hid_dim, num_units=hid_dim,
											adj_mat=adj_mat, max_diffusion_step=max_diffusion_step,
											num_nodes=num_nodes, num_proj=output_dim, filter_type=filter_type)

		decoding_cells = list()
		# first layer of the decoder
		decoding_cells.append(DCGRUCell(input_dim=input_dim, num_units=hid_dim,
										adj_mat=adj_mat, max_diffusion_step=max_diffusion_step,
										num_nodes=num_nodes, filter_type=filter_type))
		# construct multi-layer rnn
		for _ in range(1, num_rnn_layers - 1):
			decoding_cells.append(cell)
		decoding_cells.append(cell_with_projection)
		self.decoding_cells = nn.ModuleList(decoding_cells)

	def forward(self, inputs, initial_hidden_state, teacher_forcing_ratio=0.5):
		"""
		:param inputs: shape should be (seq_length+1, batch_size, num_nodes, input_dim)
		:param initial_hidden_state: the last hidden state of the encoder. (num_layers, batch, outdim)
		:param teacher_forcing_ratio:
		:return: outputs. (seq_length, batch_size, num_nodes*output_dim) (12, 50, 207*1)
		"""
		# inputs shape is (seq_length, batch, num_nodes, input_dim) (12, 50, 207, 1)
		# inputs to cell is (batch, num_nodes * input_dim)
		seq_length = inputs.shape[0]  # should be 13
		batch_size = inputs.shape[1]
		inputs = torch.reshape(inputs, (seq_length, batch_size, -1))  # (12+1, 50, 207*1)

		# tensor to store decoder outputs
		outputs = torch.zeros(seq_length, batch_size, self._num_nodes*self._output_dim)  # (13, 50, 207*1)
		# if rnn has only one layer
		# if self._num_rnn_layers == 1:
		#     # first input to the decoder is the GO Symbol
		#     current_inputs = inputs[0]  # (64, 207*1)
		#     hidden_state = prev_hidden_state[0]
		#     for t in range(1, seq_length):
		#         output, hidden_state = self.decoding_cells[0](current_inputs, hidden_state)
		#         outputs[t] = output  # (64, 207*1)
		#         teacher_force = random.random() < teacher_forcing_ratio
		#         current_inputs = (inputs[t] if teacher_force else output)

		current_input = inputs[0]  # the first input to the rnn is GO Symbol
		for t in range(1, seq_length):
			# hidden_state = initial_hidden_state[i_layer]  # i_layer=0, 1, ...
			next_input_hidden_state = []
			for i_layer in range(0, self._num_rnn_layers):
				hidden_state = initial_hidden_state[i_layer]
				output, hidden_state = self.decoding_cells[i_layer](current_input, hidden_state)
				current_input = output  # the input of present layer is the output of last layer
				next_input_hidden_state.append(hidden_state)  # store each layer's hidden state
			initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)
			outputs[t] = output  # store the last layer's output to outputs tensor
			# perform scheduled sampling teacher forcing
			teacher_force = random.random() < teacher_forcing_ratio  # a bool value
			current_input = (inputs[t] if teacher_force else output)

		return outputs[1:, :, :]

		def init_hidden(self, batch_size):
			init_states = []  # this is a list of tuples
			for i in range(self._num_rnn_layers):
				init_states.append(self.encoding_cells[i].init_hidden(batch_size))
			# init_states shape (num_layers, batch_size, num_nodes*num_units)
			return torch.stack(init_states, dim=0)