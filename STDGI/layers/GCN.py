
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html
# class GCN(nn.Module):
#     def __init__(self, c_in, c_out):
#         super(GCN, self).__init__()
#         self.projection = nn.Linear(c_in, c_out,bias=False)

#     def forward(self, node_feats, adj_matrix):
#         """
#         Inputs:
#             node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
#             adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
#                          Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
#                          Shape: [batch_size, num_nodes, num_nodes]
#         """
#         # Num neighbours = number of incoming edges
#         num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
#         node_feats = self.projection(node_feats)
#         print(node_feats.size())
#         print(adj_matrix.size())
#         node_feats = torch.matmul(node_feats, adj_matrix)
#         node_feats = node_feats / num_neighbours
#         return node_feats

# # https://github.com/nnzhan/Graph-WaveNet/issues/15
# class nconv(nn.Module):
#     def __init__(self):
#         super(nconv,self).__init__()

#     def forward(self,x, A):
#         x = torch.einsum('ntv,vw->ntw',(x,A))
#         return x.contiguous()

# class linear(nn.Module):
#     def __init__(self,c_in,c_out):
#         super(linear,self).__init__()
#         self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

#     def forward(self,x):
#         return self.mlp(x)

# class GCN(nn.Module):
#     def __init__(self,c_in, c_out):
#         super(GCN,self).__init__()
#         # self.nconv = nconv()
#         # self.mlp = linear(c_in,c_out)
#         # self.dropout = dropout
#         # self.order = order
#         self.nconv = nconv()
#         self.fc = nn.Linear(c_in, c_out, bias=False)
#         # self.act = nn.PReLU()

#     def forward(self,x,support):
#         out = []
#         for a in support:
#             x1 = self.nconv(x,a)
#             out.append(x1)
#             # for k in range(2, self.order + 1):
#             #     x2 = self.nconv(x1,a)
#             #     out.append(x2)
#             #     x1 = x2

#         h = torch.cat(out,dim=1)
#         # h = self.mlp(h)
#         # h = F.dropout(h, self.dropout, training=self.training)
#         return h

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            #squeeze(1*A*B) -> (A*B)
            #spmm = sparse multiplication
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            #bmm = batch matrix-matrix product of matrices
            out = torch.matmul(adj, seq_fts)
            out = self.act(out)
            out = torch.matmul(adj, out)
            out = self.act(out)
        if self.bias is not None:
            out += self.bias
        
        return out