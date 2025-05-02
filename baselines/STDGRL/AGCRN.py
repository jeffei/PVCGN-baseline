import torch
import torch.nn as nn

from baselines.STDGRL.AGCRNCell import AGCRNCell
from baselines.STDGRL.transformer import PositionalEmbedding, SelfAttentionLayer

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)
        
        # Temproal transformer path
        model_dim = self.hidden_dim
        self.mlp = nn.Linear(args.input_dim, model_dim)
        self.temporal_trans = SelfAttentionLayer(model_dim=model_dim, feed_forward_dim=1024, num_heads=4, dropout=0.3)
        self.pos_embedding = PositionalEmbedding(d_model=model_dim)

        # Fusion module, 每个节点对应一个独立权重
        self.ws = nn.Parameter(torch.randn(self.num_node, self.hidden_dim), requires_grad=True)
        self.wt = nn.Parameter(torch.randn(self.num_node, self.hidden_dim), requires_grad=True)


        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets=None, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        source = source[..., :self.input_dim]

        init_state = self.encoder.init_hidden(source.shape[0])
        output_1, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output_1 = output_1[:, -1:, :, :]                                   #B, 1, N, hidden

        # 时域transformer
        value_embedding = self.mlp(source)  # [B, T, N, D] ==> [B, T, N, model_dim]
        pos_embedding = self.pos_embedding(source).unsqueeze(-2)
        x = value_embedding + pos_embedding

        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B*N, T, D)
        temporal_output = self.temporal_trans(x, attn_dim=-2)
        output_2 = temporal_output.reshape(B, N, T, D)
        output_2 = output_2.permute(0, 2, 1, 3)
        output_2 = output_2[:, -1:, :, :]                       # [B, 1, N, hidden]

        # Fusion
        output = output_1 * self.ws + output_2 * self.wt


        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output