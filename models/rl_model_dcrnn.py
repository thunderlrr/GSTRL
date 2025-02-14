import torch
from torch import nn
import numpy as np
from functools import partial
import random
from afno.afno1d import AFNO1D
from torch.distributions import Categorical
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class DCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self,
                 input_dim,
                 num_units,
                 max_diffusion_step,
                 num_nodes,
                 num_proj=None,
                 activation=torch.tanh,
                 use_gc_for_ru=True,
                 filter_type='laplacian',
                 device=None):
        """
        :param num_units: the hidden dim of rnn
        :param adj_mat: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param max_diffusion_step: the max diffusion step
        :param num_nodes:
        :param num_proj: num of output dim, defaults to 1 (speed)
        :param activation: if None, don't do activation for cell state
        :param use_gc_for_ru: decide whether to use graph convolution inside rnn
        """
        super(DCGRUCell, self).__init__()
        self._activation = activation
        self.num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._use_gc_for_ru = use_gc_for_ru
        self.device = device

        if filter_type == "laplacian":
            supports_len = 1
        elif filter_type == "random_walk":
            supports_len = 1
        elif filter_type == "dual_random_walk":
            supports_len = 2
        else:
            supports_len = 1

        self.dconv_gate = DiffusionGraphConv(supports_len=supports_len,
                                             input_dim=input_dim,
                                             hid_dim=num_units,
                                             num_nodes=num_nodes,
                                             max_diffusion_step=max_diffusion_step,
                                             output_dim=num_units * 2)
        self.dconv_candidate = DiffusionGraphConv(supports_len=supports_len,
                                                  input_dim=input_dim,
                                                  hid_dim=num_units, num_nodes=num_nodes,
                                                  max_diffusion_step=max_diffusion_step,
                                                  output_dim=num_units)
        if num_proj is not None:
            self.project = nn.Linear(self._num_units, self._num_proj)

    @property
    def output_size(self):
        output_size = self.num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self.num_nodes * self._num_proj
        return output_size

    def forward(self, inputs, supports, state):
        """
        :param inputs: (B, num_nodes * input_dim)
        :param state: (B, num_nodes * num_units)
        :return:
        """
        output_size = 2 * self._num_units
        # Start with bias 1.0 to not reset and not update
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.sigmoid(
            fn(inputs, supports, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self.num_nodes, output_size))
        r, u = torch.split(
            value, split_size_or_sections=int(output_size / 2), dim=-1)
        r = torch.reshape(r, (-1, self.num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self.num_nodes * self._num_units))
        # Batch size, self.num_nodes * output_size
        c = self.dconv_candidate(inputs, supports, r * state, self._num_units)
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c
        if self._num_proj is not None:
            # Apply linear projection to state
            batch_size = inputs.shape[0]
            output = torch.reshape(new_state, shape=(-1, self._num_units))
            output = torch.reshape(self.project(output), shape=(
                batch_size, self.output_size))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    @staticmethod
    def _build_sparse_matrix(L):
        """
        Build PyTorch sparse tensor from scipy sparse matrix
        Reference: https://stackoverflow.com/questions/50665141
        :return:
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        pass

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        pass

    def init_hidden(self, batch_size):
        # State: (B, num_nodes * num_units)
        return torch.zeros(batch_size, self.num_nodes * self._num_units).to(self.device)


class DiffusionGraphConv(nn.Module):
    def __init__(self,
                 supports_len,
                 input_dim,
                 hid_dim,
                 num_nodes,
                 max_diffusion_step,
                 output_dim,
                 bias_start=0.0):
        super(DiffusionGraphConv, self).__init__()
        # Don't forget to add for x itself.
        self.num_matrices = supports_len * max_diffusion_step + 1
        input_size = input_dim + hid_dim
        self.num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self.weight = nn.Parameter(torch.FloatTensor(
            size=(input_size * self.num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        self.ln = nn.LayerNorm([output_dim])

        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs, supports, state, output_size, bias_start=0.0):
        """
        Diffusion Graph convolution with graph matrix
        :param inputs:
        :param state:
        :param output_size:
        :param bias_start:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1))
        state = torch.reshape(state, (batch_size, self.num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state
        x0 = torch.transpose(x, dim0=0, dim1=1)
        x0 = torch.transpose(x0, dim0=1, dim1=2)
        x0 = torch.reshape(
            x0, shape=[self.num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, dim=0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = torch.sparse.mm(support.to(x0.device), x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = torch.reshape(
            x, shape=[self.num_matrices, self.num_nodes, input_size, batch_size])
        x = torch.transpose(x, dim0=0, dim1=3)
        x = torch.reshape(
            x, shape=[batch_size * self.num_nodes, input_size * self.num_matrices])

        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.biases)
        x = self.ln(x.reshape(batch_size, self.num_nodes, output_size))
        return torch.reshape(x, [batch_size, self.num_nodes * output_size])

class DCRNNEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 max_diffusion_step,
                 hid_dim,
                 num_nodes,
                 num_rnn_layers,
                 filter_type,
                 device):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self._num_rnn_layers = num_rnn_layers
        self.device = device
        encoding_cells = list()
        # The first layer has different input_dim
        encoding_cells.append(DCGRUCell(input_dim=input_dim,
                                        num_units=hid_dim,
                                        max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes,
                                        filter_type=filter_type,
                                        device=device))

        # Construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(DCGRUCell(input_dim=hid_dim,
                                            num_units=hid_dim,
                                            max_diffusion_step=max_diffusion_step,
                                            num_nodes=num_nodes,
                                            filter_type=filter_type,
                                            device=device))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, supports, initial_hidden_state):
        # Inputs shape is (seq_length, batch, num_nodes, input_dim)
        # Inputs to cell is (batch, num_nodes * input_dim)
        # init_hidden_state should be (num_layers, batch_size, num_nodes*num_units)
        with torch.autograd.detect_anomaly():
            seq_length = inputs.shape[0]
            batch_size = inputs.shape[1]

            inputs = torch.reshape(
                inputs, (seq_length, batch_size, -1))

            current_inputs = inputs
            # The output hidden states, shape (num_layers, batch, outdim)
            output_hidden = []
            for i_layer in range(self._num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output_inner = []
                for t in range(seq_length):
                    _, hidden_state = self.encoding_cells[i_layer](
                        current_inputs[t, ...], supports, hidden_state)
                    output_inner.append(hidden_state)
                output_hidden.append(hidden_state)
                current_inputs = torch.stack(output_inner, dim=0).to(self.device)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self._num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        return torch.stack(init_states, dim=0)

class DCRNNFeatureExtractor(nn.Module):
    def __init__(self, encoder, device):
        super(DCRNNFeatureExtractor, self).__init__()
        self.encoder = encoder
        self.device = device

    def forward(self, source, supports):
        # Assume source data is (batch_size, time_steps, num_nodes, features)
        # Supports is adjacency matrix, assume its shape is (num_nodes, num_nodes)
        # Prepare data, transpose source to (time_steps, batch_size, num_nodes, features)
        B, T, num_nodes, input_size = source.shape
        source = source.transpose(0, 1)

        # Initialize encoder's hidden state
        batch_size = source.shape[1]
        init_hidden_state = self.encoder.init_hidden(batch_size).to(self.device)

        # Get encoder output context (features)
        context, context_all_t = self.encoder(source, supports, init_hidden_state)
        output = context_all_t
        output = output.reshape(B, T, num_nodes, -1)
        return output

def create_grid_adjacency_matrix(rows, cols, device):
    """
    Create a grid-based adjacency matrix
    - rows: number of rows in the grid
    - cols: number of columns in the grid
    """
    num_nodes = rows * cols
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Define the index of each node in the grid
    def get_index(i, j):
        return i * cols + j

    for i in range(rows):
        for j in range(cols):
            idx = get_index(i, j)

            # Check four neighboring nodes and update adjacency matrix
            if i > 0:  # Upper neighbor
                adj_matrix[idx, get_index(i - 1, j)] = 1
            if i < rows - 1:  # Lower neighbor
                adj_matrix[idx, get_index(i + 1, j)] = 1
            if j > 0:  # Left neighbor
                adj_matrix[idx, get_index(i, j - 1)] = 1
            if j < cols - 1:  # Right neighbor
                adj_matrix[idx, get_index(i, j + 1)] = 1

    adj_tensor = torch.tensor(adj_matrix).to(device)
    return adj_tensor

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, agent_num, init_pos, grid_size, energy, capacity, device):
        super(Model, self).__init__()
        self.device = device
        self.encoder = DCRNNEncoder(
            input_dim=input_size,
            max_diffusion_step=3,
            hid_dim=hidden_size*2,
            num_nodes=grid_size[0] * grid_size[1],
            num_rnn_layers=1,
            filter_type='laplacian',
            device=device
        )
        self.dcrnn = DCRNNFeatureExtractor(self.encoder, device=self.encoder.device)

        # Create adjacency matrix support, list = [Adj Mat1, Adj Mat2]
        self.supports = [create_grid_adjacency_matrix(grid_size[0], grid_size[1], device=self.device)]

        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.score = nn.Linear(grid_size[0] * grid_size[1], output_size)
        self.token = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.agent_num = agent_num
        self.init_pos = init_pos
        self.energy = energy
        self.epsilon = 0.9
        self.energy_constraint = 0.3
        self.capacity = capacity

    def get_energy_cost(self, agent_pos, candidate_pos):
        agent_pos_expanded = agent_pos.unsqueeze(0).expand_as(candidate_pos)
        differences = (agent_pos_expanded - candidate_pos) ** 2
        squared_distances = differences.sum(dim=1)
        energy_cost = torch.sqrt(squared_distances)
        return energy_cost

    def forward(self, heat_maps, grid_pos, train=True):
        B, T, H, W = heat_maps.shape

        # Ensure grid_pos is on the same device
        grid_pos_2d = torch.tensor(grid_pos, device=self.device).reshape(H * W, -1)
        grid_pos_1d = grid_pos_2d[..., 0] * W + grid_pos_2d[..., 1]

        # Ensure agent_position is on the same device
        agent_position = torch.tensor(self.init_pos, device=self.device)
        agent_positions_2d = agent_position.expand(B, self.agent_num, -1)
        agent_positions_1d = agent_positions_2d[..., 0] * W + agent_positions_2d[..., 1]

        # Ensure energy and heat_maps_vec are on the same device
        agent_energy = torch.tensor([self.energy], dtype=torch.float32, device=self.device)
        agent_energy = agent_energy.repeat(B, self.agent_num)

        heat_maps_vec = torch.tensor(heat_maps, dtype=torch.float32, device=self.device).unsqueeze(-1).clone()
        min_val = torch.min(heat_maps_vec)
        max_val = torch.max(heat_maps_vec)
        heat_maps_vec = (heat_maps_vec - min_val) / (max_val - min_val)

        # Use dcrnn to extract features, ensure all devices are consistent
        x = heat_maps_vec.reshape(B, T, H * W, -1).clone()
        dcrnn_out = self.dcrnn(x, self.supports)

        prediction_1d = []
        prediction_2d = []
        state_values = []
        log_probs = []

        for t in range(T):
            t_predicted_1d = []
            t_predicted_2d = []
            t_state_values = []
            t_log_probs = []

            curr_heat_maps = heat_maps_vec[:, t, :]

            grid_agent_count = torch.zeros(H * W, dtype=torch.int32).to(self.device)

            for b in range(B):
                batch_predicted = []
                batch_predicted_2d = []
                batch_state_values = []
                batch_log_probs = []

                for i in range(self.agent_num):
                    temp_agent_pos_2d = agent_positions_2d[b, i]
                    energy_cost = self.get_energy_cost(temp_agent_pos_2d, grid_pos_2d)
                    energy_mask = energy_cost > agent_energy[b, i]

                    combime_vec = dcrnn_out[b, t, :, :]

                    grid_demand = curr_heat_maps[b].reshape(H * W).clone()

                    agent_predicted = self.fc(combime_vec)

                    max_agents_per_grid = torch.ceil(grid_demand / self.capacity).long()
                    grid_exceed_mask = grid_agent_count >= max_agents_per_grid

                    agent_predicted[energy_mask] = -np.inf
                    agent_predicted[grid_exceed_mask] = -np.inf

                    energy_mask_2 = energy_cost > 15
                    agent_predicted[energy_mask_2] = -np.inf

                    if torch.all(agent_predicted == -np.inf):
                        agent_predicted[agent_predicted == -np.inf] = -1e9

                    log_p = torch.log_softmax(agent_predicted.squeeze(-1), dim=0)
                    probs = log_p.exp()
                    scores = self.score(probs.squeeze(-1))

                    if train:
                        if random.random() < self.epsilon:
                            _, idx = probs.max(0)
                        else:
                            idx = Categorical(probs).sample()
                    else:
                        _, idx = probs.max(0)

                    grid_index = grid_pos_1d[idx]
                    grid_agent_count[grid_index] += 1

                    current_demand = grid_demand[grid_index]
                    grid_demand[grid_index] = max(0, current_demand - self.capacity)

                    predicted_position = grid_pos_1d[idx]
                    predicted_position_2d = grid_pos_2d[idx]

                    agent_energy[b, i] -= energy_cost[idx].item()
                    batch_predicted.append(predicted_position)
                    batch_predicted_2d.append(predicted_position_2d.squeeze(0))
                    batch_log_probs.append(log_p.squeeze(-1))
                    batch_state_values.append(scores.squeeze(-1))

                t_predicted_1d.append(torch.stack(batch_predicted, dim=0))
                t_predicted_2d.append(torch.stack(batch_predicted_2d, dim=0))
                t_log_probs.append(torch.stack(batch_log_probs, dim=0))
                t_state_values.append(torch.stack(batch_state_values, dim=0))

            agent_positions_1d = torch.stack(t_predicted_1d, dim=0).squeeze(-1)
            agent_positions_2d = torch.stack(t_predicted_2d, dim=0)
            prediction_1d.append(agent_positions_1d)
            prediction_2d.append(agent_positions_2d)
            state_values.append(torch.stack(t_state_values, dim=0))
            log_probs.append(torch.stack(t_log_probs, dim=0))

        prediction_1d = torch.stack(prediction_1d, dim=0).reshape(B, self.agent_num, T)
        prediction_2d = torch.stack(prediction_2d, dim=0).reshape(B, self.agent_num, T, -1)
        log_probs = torch.stack(log_probs, dim=0).reshape(B, self.agent_num, T, -1)
        state_values = torch.stack(state_values, dim=0)
        next_state_values = state_values[1:, :].clone()
        next_state_values = torch.cat([next_state_values, torch.zeros(1, B, self.agent_num).to(self.device)])
        return prediction_1d, prediction_2d, state_values, next_state_values, log_probs