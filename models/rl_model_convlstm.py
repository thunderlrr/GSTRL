import torch
from torch import nn
import numpy as np
import random
from torch.distributions import Categorical

from SimVPv2.simvp.modules import ConvLSTMCell


class ConvLSTM_Model(nn.Module):
    """ConvLSTM Model for precipitation nowcasting."""

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(ConvLSTM_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                             configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true):
        # Permute input tensors to match ConvLSTM input format
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        embeddings = []  # Store hidden states for each timestep

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1])
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.pre_seq_length])

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            embeddings.append(h_t[self.num_layers - 1])  # Use the last layer's hidden state

        embeddings = torch.stack(embeddings, dim=1)  # [batch, T, num_hidden[-1], H, W]

        return embeddings



# Model configuration class
class convlstm_Configs:
    def __init__(self):
        self.in_shape = None  # Input shape (T, C, H, W)
        self.patch_size = 1
        self.pre_seq_length = 13  # Length of input sequence
        self.aft_seq_length = 0
        self.num_layers = 3
        self.filter_size = 5
        self.stride = 1
        self.layer_norm = True
        self.reverse_scheduled_sampling = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Create model and configuration
# configs = convlstm_Configs()
# num_layers = configs.num_layers
# num_hidden = [32, 64, 64]
# model = ConvLSTM_Model(num_layers, num_hidden, configs).to(configs.device)
#
# # Generate random input data (batch_size, seq_length, height, width, channels)
# batch_size = 3
# seq_length = configs.pre_seq_length + configs.aft_seq_length
# height, width, channels = configs.in_shape[2], configs.in_shape[3], configs.in_shape[1]
# frames_tensor = torch.randn(batch_size, seq_length, height, width, channels).to(configs.device)
#
# # Generate mask_true tensor (for mixing true input with generated output)
# mask_true = torch.ones(batch_size, configs.aft_seq_length, height, width, channels).to(configs.device)
#
# # Forward pass through the model
# next_frames, loss = model(frames_tensor, mask_true)
#
# print("Next frames shape:", next_frames.shape)  # Print the shape of the predicted frames
# print("Loss:", loss.item())  # Print the loss value


class Model(nn.Module):  # 多对一版本梯度图错误修改版
    def __init__(self, input_size, hidden_size, output_size, agent_num, init_pos, grid_size, energy, capacity, device):
        super(Model, self).__init__()
        T = 12  # Sequence length
        H = grid_size[0]
        W = grid_size[1]
        in_shape = (T, input_size, H, W)  # (T, num_features, H, W)
        
        self.configs = convlstm_Configs()
        self.configs.in_shape = in_shape
        num_layers = self.configs.num_layers
        num_hidden = [32, 64, 64]
        self.convlstm = ConvLSTM_Model(num_layers, num_hidden, self.configs).to(self.configs.device)
        self.input_size = input_size

        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.score = nn.Linear(grid_size[0] * grid_size[1], output_size)
        self.token = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.agent_num = agent_num
        self.init_pos = init_pos
        self.energy = energy
        self.device = device
        self.epsilon = 0.9
        self.energy_constraint = 0.3
        self.capacity = capacity  # Fixed service capacity per agent at each timestep

    def get_energy_cost(self, agent_pos, candidate_pos):
        agent_pos_expanded = agent_pos.unsqueeze(0).expand_as(candidate_pos)
        differences = (agent_pos_expanded - candidate_pos) ** 2
        squared_distances = differences.sum(dim=1)
        energy_cost = torch.sqrt(squared_distances)
        return energy_cost

    def forward(self, heat_maps, grid_pos, train=True):
        B, T, H, W = heat_maps.shape
        in_shape = (T, 1, H, W)  # Input shape
        grid_pos_2d = torch.tensor(grid_pos).reshape(H * W, -1).to(self.device)
        grid_pos_1d = grid_pos_2d[..., 0] * W + grid_pos_2d[..., 1]

        agent_position = torch.tensor(self.init_pos).to(self.device)
        agent_positions_2d = agent_position.expand(B, self.agent_num, -1)
        agent_positions_1d = agent_positions_2d[..., 0] * W + agent_positions_2d[..., 1]
        agent_energy = torch.tensor([self.energy], dtype=torch.float32).to(self.device)
        agent_energy = agent_energy.repeat(B, self.agent_num)

        heat_maps_vec = torch.tensor(heat_maps, dtype=torch.float32).unsqueeze(-1).to(self.device).clone()
        min_val = torch.min(heat_maps_vec)
        max_val = torch.max(heat_maps_vec)
        heat_maps_vec = (heat_maps_vec - min_val) / (max_val - min_val)

        x = heat_maps_vec.reshape(B, T, H, W, 1).clone()
        mask_true = torch.ones(B, self.configs.aft_seq_length, H, W, self.input_size).to(self.configs.device)
        convlstm_out = self.convlstm(x, mask_true)
        convlstm_out = convlstm_out.reshape(B, T, H * W, -1)

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

                    grid_demand = curr_heat_maps[b].reshape(H * W).clone()
                    combine_vec = convlstm_out[b, t, :, :]
                    agent_predicted = self.fc(combine_vec)

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
