import torch
from torch import nn
import numpy as np
from functools import partial
import random
from afno.afno1d import AFNO1D
from torch.distributions import Categorical 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # Define layers for feature transformation
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sparsity_threshold=0.01,
        use_fno=False,
        use_blocks=False,
    ):
        super().__init__()
        
        # Normalization and filtering layers
        self.norm1 = norm_layer(dim)
        self.filter = AFNO1D(
            hidden_size=dim,
            num_blocks=1,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=1,
            hidden_size_factor=1,
        )

        # Dropout and MLP layers
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.double_skip = True

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x

class FftNet(nn.Module):
    def __init__(
        self,
        pos_num,
        embed_dim=32,
        depth=2,
        mlp_ratio=4.0,
        representation_size=None,
        uniform_drop=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        dropcls=0,
        sparsity_threshold=0.01,
        use_fno=False,
        use_blocks=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Position embedding and dropout
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_num, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Configure dropout path rates
        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    sparsity_threshold=sparsity_threshold,
                    use_fno=use_fno,
                    use_blocks=use_blocks,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, agent_num, init_pos, grid_size, energy, capacity, device):
        super(Model, self).__init__()
        
        # Network layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.spatial_emb = nn.Linear(1, hidden_size)
        self.spatial_layer = FftNet(grid_size[0] * grid_size[1], hidden_size, sparsity_threshold=0.01, depth=2)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.score = nn.Linear(grid_size[0] * grid_size[1], output_size)
        self.token = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.agent_encoder_fc = nn.Linear(4, hidden_size*2)
        self.agent_norm = nn.LayerNorm(hidden_size*2)
        self.gate_weight = nn.Parameter(torch.tensor(0.5))

        # Model parameters
        self.agent_num = agent_num
        self.init_pos = init_pos
        self.energy = energy
        self.device = device
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
        
        # Initialize positions and grid
        grid_pos_2d = torch.tensor(grid_pos).reshape(H * W, -1).to(self.device)
        grid_pos_1d = grid_pos_2d[..., 0] * W + grid_pos_2d[..., 1]

        agent_position = torch.tensor(self.init_pos).to(self.device)
        agent_positions_2d = agent_position.expand(B, self.agent_num, -1)
        agent_positions_1d = agent_positions_2d[..., 0] * W + agent_positions_2d[..., 1]
        
        # Initialize agent energy
        agent_energy = torch.tensor([self.energy], dtype=torch.float32).to(self.device)
        agent_energy = agent_energy.repeat(B, self.agent_num)

        # Process heat maps
        heat_maps_vec = torch.tensor(heat_maps, dtype=torch.float32).unsqueeze(-1).to(self.device).clone()
        min_val = torch.min(heat_maps_vec)
        max_val = torch.max(heat_maps_vec)
        heat_maps_vec = (heat_maps_vec - min_val) / (max_val - min_val + 1e-5)

        # LSTM processing
        x = self.input_layer(heat_maps_vec).reshape(B * H * W, T, -1).clone()
        lstm_out, (hidden, cell) = self.lstm(x)
        temporal = lstm_out[:, -1, :].reshape(B, H * W, -1)

        # Initialize prediction containers
        prediction_1d = []
        prediction_2d = []
        state_values = []
        log_probs = []

        # Main prediction loop
        for t in range(T):
            t_predicted_1d = []
            t_predicted_2d = []
            t_state_values = []
            t_log_probs = []

            curr_heat_maps = heat_maps_vec[:, t, :]
            curr_heat_map_vec = self.spatial_emb(curr_heat_maps.reshape(B, H * W, -1))
            curr_heat_map_vec = self.spatial_layer(curr_heat_map_vec)

            grid_agent_count = torch.zeros(H * W, dtype=torch.int32).to(self.device)

            for b in range(B):
                batch_predicted = []
                batch_predicted_2d = []
                batch_state_values = []
                batch_log_probs = []
                spatial_vec = curr_heat_map_vec[b]
                temporal_vec = temporal[b]

                for i in range(self.agent_num):
                    temp_agent_pos_2d = agent_positions_2d[b, i]

                    # Process agent state
                    agent_state = torch.tensor([
                        agent_positions_2d[b, i][0].item(),
                        agent_positions_2d[b, i][1].item(),
                        agent_energy[b, i].item(),
                        self.capacity
                    ], dtype=torch.float32).to(self.device)

                    agent_state_fc = self.agent_encoder_fc(agent_state)
                    agent_state_fc = self.agent_norm(agent_state_fc)
                    agent_state_fc = agent_state_fc.unsqueeze(0)
                    agent_state_fc = agent_state_fc.expand(H*W, -1)

                    # Calculate masks and predictions
                    energy_cost = self.get_energy_cost(temp_agent_pos_2d, grid_pos_2d)
                    energy_mask = energy_cost > agent_energy[b, i]

                    combime_vec = torch.cat([spatial_vec, temporal_vec], dim=-1)
                    combime_vec_plus = combime_vec * (1 - self.gate_weight) + agent_state_fc * self.gate_weight

                    grid_demand = curr_heat_maps[b].reshape(H * W).clone()
                    limited_demand = torch.min(grid_demand, torch.tensor(self.capacity, device=self.device))

                    agent_predicted = self.fc(combime_vec_plus)

                    # Apply masks
                    max_agents_per_grid = torch.ceil(grid_demand / self.capacity).long()
                    grid_exceed_mask = grid_agent_count >= max_agents_per_grid

                    agent_predicted[energy_mask] = -np.inf
                    agent_predicted[grid_exceed_mask] = -np.inf
                    
                    energy_mask_2 = energy_cost > 15
                    agent_predicted[energy_mask_2] = -np.inf

                    if torch.all(agent_predicted == -np.inf):
                        agent_predicted[agent_predicted == -np.inf] = -1e9

                    # Calculate probabilities
                    log_p = torch.log_softmax(agent_predicted.squeeze(-1), dim=0)
                    probs = log_p.exp()

                    # Select action
                    if train:
                        if random.random() < self.epsilon:
                            _, idx = probs.max(0)
                        else:
                            idx = Categorical(probs).sample()
                    else:
                        _, idx = probs.max(0)

                    # Update grid and agent states
                    grid_index = grid_pos_1d[idx]
                    grid_agent_count[grid_index] += 1

                    current_demand = grid_demand[grid_index]
                    grid_demand[grid_index] = max(0, current_demand - self.capacity)

                    predicted_position = grid_pos_1d[idx]
                    predicted_position_2d = grid_pos_2d[idx]

                    agent_energy[b, i] -= energy_cost[idx].item()

                    print(f"Time {t}, Agent {i} (Batch {b}): Predicted Position (2D): {predicted_position_2d.squeeze(0).cpu().numpy()}")

                    # Store predictions
                    batch_predicted.append(predicted_position)
                    batch_predicted_2d.append(predicted_position_2d.squeeze(0))
                    batch_log_probs.append(log_p.squeeze(-1))
                    
                    scores = self.score(probs.squeeze(-1))
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

        # Format final outputs
        prediction_1d = torch.stack(prediction_1d, dim=0).reshape(B, self.agent_num, T)
        prediction_2d = torch.stack(prediction_2d, dim=0).reshape(B, self.agent_num, T, -1)
        log_probs = torch.stack(log_probs, dim=0).reshape(B, self.agent_num, T, -1)
        state_values = torch.stack(state_values, dim=0)
        next_state_values = state_values[1:, :].clone()
        next_state_values = torch.cat([next_state_values, torch.zeros(1, B, self.agent_num).to(self.device)])
        
        return prediction_1d, prediction_2d, state_values, next_state_values, log_probs