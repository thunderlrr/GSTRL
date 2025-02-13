import torch
from torch import nn
import numpy as np
import random
from torch.distributions import Categorical

from SimVPv2.simvp.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                           HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                           SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock)

class SimVP_Model(nn.Module):
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))

        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec)

        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(T*hid_S, hid_T, N_T)
        else:
            self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)

        return Y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC( C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        if enc1.shape[-2:] != hid.shape[-2:]:
            enc1 = torch.nn.functional.interpolate(enc1, size=hid.shape[-2:], mode='bilinear', align_corners=False)
        Y = self.dec[-1](hid + enc1)
        Y = Y[:, :, :51, :]
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_in,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)

        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        y = z.reshape(B, T, C, H, W)
        return y


class MetaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type == 'mlp':
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'moga':
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, agent_num, init_pos, grid_size, energy, capacity, device):
        super(Model, self).__init__()
        T = 12
        H = grid_size[0]
        W = grid_size[1]
        in_shape = (T, input_size, H, W)

        self.simvp = SimVP_Model(in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gsta')
        self.score = nn.Linear(grid_size[0] * grid_size[1], output_size)
        self.token = nn.Parameter(torch.randn(1, 1, 1, 1))
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
        in_shape = (T, 1, H, W)
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
        x = heat_maps_vec.reshape(B, T, 1, H, W).clone()
        simvp_out = self.simvp(x)
        simvp_out = simvp_out.reshape(B,T,H*W,-1)

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

                    agent_predicted = simvp_out[b, t, :, :]
                    grid_demand = curr_heat_maps[b].reshape(H * W).clone()

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
