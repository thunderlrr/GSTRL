import torch 
import torch.nn.functional as F
import numpy as np
import pandas as pd

def check_device(device=None):
    if device is None:
        print(
            "`device` is missing, try to train and evaluate the model on default device."
        )
        if device == "cpu":
            return torch.device("cpu")
     
    else:
        print(torch.device)
        if isinstance(device, torch.device):
            return device
        else:
            return torch.device(device)
        
def get_rewards(predicted_positions, heat_map, train=True):
    # Get shape parameters
    B, T, H, W = heat_map.shape
    new_heat_map = heat_map.reshape(B, T, -1)
    
    predicted_positions = predicted_positions.reshape(B, T, -1)
    rewards = torch.gather(new_heat_map, 2, predicted_positions)

    epsilon = 1e-8
    if train:
        # Normalize rewards during training for stability
        rewards = (rewards-rewards.mean()) / (rewards.std()+epsilon)
        return rewards
    else:
        return rewards

def get_rewards_2(predicted_positions, heat_map, capacity, train=True):
    # Get shape parameters
    B, T, H, W = heat_map.shape
    new_heat_map = heat_map.reshape(B, T, -1)
    predicted_positions = predicted_positions.reshape(B, T, -1)
    coverage = torch.zeros_like(predicted_positions, dtype=torch.float32)

    # Calculate coverage for each agent
    for b in range(B):
        for t in range(T):
            grid_demand = new_heat_map[b, t, :].clone()
            for agent_idx in range(predicted_positions.shape[2]):
                pos = predicted_positions[b, t, agent_idx]
                demand_here = grid_demand[pos]
                
                if demand_here > 0:
                    cover = min(demand_here, capacity)
                    coverage[b, t, agent_idx] = cover
                    grid_demand[pos] -= cover

    if train:
        # Normalize coverage during training
        epsilon = 1e-8
        coverage_sum = coverage.sum(dim=-1).sum(dim=-1).sum(dim=0)
        coverage_sum = (coverage_sum - coverage_sum.mean()) / (coverage_sum.std() + epsilon)
        return coverage_sum
    else:
        return coverage

def get_values(predicted_positions, heat_map, state_values, device, train=True):
    # Parameters
    epsilon = 1e-8
    gamma = 0.99
    trace_decay = 0.99

    # Get shape parameters
    B, agent_num, T = predicted_positions.shape
    B, T, H, W = heat_map.shape

    predicted_positions = predicted_positions.reshape(B, agent_num, T)
    state_values = state_values.reshape(B, agent_num, T)
    new_heat_map = heat_map.reshape(B, T, H * W)
    
    predicted_positions = predicted_positions.permute(0, 2, 1).reshape(B, T, agent_num)
    r_steps = torch.gather(new_heat_map, 2, predicted_positions).permute(0, 2, 1)
    
    # Calculate returns and advantages
    rewards = []
    advantages_list = []
    for b in range(B):
        for i in range(agent_num):
            R = 0
            advantage = 0
            next_value = 0
            r_samples = r_steps[b][i][:].cpu().tolist()
            v_samples = state_values[b][i][:].cpu().tolist()
            rewards_sample = []
            advantage_sample = []
            
            for r, v in zip(r_samples[::-1], v_samples[::-1]):
                R = r + gamma * R
                v = v if r != 0 else 0
                rewards_sample.insert(0, R)
                td_error = r + next_value * gamma - v
                advantage = td_error + advantage * gamma * trace_decay
                next_value = v
                advantage_sample.insert(0, advantage)
            advantages_list.extend(advantage_sample)
            rewards.extend(rewards_sample)
    
    # Normalize returns and advantages
    returns = torch.tensor(rewards).to(device)
    returns = (returns - returns.mean()) / (returns.std() + epsilon)

    advantages = torch.tensor(advantages_list).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + epsilon)
    
    return returns, advantages

def cal_loss_2(state_values, action_probs, rewards, device, advantages=None):
    agent_num, B, T = state_values.shape
    state_values = state_values.reshape(-1, 1)
    actions = torch.argmax(action_probs, dim=-1).cpu().numpy()
    actions = torch.LongTensor(actions).to(device).to(torch.int64)
    log_probs = action_probs.gather(dim=-1, index=actions.unsqueeze(-1)).reshape(-1, 1)
    
    value_losses = F.smooth_l1_loss(state_values, rewards.detach()).sum()
    policy_losses = (-log_probs.squeeze(1) * advantages.detach()).mean()
       
    return value_losses, policy_losses

def compute_region_1d_indices(grid_size, region_size):
    regions_per_dim = grid_size // region_size
    region_indices = {}

    for i in range(regions_per_dim):
        for j in range(regions_per_dim):
            region_index = i * regions_per_dim + j
            indices = []
            for x in range(region_size):
                for y in range(region_size):
                    index = (i * region_size + x) * grid_size + (j * region_size + y)
                    indices.append(index)
            region_indices[region_index] = indices

    return region_indices

def compute_region_coordinates(grid_size, region_size):
    regions_per_dim = grid_size // region_size
    region_coordinates = {}

    for i in range(regions_per_dim):
        for j in range(regions_per_dim):
            region_index = i * regions_per_dim + j
            coordinates = []
            for x in range(region_size):
                for y in range(region_size):
                    grid_x = i * region_size + x
                    grid_y = j * region_size + y
                    coordinates.append((grid_x, grid_y))
            region_coordinates[region_index] = coordinates

    return region_coordinates

def region_centers(grid_size, region_size):
    regions_per_dim = grid_size // region_size
    centers = []

    for i in range(regions_per_dim):
        for j in range(regions_per_dim):
            center_x = i * region_size + region_size // 2
            center_y = j * region_size + region_size // 2
            centers.append((center_x, center_y))

    return centers

def cal_loss(state_values, next_state_values, action_probs, rewards, device):
    # Get shape parameters
    T, B, agent_num = state_values.shape
    gamma = 0.99

    state_values = state_values.reshape(B, agent_num, T)
    next_state_values = next_state_values.reshape(B, agent_num, T)
    
    # Calculate target values and critic loss
    target_values = rewards + gamma * next_state_values
    critic_loss = F.mse_loss(state_values, target_values)

    # Calculate actor loss using advantages
    actions = torch.argmax(action_probs, dim=-1).cpu().numpy()
    actions = torch.LongTensor(actions).to(device)
    log_probs = action_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    
    advantages = (target_values - state_values).detach()
    actor_loss = (-log_probs.squeeze(1) * advantages).mean()

    return actor_loss, critic_loss