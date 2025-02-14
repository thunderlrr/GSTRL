import numpy as np
import torch
from torch import nn, optim
import argparse
from datetime import datetime
from utils.helper import *
from utils.data_process_bj import *
from models.rl_model_ok import Model
import time
import wandb

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(1)

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="", help="data path")
    parser.add_argument("--start_hour", type=int, default=10, help="start hour")
    parser.add_argument("--end_hour", type=int, default=22, help="end hour") 
    parser.add_argument("--time_interval", type=int, default=60, help="time interval")
    parser.add_argument("--agent_num", type=int, default=10, help="number of agents")
    parser.add_argument("--init_pos", type=list, default=[16,16])
    parser.add_argument("--energy", type=int, default=30)
    parser.add_argument("--grid_size", type=list, default=[32,32])
    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=600, help="training epochs")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--save_iter", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="/data/results")
    parser.add_argument("--model_id", type=int, default=53)
    parser.add_argument("--gpu", type=int, default=2, help="which gpu to run")
    parser.add_argument("--capacity", type=int, default=10, help="service capacity of each agent")

    args = parser.parse_args()
    print(args)
    return args

def train_batch(heat_map, pred_heat_map, neighbor_grid, model, optimizer, device):
    # Move data to device and convert type
    heat_map = heat_map.to(torch.float32).to(device)
    
    # Forward pass
    prediction_1d, prediction_2d, state_values, next_state_values, log_probs = model(heat_map, neighbor_grid, True)
    rewards, advantages = get_values(prediction_1d, heat_map, state_values, device)
    
    # Calculate losses
    value_loss, policy_loss = cal_loss_2(state_values, log_probs, rewards, device, advantages)
    loss = 0.1 * value_loss + policy_loss

    # Backward pass with anomaly detection
    with torch.autograd.detect_anomaly():
        loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

def evaluate(data_loader, model, neighbor_grid, capacity, device, val=True):
    eval_losses = []
    coverages = []
    total_time = 0
    
    with torch.no_grad():
        model.eval()
        for (heat_map, pred_heat_map) in data_loader:
            heat_map = heat_map.to(torch.float32).to(device)
            pred_heat_map = pred_heat_map.to(torch.float32).to(device)

            # Forward pass and timing
            start_time = time.time()
            prediction_1d, prediction_2d, state_values, next_state_values, log_probs = model(pred_heat_map, neighbor_grid, False)
            end_time = time.time()
            total_time += (end_time - start_time)

            # Calculate coverage and losses
            coverage = get_rewards_2(prediction_2d, heat_map, capacity, False)
            rewards, advantages = get_values(prediction_1d, heat_map, state_values, device)
            value_loss, policy_loss = cal_loss_2(state_values, log_probs, rewards, device, advantages)
            loss = 0.1 * value_loss + policy_loss
            
            eval_losses.append(loss.item())
            coverages.append(coverage)

    # Calculate average daily coverage
    coverages = torch.cat(coverages, dim=0).sum()
    total_coverage = coverages.item()
    days = data_loader.dataset.data.shape[0]
    avg_daily_coverage = total_coverage / days

    return np.mean(eval_losses), avg_daily_coverage

def main():
    torch.autograd.set_detect_anomaly(True)

    args = get_config()
    device = check_device(args.gpu)
    capacity = args.capacity

    # Set training period
    start_day = datetime(2013, 12, 10)
    end_day = datetime(2015, 1, 28)
    
    nb_ts_per_day = 1440 // args.time_interval
    train_loader, val_loader, test_loader, neighbor_grid = get_dataloader(args.data_path, start_day, end_day, nb_ts_per_day,
                                                         args.batch_size, args.grid_size)
    
    # Initialize model
    model = Model(args.input_size, args.hidden_size,
                  args.output_size, args.agent_num, args.init_pos, args.grid_size, args.energy, args.capacity, device).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", total_params)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    save_path = args.save_dir + "model_" + str(args.model_id) + ".pth"
    
    iter = 0
    max_coverage = 0
    train_losses = []
    train_coverages = []

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_train_losses = []
        epoch_train_coverages = []
        
        for (heat_map, pred_heat_map) in train_loader:
            heat_map = heat_map.to(torch.float32).to(device)
            pred_heat_map = pred_heat_map.to(torch.float32).to(device)
            
            loss = train_batch(heat_map, pred_heat_map, neighbor_grid, model, optimizer, device)
            epoch_train_losses.append(loss)
            iter += 1
            
            if iter % args.save_iter == 0:
                model.eval()
                val_loss, val_coverage = evaluate(val_loader, model, neighbor_grid, capacity, device)
                epoch_train_coverages.append(val_coverage)
            
                if val_coverage > max_coverage:
                    max_coverage = val_coverage
                    print("Save model, max coverage:", max_coverage)
                    torch.save(model.state_dict(), save_path)
                    
                print(f"Epoch [{epoch}/{args.epochs}] ({iter}) Train loss: {np.mean(epoch_train_losses):.4f}, "
                      f"Val loss: {val_loss:.4f}, Coverage: {val_coverage:.2f}")
                model.train()

        train_losses.append(np.mean(epoch_train_losses))
        train_coverages.append(np.mean(epoch_train_coverages))

    # Final evaluation
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_loss, test_coverage = evaluate(test_loader, model, neighbor_grid, capacity, device)
    print("Final Test Loss:", test_loss)
    print("Final Test Coverage:", test_coverage)
    
    # Save metrics
    np.savetxt('train_losses.txt', train_losses)
    np.savetxt('train_coverages.txt', train_coverages)

if __name__ == "__main__":
    main()
