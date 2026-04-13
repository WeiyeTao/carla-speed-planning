"""
Imitation learning training for ANN and SNN planners.
Generates training data using the rule-based planner in CARLA,
then trains neural network planners to mimic its behavior.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import config as cfg

from planners.rule_planner import RulePlanner
from planners.ann_planner import ANNPlanner, ANNModel
from planners.snn_planner import SNNPlanner, SNNModel


def generate_dataset_from_carla(env, planner, num_episodes=5,
                                 max_steps=2000):
    """
    Generate imitation learning dataset by running the rule planner in CARLA.

    Returns:
        states: np.array (N, 3)
        targets: np.array (N,)
    """
    states_list = []
    targets_list = []

    for ep in range(num_episodes):
        print(f"  Collecting episode {ep + 1}/{num_episodes}")
        state = env.reset()

        for step in range(max_steps):
            target_speed = planner.get_target_speed(state)

            states_list.append(state.copy())
            targets_list.append(target_speed)

            # Use simple PID to get control
            from controller.pid_controller import PIDController
            pid = PIDController()
            speed = state[0]
            throttle, brake = pid.run(target_speed, speed)
            steer = env.get_waypoint_direction()

            state, done, _ = env.step(throttle, brake, steer)
            if done:
                break

    states = np.array(states_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32)

    return states, targets


def generate_synthetic_dataset(num_samples=None):
    """
    Generate a synthetic dataset without CARLA by sampling states
    and computing rule-based targets.
    Useful for offline training / testing.
    """
    num_samples = num_samples if num_samples is not None else cfg.SYNTHETIC_NUM_SAMPLES
    planner = RulePlanner()

    speeds = np.random.uniform(cfg.SYNTHETIC_SPEED_RANGE[0], cfg.SYNTHETIC_SPEED_RANGE[1], num_samples).astype(np.float32)
    curvatures = np.random.exponential(cfg.SYNTHETIC_CURVATURE_SCALE, num_samples).astype(np.float32)
    distances = np.random.uniform(cfg.SYNTHETIC_DISTANCE_RANGE[0], cfg.SYNTHETIC_DISTANCE_RANGE[1], num_samples).astype(np.float32)

    states = np.stack([speeds, curvatures, distances], axis=1)
    targets = np.array(
        [planner.get_target_speed(s) for s in states],
        dtype=np.float32
    )

    return states, targets


def train_ann(states, targets, epochs=None, lr=None, batch_size=None,
              save_path='models/ann_planner.pth'):
    """
    Train the ANN planner via supervised learning.

    Args:
        states: (N, 3) array of input states
        targets: (N,) array of target speeds
        epochs: training epochs
        lr: learning rate
        batch_size: batch size
        save_path: where to save trained model

    Returns:
        Trained ANNPlanner
    """
    epochs = epochs if epochs is not None else cfg.TRAIN_EPOCHS
    lr = lr if lr is not None else cfg.ANN_LR
    batch_size = batch_size if batch_size is not None else cfg.TRAIN_BATCH_SIZE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    X = torch.tensor(states, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ANNModel(input_dim=cfg.ANN_INPUT_DIM, hidden_dim=cfg.ANN_HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    print("\n  Training ANN planner...")
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.ANN_GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path + '.best')

        if (epoch + 1) % cfg.TRAIN_LOG_INTERVAL == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    # Load best model
    best_path = save_path + '.best'
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        os.rename(best_path, save_path)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)

    print(f"  ANN model saved to {save_path} (best loss: {best_loss:.6f})")

    planner = ANNPlanner(model_path=save_path, device=str(device))
    return planner


def train_snn(states, targets, epochs=None, lr=None, batch_size=None,
              num_steps=None,
              save_path='models/snn_planner.pth'):
    """
    Train the SNN planner via supervised learning.

    Args:
        states: (N, 3) array of input states
        targets: (N,) array of target speeds
        epochs: training epochs
        lr: learning rate (lower than ANN for SNN stability)
        batch_size: batch size
        num_steps: number of SNN simulation steps
        save_path: where to save trained model

    Returns:
        Trained SNNPlanner
    """
    epochs = epochs if epochs is not None else cfg.TRAIN_EPOCHS
    lr = lr if lr is not None else cfg.SNN_LR
    batch_size = batch_size if batch_size is not None else cfg.TRAIN_BATCH_SIZE
    num_steps = num_steps if num_steps is not None else cfg.SNN_NUM_STEPS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = torch.tensor(states, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SNNModel(input_dim=cfg.SNN_INPUT_DIM, hidden_dim=cfg.SNN_HIDDEN_DIM,
                     num_steps=num_steps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    print("\n  Training SNN planner...")
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output, _ = model(batch_x)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for SNN stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.SNN_GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path + '.best')

        if (epoch + 1) % cfg.TRAIN_LOG_INTERVAL == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    # Load best model
    best_path = save_path + '.best'
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        os.rename(best_path, save_path)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)

    print(f"  SNN model saved to {save_path} (best loss: {best_loss:.6f})")

    planner = SNNPlanner(model_path=save_path, num_steps=num_steps,
                         device=str(device))
    return planner
