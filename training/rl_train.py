"""
Reinforcement learning training (optional).
Trains ANN/SNN planners using a reward function that balances
energy efficiency, smoothness, collision avoidance, and progress.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config as cfg


class RLTrainer:
    """
    Simple REINFORCE-based RL trainer for speed planners.

    Reward:
        R = -energy - lambda1 * |jerk| - lambda2 * collision + lambda3 * progress
    """

    def __init__(self, planner_model, lr=None, gamma=None,
                 lambda1=None, lambda2=None, lambda3=None):
        """
        Args:
            planner_model: the nn.Module to train (ANN or SNN model)
            lr: learning rate
            gamma: discount factor
            lambda1: jerk penalty weight
            lambda2: collision penalty weight
            lambda3: progress reward weight
        """
        self.model = planner_model
        lr = lr if lr is not None else cfg.RL_LR
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma if gamma is not None else cfg.RL_GAMMA
        self.lambda1 = lambda1 if lambda1 is not None else cfg.RL_LAMBDA_JERK
        self.lambda2 = lambda2 if lambda2 is not None else cfg.RL_LAMBDA_COLLISION
        self.lambda3 = lambda3 if lambda3 is not None else cfg.RL_LAMBDA_PROGRESS

        self.log_probs = []
        self.rewards = []

    def compute_reward(self, throttle, brake, speed, prev_speed,
                       prev_prev_speed, collision=False, progress=0.0,
                       dt=0.05):
        """
        Compute reward for a single timestep.

        Args:
            throttle: current throttle value
            brake: current brake value
            speed: current speed
            prev_speed: speed at t-1
            prev_prev_speed: speed at t-2
            collision: whether collision occurred
            progress: waypoint progress delta
            dt: timestep

        Returns:
            reward (float)
        """
        # Energy cost
        energy = throttle ** 2 + cfg.ENERGY_BRAKE_COEFF * brake

        # Jerk computation
        if prev_prev_speed is not None and prev_speed is not None:
            acc = (speed - prev_speed) / dt
            prev_acc = (prev_speed - prev_prev_speed) / dt
            jerk = abs(acc - prev_acc)
        else:
            jerk = 0.0

        # Collision penalty
        coll_penalty = 1.0 if collision else 0.0

        reward = (
            -energy
            - self.lambda1 * jerk
            - self.lambda2 * coll_penalty
            + self.lambda3 * progress
        )

        return reward

    def store_transition(self, log_prob, reward):
        """Store a transition for the current episode."""
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def update(self):
        """
        Update policy using REINFORCE.
        Call at the end of each episode.
        """
        if not self.rewards:
            return 0.0

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        # Clear buffers
        total_reward = sum(self.rewards)
        self.log_probs = []
        self.rewards = []

        return total_reward

    def train_episode(self, env, planner_type='ann', pid_controller=None):
        """
        Run one RL training episode in CARLA.

        Args:
            env: CarlaEnv instance
            planner_type: 'ann' or 'snn'
            pid_controller: PIDController instance

        Returns:
            total_reward for the episode
        """
        from controller.pid_controller import PIDController

        if pid_controller is None:
            pid_controller = PIDController()

        state = env.reset()
        pid_controller.reset()

        prev_speed = None
        prev_prev_speed = None
        prev_wp_idx = 0

        done = False
        while not done:
            # Forward pass with gradient
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            x.requires_grad_(True)

            if planner_type == 'snn':
                output, _ = self.model(x)
            else:
                output = self.model(x)

            # Add small noise for exploration
            noise = torch.randn_like(output) * cfg.RL_EXPLORATION_NOISE
            target_speed_tensor = output + noise

            # Compute log probability (Gaussian policy)
            log_prob = -0.5 * ((target_speed_tensor - output) ** 2)
            log_prob = log_prob.sum()

            target_speed = float(
                np.clip(target_speed_tensor.item(), cfg.SNN_OUTPUT_CLIP_MIN, cfg.SNN_OUTPUT_CLIP_MAX)
            )

            # PID control
            current_speed = state[0]
            throttle, brake = pid_controller.run(target_speed, current_speed)
            steer = env.get_waypoint_direction()

            state, done, info = env.step(throttle, brake, steer)

            # Progress
            wp_idx = info['waypoint_idx']
            progress = wp_idx - prev_wp_idx
            prev_wp_idx = wp_idx

            # Reward
            reward = self.compute_reward(
                throttle, brake, state[0],
                prev_speed, prev_prev_speed,
                collision=False, progress=progress
            )

            self.store_transition(log_prob, reward)

            prev_prev_speed = prev_speed
            prev_speed = state[0]

        total_reward = self.update()
        return total_reward

    def save_model(self, path):
        """Save the trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
