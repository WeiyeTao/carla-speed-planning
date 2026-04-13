"""
ANN-based speed planner.
Uses a simple MLP to predict target speed from low-dimensional state.
Trained via imitation learning from the rule-based planner.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import config as cfg


class ANNModel(nn.Module):
    """Simple MLP: 3 -> 64 -> 64 -> 1"""

    def __init__(self, input_dim=None, hidden_dim=None):
        super().__init__()
        input_dim = input_dim if input_dim is not None else cfg.ANN_INPUT_DIM
        hidden_dim = hidden_dim if hidden_dim is not None else cfg.ANN_HIDDEN_DIM
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class ANNPlanner:
    """ANN planner wrapping the MLP model."""

    def __init__(self, model_path=None, input_dim=None, hidden_dim=None,
                 device='cpu'):
        self.device = torch.device(device)
        input_dim = input_dim if input_dim is not None else cfg.ANN_INPUT_DIM
        hidden_dim = hidden_dim if hidden_dim is not None else cfg.ANN_HIDDEN_DIM
        self.model = ANNModel(input_dim, hidden_dim).to(self.device)
        self.name = "ANN"

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.model.eval()

    def get_target_speed(self, state):
        """
        Predict target speed from state.

        state = [speed, curvature, distance_to_waypoint]
        """
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            x = x.to(self.device)
            pred = self.model(x).item()
        # Clamp to reasonable range
        return float(np.clip(pred, cfg.ANN_OUTPUT_CLIP_MIN, cfg.ANN_OUTPUT_CLIP_MAX))

    def save(self, path):
        """Save model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load model weights."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.model.eval()
