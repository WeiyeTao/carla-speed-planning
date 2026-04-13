"""
SNN-based speed planner.
Uses snnTorch with LIF neurons and direct current injection to predict target speed.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import config as cfg

try:
    import snntorch as snn
    from snntorch import surrogate
    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False
    print("Warning: snntorch not installed. SNN planner will not work.")


class SNNModel(nn.Module):
    """
    SNN with LIF neurons.
    Architecture: Norm -> Linear -> LIF -> Linear -> LIF -> Linear -> LIF
    Uses direct current injection (no stochastic encoding) and
    mean membrane potential readout for stable training.
    """

    def __init__(self, input_dim=None, hidden_dim=None, num_steps=None,
                 beta=None):
        super().__init__()
        input_dim = input_dim if input_dim is not None else cfg.SNN_INPUT_DIM
        hidden_dim = hidden_dim if hidden_dim is not None else cfg.SNN_HIDDEN_DIM
        num_steps = num_steps if num_steps is not None else cfg.SNN_NUM_STEPS
        beta = beta if beta is not None else cfg.SNN_BETA
        self.num_steps = num_steps

        spike_grad = surrogate.fast_sigmoid(slope=cfg.SNN_SURROGATE_SLOPE)

        # Input normalization
        self.norm = nn.BatchNorm1d(input_dim)

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Output layer
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                               output=True)

    def forward(self, x):
        """
        Forward pass with spike processing over num_steps.

        Args:
            x: input tensor (batch, input_dim)

        Returns:
            output: predicted target speed (batch, 1)
            spike_record: list of spike tensors for visualization
        """
        # Normalize input
        x_norm = self.norm(x)

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spike_record = []
        mem3_record = []

        for step in range(self.num_steps):
            # Direct current injection: feed normalized continuous
            # values every timestep (no stochastic encoding)
            cur1 = self.fc1(x_norm)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spike_record.append(spk1.detach())
            mem3_record.append(mem3)

        # Mean membrane potential across all timesteps for stable readout
        mem3_stack = torch.stack(mem3_record, dim=0)  # (T, batch, 1)
        output = mem3_stack.mean(dim=0)  # (batch, 1)

        return output, spike_record


class SNNPlanner:
    """SNN planner wrapping the SNN model."""

    def __init__(self, model_path=None, input_dim=None, hidden_dim=None,
                 num_steps=None, device='cpu'):
        if not SNN_AVAILABLE:
            raise ImportError(
                "snntorch is required for SNN planner. "
                "Install with: pip install snntorch"
            )

        self.device = torch.device(device)
        input_dim = input_dim if input_dim is not None else cfg.SNN_INPUT_DIM
        hidden_dim = hidden_dim if hidden_dim is not None else cfg.SNN_HIDDEN_DIM
        num_steps = num_steps if num_steps is not None else cfg.SNN_NUM_STEPS
        self.num_steps = num_steps
        self.model = SNNModel(
            input_dim, hidden_dim, num_steps
        ).to(self.device)
        self.name = "SNN"

        self._last_spike_record = None

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.model.eval()

    def get_target_speed(self, state):
        """
        Predict target speed from state using SNN.

        state = [speed, curvature, distance_to_waypoint]
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            x = x.to(self.device)
            output, spike_record = self.model(x)
            self._last_spike_record = spike_record
            pred = output.item()

        return float(np.clip(pred, cfg.SNN_OUTPUT_CLIP_MIN, cfg.SNN_OUTPUT_CLIP_MAX))

    def get_last_spikes(self):
        """Return spike record from the last forward pass (for visualization)."""
        return self._last_spike_record

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
