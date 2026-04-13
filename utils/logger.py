"""
Logging utility for recording per-timestep data during episodes.
Saves logs as both .csv and .npz formats.
"""

import os
import csv
import numpy as np
import config as cfg


class EpisodeLogger:
    """Logs per-timestep driving data for a single episode."""

    FIELDS = [
        'time', 'speed', 'target_speed', 'throttle', 'brake',
        'steer', 'energy', 'curvature', 'distance_traveled'
    ]

    def __init__(self):
        self.data = {field: [] for field in self.FIELDS}
        self.cumulative_energy = 0.0
        self.cumulative_distance = 0.0

    def reset(self):
        """Clear all logged data."""
        self.data = {field: [] for field in self.FIELDS}
        self.cumulative_energy = 0.0
        self.cumulative_distance = 0.0

    def log(self, time_step, speed, target_speed, throttle, brake,
            steer, curvature, displacement=0.0):
        """
        Log a single timestep.

        Energy model: energy = throttle^2 + 0.1 * brake
        """
        energy = throttle ** 2 + cfg.ENERGY_BRAKE_COEFF * brake
        self.cumulative_energy += energy
        self.cumulative_distance += displacement

        self.data['time'].append(time_step)
        self.data['speed'].append(speed)
        self.data['target_speed'].append(target_speed)
        self.data['throttle'].append(throttle)
        self.data['brake'].append(brake)
        self.data['steer'].append(steer)
        self.data['energy'].append(energy)
        self.data['curvature'].append(curvature)
        self.data['distance_traveled'].append(self.cumulative_distance)

    def get_total_energy(self):
        """Return cumulative energy for the episode."""
        return self.cumulative_energy

    def get_total_distance(self):
        """Return cumulative distance traveled for the episode."""
        return self.cumulative_distance

    def save_csv(self, filepath):
        """Save logged data to a CSV file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.FIELDS)
            n = len(self.data['time'])
            for i in range(n):
                row = [self.data[field][i] for field in self.FIELDS]
                writer.writerow(row)

    def save_npz(self, filepath):
        """Save logged data to a NPZ file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        arrays = {field: np.array(self.data[field]) for field in self.FIELDS}
        np.savez(filepath, **arrays)

    def save(self, filepath_base):
        """Save in both CSV and NPZ formats."""
        self.save_csv(filepath_base + '.csv')
        self.save_npz(filepath_base + '.npz')

    def get_data(self):
        """Return data as dict of numpy arrays."""
        return {field: np.array(vals) for field, vals in self.data.items()}
