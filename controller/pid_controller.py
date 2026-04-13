"""
PID controller for longitudinal (speed) control.
Converts target_speed into throttle/brake commands.
"""

import numpy as np
import config as cfg


class PIDController:
    """PID controller for speed tracking."""

    def __init__(self, kp=None, ki=None, kd=None, dt=None):
        self.kp = kp if kp is not None else cfg.PID_KP
        self.ki = ki if ki is not None else cfg.PID_KI
        self.kd = kd if kd is not None else cfg.PID_KD
        self.dt = dt if dt is not None else cfg.DT

        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        """Reset controller internal state."""
        self.integral = 0.0
        self.prev_error = 0.0

    def run(self, target_speed, current_speed):
        """
        Compute throttle and brake from speed error.

        Args:
            target_speed: desired speed (m/s)
            current_speed: current speed (m/s)

        Returns:
            (throttle, brake) both in [0, 1]
        """
        error = target_speed - current_speed

        # PID terms
        self.integral += error * self.dt
        # Anti-windup: clamp integral
        self.integral = np.clip(self.integral, -cfg.PID_WINDUP_LIMIT, cfg.PID_WINDUP_LIMIT)

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Convert to throttle/brake
        if output >= 0:
            throttle = float(np.clip(output, 0.0, 1.0))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-output, 0.0, 1.0))

        return throttle, brake
