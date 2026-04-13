"""
Rule-based speed planner.
Uses a simple interpretable function to compute target speed
based on curvature and distance to the next waypoint.
"""

import numpy as np
import config as cfg


class RulePlanner:
    """Rule-based planner using curvature-inverse speed formula."""

    def __init__(self, max_speed=None, k=None, alpha=None,
                 min_speed=None):
        """
        Args:
            max_speed: maximum target speed (m/s)
            k: base speed constant
            alpha: curvature sensitivity
            min_speed: minimum target speed (m/s)
        """
        self.max_speed = max_speed if max_speed is not None else cfg.RULE_MAX_SPEED
        self.k = k if k is not None else cfg.RULE_K
        self.alpha = alpha if alpha is not None else cfg.RULE_ALPHA
        self.min_speed = min_speed if min_speed is not None else cfg.RULE_MIN_SPEED
        self.name = "Rule-based"

    def get_target_speed(self, state):
        """
        Compute target speed from state.

        state = [speed, curvature, distance_to_waypoint]

        Formula: target_speed = k / (1 + alpha * |curvature|)
        Clamped to [min_speed, max_speed].
        """
        curvature = abs(state[1])

        target = self.k / (1.0 + self.alpha * curvature)
        target = np.clip(target, self.min_speed, self.max_speed)

        return float(target)
