"""
Evaluation metrics for comparing planner performance.
Computes energy, average speed, travel time, and jerk from logged data.
"""

import numpy as np
import config as cfg


def compute_total_energy(data):
    """Total accumulated energy over the episode."""
    return float(np.sum(data['energy']))


def compute_average_speed(data):
    """Average speed over the episode."""
    return float(np.mean(data['speed']))


def compute_travel_time(data, dt=None):
    """Total travel time (number of steps * dt)."""
    dt = dt if dt is not None else cfg.DT
    return len(data['time']) * dt


def compute_jerk(data, dt=None):
    """
    Compute jerk (rate of change of acceleration).

    jerk[t] = (acc[t] - acc[t-1]) / dt
    acc[t] = (speed[t] - speed[t-1]) / dt

    Returns: (jerk_array, mean_abs_jerk, max_abs_jerk)
    """
    dt = dt if dt is not None else cfg.DT
    speed = data['speed']

    if len(speed) < 3:
        return np.array([0.0]), 0.0, 0.0

    # Acceleration
    acc = np.diff(speed) / dt

    # Jerk
    jerk = np.diff(acc) / dt

    mean_abs_jerk = float(np.mean(np.abs(jerk)))
    max_abs_jerk = float(np.max(np.abs(jerk)))

    return jerk, mean_abs_jerk, max_abs_jerk


def compute_all_metrics(data, dt=None):
    """
    Compute all evaluation metrics from logged data.

    Args:
        data: dict with keys 'time', 'speed', 'energy', etc.
        dt: simulation timestep

    Returns:
        dict with all metrics
    """
    dt = dt if dt is not None else cfg.DT
    jerk_values, mean_abs_jerk, max_abs_jerk = compute_jerk(data, dt)

    metrics = {
        'total_energy': compute_total_energy(data),
        'average_speed': compute_average_speed(data),
        'travel_time': compute_travel_time(data, dt),
        'mean_abs_jerk': mean_abs_jerk,
        'max_abs_jerk': max_abs_jerk,
        'max_speed': float(np.max(data['speed'])),
        'min_speed': float(np.min(data['speed'])),
        'total_steps': len(data['time']),
    }

    # Distance-normalized energy
    if 'distance_traveled' in data and len(data['distance_traveled']) > 0:
        total_distance = float(data['distance_traveled'][-1])
    else:
        total_distance = 0.0
    metrics['distance_traveled'] = total_distance
    metrics['energy_per_meter'] = (metrics['total_energy'] / total_distance
                                   if total_distance > 0 else float('inf'))

    return metrics


def print_metrics(metrics, planner_name=""):
    """Print metrics in a formatted table."""
    header = f"Metrics for {planner_name}" if planner_name else "Metrics"
    print(f"\n{'=' * 50}")
    print(f"  {header}")
    print(f"{'=' * 50}")
    print(f"  Total Energy:      {metrics['total_energy']:.4f}")
    if 'energy_per_meter' in metrics and metrics.get('distance_traveled', 0) > 0:
        print(f"  Energy/Meter:      {metrics['energy_per_meter']:.6f}")
        print(f"  Distance Traveled: {metrics['distance_traveled']:.2f} m")
    print(f"  Average Speed:     {metrics['average_speed']:.4f} m/s")
    print(f"  Travel Time:       {metrics['travel_time']:.2f} s")
    print(f"  Mean |Jerk|:       {metrics['mean_abs_jerk']:.4f} m/s^3")
    print(f"  Max |Jerk|:        {metrics['max_abs_jerk']:.4f} m/s^3")
    print(f"  Max Speed:         {metrics['max_speed']:.4f} m/s")
    print(f"  Min Speed:         {metrics['min_speed']:.4f} m/s")
    print(f"  Total Steps:       {metrics['total_steps']}")
    print(f"{'=' * 50}\n")


def compare_planners(results_dict):
    """
    Compare metrics across planners.

    Args:
        results_dict: {planner_name: metrics_dict}

    Returns:
        Summary comparison dict.
    """
    summary = {}
    metric_keys = ['total_energy', 'energy_per_meter', 'distance_traveled',
                   'average_speed', 'travel_time',
                   'mean_abs_jerk', 'max_abs_jerk']

    for key in metric_keys:
        summary[key] = {}
        for name, metrics in results_dict.items():
            summary[key][name] = metrics[key]

    print("\n" + "=" * 90)
    print("  PLANNER COMPARISON")
    print("=" * 90)
    print(f"  {'Metric':<20} ", end="")
    for name in results_dict:
        print(f"{name:<22} ", end="")
    print()
    print("-" * 90)

    for key in metric_keys:
        label = key.replace('_', ' ').title()
        print(f"  {label:<20} ", end="")
        for name in results_dict:
            val = results_dict[name][key]
            std_dict = results_dict[name].get('_std', {})
            std = std_dict.get(key, 0.0)
            if std > 0:
                print(f"{val:.4f}+/-{std:.4f}  ", end="")
            else:
                print(f"{val:<22.4f} ", end="")
        print()

    print("=" * 90)

    return summary
