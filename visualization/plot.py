"""
Matplotlib-based plotting for evaluation results.
Generates comparison plots across planners.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import config as cfg


def plot_speed_vs_time(data_dict, dt=None, save_path='plots/speed_vs_time.png'):
    """
    Plot speed over time for all planners.

    Args:
        data_dict: {planner_name: data_dict_from_logger}
        dt: timestep
        save_path: output file path
    """
    dt = dt if dt is not None else cfg.DT
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=cfg.PLOT_FIGSIZE_WIDE)

    for name, data in data_dict.items():
        time_axis = np.arange(len(data['speed'])) * dt
        ax.plot(time_axis, data['speed'], label=f'{name} (actual)', linewidth=1.5)
        ax.plot(time_axis, data['target_speed'], '--',
                label=f'{name} (target)', alpha=0.6, linewidth=1.0)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_energy_vs_time(data_dict, dt=None,
                         save_path='plots/energy_vs_time.png'):
    """
    Plot cumulative energy over time for all planners.
    """
    dt = dt if dt is not None else cfg.DT
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=cfg.PLOT_FIGSIZE_WIDE)

    for name, data in data_dict.items():
        time_axis = np.arange(len(data['energy'])) * dt
        cum_energy = np.cumsum(data['energy'])
        ax.plot(time_axis, cum_energy, label=name, linewidth=1.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Energy')
    ax.set_title('Energy vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_control_signals(data_dict, dt=None,
                          save_path='plots/control_signals.png'):
    """
    Plot throttle and brake signals for all planners.
    """
    dt = dt if dt is not None else cfg.DT
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_planners = len(data_dict)
    fig, axes = plt.subplots(n_planners, 1, figsize=(12, 4 * n_planners),
                              sharex=True)
    if n_planners == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, data_dict.items()):
        time_axis = np.arange(len(data['throttle'])) * dt
        ax.plot(time_axis, data['throttle'], label='Throttle',
                color='green', alpha=0.7)
        ax.plot(time_axis, data['brake'], label='Brake',
                color='red', alpha=0.7)
        ax.set_ylabel('Control Value')
        ax.set_title(f'{name} - Control Signals')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_jerk_vs_time(data_dict, dt=None,
                       save_path='plots/jerk_vs_time.png'):
    """
    Plot jerk (smoothness metric) over time for all planners.
    """
    dt = dt if dt is not None else cfg.DT
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=cfg.PLOT_FIGSIZE_WIDE)

    for name, data in data_dict.items():
        speed = data['speed']
        if len(speed) < 3:
            continue
        acc = np.diff(speed) / dt
        jerk = np.diff(acc) / dt
        time_axis = np.arange(len(jerk)) * dt
        ax.plot(time_axis, jerk, label=name, alpha=0.7, linewidth=1.0)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Jerk (m/s^3)')
    ax.set_title('Jerk vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_energy_bar_chart(metrics_dict,
                           save_path='plots/energy_comparison.png'):
    """
    Bar chart comparing total energy across planners.

    Args:
        metrics_dict: {planner_name: metrics_dict}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    names = list(metrics_dict.keys())
    energies = [metrics_dict[n]['total_energy'] for n in names]

    fig, ax = plt.subplots(figsize=cfg.PLOT_FIGSIZE_BAR)

    colors = cfg.PLOT_COLORS
    bars = ax.bar(names, energies, color=colors[:len(names)], width=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Total Energy')
    ax.set_title('Total Energy Comparison')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_metrics_comparison(metrics_dict,
                             save_path='plots/metrics_comparison.png'):
    """
    Multi-bar chart comparing all key metrics across planners.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    names = list(metrics_dict.keys())
    metric_keys = ['total_energy', 'energy_per_meter', 'average_speed',
                   'travel_time', 'mean_abs_jerk', 'max_abs_jerk']
    metric_labels = ['Total Energy', 'Energy/Meter', 'Avg Speed (m/s)',
                     'Travel Time (s)', 'Mean |Jerk|', 'Max |Jerk|']

    fig, axes = plt.subplots(1, 6, figsize=cfg.PLOT_FIGSIZE_METRICS)
    colors = cfg.PLOT_COLORS

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        values = [metrics_dict[n][key] for n in names]
        bars = ax.bar(names, values, color=colors[:len(names)], width=0.5)
        ax.set_title(label)
        ax.grid(True, axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(values),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_spike_raster(spike_record, num_neurons=None,
                       save_path='plots/spike_raster.png'):
    """
    Plot SNN spike raster diagram.

    Args:
        spike_record: list of spike tensors (from SNN forward pass)
        num_neurons: number of neurons to display
        save_path: output file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if spike_record is None or len(spike_record) == 0:
        print("  No spike data to plot.")
        return

    num_neurons = num_neurons if num_neurons is not None else cfg.SPIKE_RASTER_NEURONS

    fig, ax = plt.subplots(figsize=cfg.PLOT_FIGSIZE_SPIKE)

    # spike_record: list of (batch, hidden_dim) tensors
    # Take first batch element
    spike_trains = []
    for t, spk in enumerate(spike_record):
        if hasattr(spk, 'numpy'):
            spk_np = spk.cpu().numpy()
        else:
            spk_np = np.array(spk)
        if spk_np.ndim > 1:
            spk_np = spk_np[0]  # first batch element
        spike_trains.append(spk_np[:num_neurons])

    spike_matrix = np.array(spike_trains)  # (num_steps, num_neurons)

    # Create event plot
    events = []
    for neuron_idx in range(min(num_neurons, spike_matrix.shape[1])):
        spike_times = np.where(spike_matrix[:, neuron_idx] > 0)[0]
        events.append(spike_times)

    ax.eventplot(events, lineoffsets=range(len(events)),
                  linelengths=0.8, colors='black')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Neuron Index')
    ax.set_title('SNN Spike Raster')
    ax.set_xlim(0, len(spike_record))
    ax.set_ylim(-0.5, len(events) - 0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI)
    plt.close()
    print(f"  Saved: {save_path}")


def generate_all_plots(data_dict, metrics_dict, spike_record=None,
                        output_dir='plots', dt=None):
    """
    Generate comparison plots based on config toggles.

    Args:
        data_dict: {planner_name: logged_data}
        metrics_dict: {planner_name: metrics}
        spike_record: SNN spike data (optional)
        output_dir: output directory
        dt: simulation timestep
    """
    print("\nGenerating plots...")
    dt = dt if dt is not None else cfg.DT

    if cfg.PLOT_SPEED_VS_TIME:
        plot_speed_vs_time(data_dict, dt,
                           os.path.join(output_dir, 'speed_vs_time.png'))
    if cfg.PLOT_ENERGY_VS_TIME:
        plot_energy_vs_time(data_dict, dt,
                             os.path.join(output_dir, 'energy_vs_time.png'))
    if cfg.PLOT_CONTROL_SIGNALS:
        plot_control_signals(data_dict, dt,
                              os.path.join(output_dir, 'control_signals.png'))
    if cfg.PLOT_JERK_VS_TIME:
        plot_jerk_vs_time(data_dict, dt,
                           os.path.join(output_dir, 'jerk_vs_time.png'))
    if cfg.PLOT_ENERGY_BAR:
        plot_energy_bar_chart(metrics_dict,
                               os.path.join(output_dir, 'energy_comparison.png'))
    if cfg.PLOT_METRICS_COMPARISON:
        plot_metrics_comparison(metrics_dict,
                                 os.path.join(output_dir,
                                              'metrics_comparison.png'))

    if cfg.PLOT_SPIKE_RASTER and spike_record is not None:
        plot_spike_raster(spike_record,
                           save_path=os.path.join(output_dir,
                                                   'spike_raster.png'))

    print("All plots generated.\n")
