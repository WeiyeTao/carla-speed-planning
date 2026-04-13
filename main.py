"""
Main entry point for the CARLA speed planning comparison experiment.

Usage:
    python main.py --mode all          # Train + evaluate all planners
    python main.py --mode train        # Train ANN and SNN only
    python main.py --mode evaluate     # Evaluate all planners (models must exist)
    python main.py --mode plot         # Generate plots from existing logs

Options:
    --host          CARLA server host (default: localhost)
    --port          CARLA server port (default: 2000)
    --episodes      Number of episodes per planner (default: 3)
    --max-steps     Max steps per episode (default: 2000)
    --train-epochs  Training epochs (default: 100)
    --no-debug      Disable CARLA debug overlay
    --synthetic     Use synthetic data for training (no CARLA needed)
    --realtime      Slow down simulation to ~real-time for visual observation
"""

import argparse
import os
import sys
import time
import math
import numpy as np
import config as cfg

from env.carla_env import CarlaEnv
from controller.pid_controller import PIDController
from planners.rule_planner import RulePlanner
from planners.ann_planner import ANNPlanner
from planners.snn_planner import SNNPlanner
from utils.logger import EpisodeLogger
from evaluation.metrics import (compute_all_metrics, print_metrics,
                                 compare_planners)
from visualization.plot import generate_all_plots, plot_spike_raster
from visualization.debug_draw import DebugDraw
from training.imitation import (train_ann, train_snn,
                                 generate_dataset_from_carla,
                                 generate_synthetic_dataset)


# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')
PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')

ANN_MODEL_PATH = os.path.join(MODELS_DIR, 'ann_planner.pth')
SNN_MODEL_PATH = os.path.join(MODELS_DIR, 'snn_planner.pth')


def parse_args():
    parser = argparse.ArgumentParser(
        description='CARLA Speed Planning Comparison')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'train', 'evaluate', 'plot'],
                        help='Execution mode')
    parser.add_argument('--host', type=str, default=cfg.CARLA_HOST)
    parser.add_argument('--port', type=int, default=cfg.CARLA_PORT)
    parser.add_argument('--episodes', type=int, default=cfg.EVAL_EPISODES,
                        help='Episodes per planner for evaluation')
    parser.add_argument('--max-steps', type=int, default=cfg.MAX_STEPS)
    parser.add_argument('--train-epochs', type=int, default=cfg.TRAIN_EPOCHS)
    parser.add_argument('--no-debug', action='store_true',
                        help='Disable CARLA debug overlay')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for training')
    parser.add_argument('--collect-episodes', type=int, default=cfg.COLLECT_EPISODES,
                        help='Episodes to collect for training data')
    parser.add_argument('--realtime', action='store_true',
                        help='Slow simulation to real-time for visual observation')
    return parser.parse_args()


# ── Training Phase ───────────────────────────────────────────────────────
def run_training(args, env=None):
    """Train ANN and SNN planners via imitation learning."""
    print("\n" + "=" * 60)
    print("  PHASE: TRAINING (Imitation Learning)")
    print("=" * 60)

    use_synthetic = args.synthetic or cfg.USE_SYNTHETIC_DATA or env is None
    if use_synthetic:
        print("\n  Generating synthetic training dataset...")
        states, targets = generate_synthetic_dataset(num_samples=cfg.SYNTHETIC_NUM_SAMPLES)
    else:
        print("\n  Collecting training data from CARLA...")
        rule_planner = RulePlanner()
        states, targets = generate_dataset_from_carla(
            env, rule_planner,
            num_episodes=args.collect_episodes,
            max_steps=args.max_steps
        )

    print(f"  Dataset: {len(states)} samples")

    ann_planner = None
    snn_planner = None

    # Train ANN (only if active)
    if 'ann' in cfg.ACTIVE_PLANNERS:
        ann_planner = train_ann(
            states, targets,
            epochs=args.train_epochs,
            save_path=ANN_MODEL_PATH
        )

    # Train SNN (only if active)
    if 'snn' in cfg.ACTIVE_PLANNERS:
        snn_planner = train_snn(
            states, targets,
            epochs=args.train_epochs,
            save_path=SNN_MODEL_PATH
        )

    return ann_planner, snn_planner


# ── Episode Runner ───────────────────────────────────────────────────────
def run_episode(env, planner, pid, logger, debug_draw=None,
                episode_id=0, spawn_index=None, realtime=False):
    """
    Run a single episode with the given planner.

    Returns:
        data: dict of logged data arrays
        metrics: dict of computed metrics
    """
    state = env.reset(spawn_index=spawn_index)
    pid.reset()
    logger.reset()

    done = False
    step = 0
    last_spike_record = None
    step_displacement = 0.0
    start_time = time.time()

    sp_info = f", spawn={spawn_index}" if spawn_index is not None else ""
    print(f"    Episode {episode_id + 1} started "
          f"({planner.name} planner{sp_info})...")

    while not done:
        # Planner produces target speed
        target_speed = planner.get_target_speed(state)

        # PID produces throttle/brake
        current_speed = state[0]
        throttle, brake = pid.run(target_speed, current_speed)

        # Lateral control: follow waypoints
        steer = env.get_waypoint_direction()

        # Log
        curvature = env.get_curvature()
        logger.log(
            time_step=step,
            speed=current_speed,
            target_speed=target_speed,
            throttle=throttle,
            brake=brake,
            steer=steer,
            curvature=curvature,
            displacement=step_displacement
        )

        # Debug overlay
        if debug_draw is not None:
            elapsed = time.time() - start_time
            debug_draw.draw_all(
                vehicle=env.vehicle,
                planner_name=planner.name,
                target_speed=target_speed,
                curvature=curvature,
                energy=logger.get_total_energy(),
                waypoints=env.waypoints,
                current_wp_idx=env.current_wp_idx,
                distance=logger.get_total_distance(),
                episode_id=episode_id,
                spawn_index=spawn_index,
                elapsed_time=elapsed
            )

        # Step
        state, done, info = env.step(throttle, brake, steer)
        step += 1
        step_displacement = info.get('displacement', 0.0)

        # Capture SNN spikes
        if hasattr(planner, 'get_last_spikes'):
            spikes = planner.get_last_spikes()
            if spikes is not None:
                last_spike_record = spikes

        # Real-time playback: sleep to match simulation dt
        if realtime:
            elapsed_wall = time.time() - start_time
            sim_time = step * env.dt
            sleep_time = sim_time - elapsed_wall
            if sleep_time > 0:
                time.sleep(sleep_time)

    data = logger.get_data()
    metrics = compute_all_metrics(data, dt=env.dt)

    print(f"    Episode {episode_id + 1} done: "
          f"{step} steps, energy={metrics['total_energy']:.2f}")

    return data, metrics, last_spike_record


# ── Evaluation Phase ─────────────────────────────────────────────────────
def run_evaluation(args, env, planners, use_debug=True):
    """
    Evaluate all planners across multiple spawn points and episodes.

    Args:
        args: parsed arguments
        env: CarlaEnv instance
        planners: list of planner instances
        use_debug: whether to draw debug overlay

    Returns:
        all_data: {planner_name: data (from best episode)}
        all_metrics: {planner_name: averaged metrics with _std}
        snn_spikes: last spike record from SNN
    """
    print("\n" + "=" * 60)
    print("  PHASE: EVALUATION")
    print("=" * 60)

    pid = PIDController()
    logger = EpisodeLogger()
    debug_draw = DebugDraw(env.world) if use_debug else None

    spawn_indices = cfg.SPAWN_POINT_INDICES

    all_data = {}
    all_metrics = {}
    snn_spikes = None

    for planner in planners:
        print(f"\n  --- Evaluating: {planner.name} ---")
        print(f"  Spawn points: {spawn_indices}, "
              f"Episodes per spawn: {args.episodes}")

        episode_metrics = []
        best_data = None
        best_energy = float('inf')

        for sp_idx in spawn_indices:
            for ep in range(args.episodes):
                data, metrics, spikes = run_episode(
                    env, planner, pid, logger,
                    debug_draw=debug_draw,
                    episode_id=ep,
                    spawn_index=sp_idx,
                    realtime=args.realtime
                )

                # Save episode log with spawn point in filename
                log_base = os.path.join(
                    LOGS_DIR,
                    f'{planner.name.lower()}_sp{sp_idx}_ep{ep}'
                )
                logger.save(log_base)

                episode_metrics.append(metrics)

                # Keep the episode with lowest energy as representative
                if metrics['total_energy'] < best_energy:
                    best_energy = metrics['total_energy']
                    best_data = data

                if spikes is not None:
                    snn_spikes = spikes

        # Compute mean and stddev across all episodes x spawn points
        avg_metrics = {}
        std_metrics = {}
        for key in episode_metrics[0]:
            vals = [m[key] for m in episode_metrics]
            avg_metrics[key] = float(np.mean(vals))
            std_metrics[key] = float(np.std(vals))

        avg_metrics['_std'] = std_metrics

        all_data[planner.name] = best_data
        all_metrics[planner.name] = avg_metrics

        print_metrics(avg_metrics, planner.name)

    # Comparison
    compare_planners(all_metrics)

    return all_data, all_metrics, snn_spikes


# ── Plot from existing logs ──────────────────────────────────────────────
def plot_from_logs(args):
    """Generate plots from existing log files."""
    print("\n  Loading logs from:", LOGS_DIR)

    data_dict = {}
    metrics_dict = {}

    for planner_name in ['Rule-based', 'ANN', 'SNN']:
        name_lower = planner_name.lower()
        # Try new naming convention first (with spawn point), fall back to old
        log_path = os.path.join(LOGS_DIR, f'{name_lower}_sp0_ep0.npz')
        if not os.path.exists(log_path):
            log_path = os.path.join(LOGS_DIR, f'{name_lower}_ep0.npz')

        if not os.path.exists(log_path):
            print(f"  Warning: no log found for {planner_name}, skipping")
            continue

        loaded = np.load(log_path)
        data = {key: loaded[key] for key in loaded.files}
        data_dict[planner_name] = data
        metrics_dict[planner_name] = compute_all_metrics(data)

    if data_dict:
        generate_all_plots(data_dict, metrics_dict,
                            output_dir=PLOTS_DIR)
    else:
        print("  No logs found. Run evaluation first.")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Plot-only mode ──
    if args.mode == 'plot':
        plot_from_logs(args)
        return

    # ── Connect to CARLA ──
    print("\n  Connecting to CARLA...")
    env = CarlaEnv(
        host=args.host,
        port=args.port,
        dt=cfg.DT,
        max_steps=args.max_steps
    )

    try:
        env.connect()
        print("  Connected to CARLA successfully.")

        # ── Training ──
        if args.mode in ('all', 'train') and cfg.STAGE_TRAIN:
            ann_planner, snn_planner = run_training(args, env)
        else:
            ann_planner = None
            snn_planner = None

        # ── Evaluation ──
        if args.mode in ('all', 'evaluate') and cfg.STAGE_EVALUATE:
            # Build planner list based on ACTIVE_PLANNERS config
            planners = []

            if 'rule' in cfg.ACTIVE_PLANNERS:
                planners.append(RulePlanner())

            if 'ann' in cfg.ACTIVE_PLANNERS:
                if ann_planner is None:
                    if os.path.exists(ANN_MODEL_PATH):
                        ann_planner = ANNPlanner(model_path=ANN_MODEL_PATH)
                    else:
                        print("  ANN model not found. "
                              "Training with synthetic data...")
                        states, targets = generate_synthetic_dataset()
                        ann_planner = train_ann(states, targets,
                                                save_path=ANN_MODEL_PATH)
                planners.append(ann_planner)

            if 'snn' in cfg.ACTIVE_PLANNERS:
                if snn_planner is None:
                    if os.path.exists(SNN_MODEL_PATH):
                        snn_planner = SNNPlanner(model_path=SNN_MODEL_PATH)
                    else:
                        print("  SNN model not found. "
                              "Training with synthetic data...")
                        states, targets = generate_synthetic_dataset()
                        snn_planner = train_snn(states, targets,
                                                 save_path=SNN_MODEL_PATH)
                planners.append(snn_planner)

            use_debug = not args.no_debug

            all_data, all_metrics, snn_spikes = run_evaluation(
                args, env, planners, use_debug=use_debug
            )

            # Generate plots
            if cfg.STAGE_PLOT:
                generate_all_plots(
                    all_data, all_metrics,
                    spike_record=snn_spikes,
                    output_dir=PLOTS_DIR
                )

            print("\n  Experiment complete!")
            print(f"  Logs saved to: {LOGS_DIR}")
            print(f"  Plots saved to: {PLOTS_DIR}")

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        env.close()
        print("  CARLA connection closed.")


if __name__ == '__main__':
    main()
