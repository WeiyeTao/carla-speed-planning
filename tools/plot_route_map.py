"""
Bird's-eye view of evaluation routes – rendered inside CARLA and saved as matplotlib plot.

Connects to a running CARLA server, builds the route for each spawn point,
draws them in the CARLA world using debug lines, moves the spectator camera
to a top-down view, and also saves a matplotlib figure to plots/route_map.png.

Usage:
    python3 tools/plot_route_map.py [--host localhost] [--port 2000] [--duration 30]

Requires a running CARLA server. Does not modify any project code.
"""

import sys
import os
import argparse
import time
import math

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import carla
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import config as cfg


def hex_to_carla_color(hex_str, a=255):
    """Convert hex color string to carla.Color."""
    hex_str = hex_str.lstrip('#')
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return carla.Color(r, g, b, a)


def build_route(carla_map, spawn_transform, total_distance, spacing):
    """Build a waypoint route from a spawn transform (mirrors CarlaEnv._build_route)."""
    start_wp = carla_map.get_waypoint(
        spawn_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    route = [start_wp]
    dist_covered = 0.0
    current = start_wp
    while dist_covered < total_distance:
        nexts = current.next(spacing)
        if not nexts:
            break
        current = nexts[0]
        route.append(current)
        dist_covered += spacing
    return route


def draw_routes_in_carla(world, routes, colors_hex, draw_life_time):
    """Draw all routes in the CARLA world using debug lines and points."""
    debug = world.debug
    z_offset = 1.0  # slightly above road surface

    for i, (sp_idx, route) in enumerate(routes):
        color = hex_to_carla_color(colors_hex[i % len(colors_hex)])
        dim_color = hex_to_carla_color(colors_hex[i % len(colors_hex)], a=180)

        for j in range(len(route) - 1):
            loc0 = route[j].transform.location + carla.Location(z=z_offset)
            loc1 = route[j + 1].transform.location + carla.Location(z=z_offset)
            debug.draw_line(loc0, loc1, thickness=0.5, color=color,
                            life_time=draw_life_time)

        # Start point – large green sphere
        start_loc = route[0].transform.location + carla.Location(z=z_offset + 0.5)
        debug.draw_point(start_loc, size=0.5,
                         color=carla.Color(0, 255, 0), life_time=draw_life_time)
        debug.draw_string(start_loc + carla.Location(z=1.5),
                          f"S{sp_idx} Start", color=carla.Color(255, 255, 255),
                          life_time=draw_life_time)

        # End point – red sphere
        end_loc = route[-1].transform.location + carla.Location(z=z_offset + 0.5)
        debug.draw_point(end_loc, size=0.5,
                         color=carla.Color(255, 0, 0), life_time=draw_life_time)
        debug.draw_string(end_loc + carla.Location(z=1.5),
                          f"S{sp_idx} End", color=carla.Color(255, 255, 255),
                          life_time=draw_life_time)

        # Direction arrows every ~60m
        arrow_step = max(1, int(60 / cfg.ROUTE_SPACING))
        for j in range(arrow_step, len(route) - 1, arrow_step):
            loc = route[j].transform.location + carla.Location(z=z_offset)
            fwd = route[j].transform.get_forward_vector()
            end = loc + carla.Location(x=fwd.x * 5, y=fwd.y * 5, z=0)
            debug.draw_arrow(loc, end, thickness=0.3, arrow_size=0.5,
                             color=color, life_time=draw_life_time)


def set_spectator_birdseye(world, routes):
    """Move spectator camera to a top-down view covering all routes."""
    # Collect all waypoint locations
    all_x, all_y, all_z = [], [], []
    for _, route in routes:
        for wp in route:
            loc = wp.transform.location
            all_x.append(loc.x)
            all_y.append(loc.y)
            all_z.append(loc.z)

    cx = (min(all_x) + max(all_x)) / 2
    cy = (min(all_y) + max(all_y)) / 2
    cz = np.mean(all_z)

    # Height based on route spread
    span_x = max(all_x) - min(all_x)
    span_y = max(all_y) - min(all_y)
    height = max(span_x, span_y) * 0.8 + 50  # enough to see everything

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        carla.Location(x=cx, y=cy, z=cz + height),
        carla.Rotation(pitch=-90, yaw=0, roll=0),  # straight down
    ))
    return cx, cy, height


def main():
    parser = argparse.ArgumentParser(description="Plot bird's-eye view of CARLA routes")
    parser.add_argument('--host', default=cfg.CARLA_HOST)
    parser.add_argument('--port', type=int, default=cfg.CARLA_PORT)
    parser.add_argument('--duration', type=int, default=30,
                        help='Seconds to keep routes visible in CARLA (default: 30)')
    args = parser.parse_args()

    # Connect to CARLA
    client = carla.Client(args.host, args.port)
    client.set_timeout(cfg.CARLA_TIMEOUT)
    world = client.get_world()
    carla_map = world.get_map()
    print(f"Connected to CARLA – map: {carla_map.name}")

    spawn_points = carla_map.get_spawn_points()
    indices = cfg.SPAWN_POINT_INDICES
    colors_hex = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    # --- Build routes ---
    routes = []       # list of (sp_idx, [waypoints])
    routes_xy = []    # for matplotlib
    for sp_idx in indices:
        if sp_idx >= len(spawn_points):
            print(f"Warning: spawn index {sp_idx} out of range, skipping")
            continue
        route = build_route(
            carla_map, spawn_points[sp_idx],
            cfg.ROUTE_TOTAL_DISTANCE, cfg.ROUTE_SPACING,
        )
        xs = [wp.transform.location.x for wp in route]
        ys = [wp.transform.location.y for wp in route]
        routes.append((sp_idx, route))
        routes_xy.append((sp_idx, xs, ys))
        print(f"  Spawn {sp_idx}: {len(route)} waypoints")

    if not routes:
        print("No valid routes to plot.")
        return

    # --- Draw in CARLA ---
    draw_life_time = float(args.duration)
    draw_routes_in_carla(world, routes, colors_hex, draw_life_time)
    set_spectator_birdseye(world, routes)

    # Tick a few times to make sure debug draws are rendered
    settings = world.get_settings()
    was_sync = settings.synchronous_mode
    if was_sync:
        for _ in range(5):
            world.tick()

    print(f"\nRoutes drawn in CARLA window (visible for {args.duration}s).")
    print("Switch to the CARLA window to see the bird's-eye view.")
    print("You can take a screenshot from CARLA, or use the matplotlib plot below.\n")

    # --- Matplotlib plot (also saved) ---
    fig, ax = plt.subplots(figsize=(10, 10), dpi=cfg.PLOT_DPI)

    handles = []
    for i, (sp_idx, xs, ys) in enumerate(routes_xy):
        color = colors_hex[i % len(colors_hex)]
        ys_inv = [-y for y in ys]

        ax.plot(xs, ys_inv, color=color, linewidth=2, alpha=0.85)

        ax.scatter(xs[0], ys_inv[0], color=color, s=120, zorder=5,
                   marker='o', edgecolors='black', linewidths=1.0)
        ax.scatter(xs[-1], ys_inv[-1], color=color, s=120, zorder=5,
                   marker='s', edgecolors='black', linewidths=1.0)

        step = max(1, len(xs) // 8)
        for j in range(step, len(xs) - 1, step):
            dx = xs[j + 1] - xs[j]
            dy = ys_inv[j + 1] - ys_inv[j]
            ax.annotate('', xy=(xs[j] + dx, ys_inv[j] + dy),
                        xytext=(xs[j], ys_inv[j]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.8))

        handles.append(mpatches.Patch(color=color, label=f'Spawn point {sp_idx}'))

    handles.append(plt.Line2D([0], [0], marker='o', color='grey', linestyle='None',
                              markersize=8, markeredgecolor='black', label='Start'))
    handles.append(plt.Line2D([0], [0], marker='s', color='grey', linestyle='None',
                              markersize=8, markeredgecolor='black', label='End'))

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f"Evaluation Routes – Bird's-Eye View\n"
                 f"(Map: {carla_map.name}, {cfg.ROUTE_TOTAL_DISTANCE:.0f} m per route)",
                 fontsize=14)
    ax.legend(handles=handles, fontsize=10, loc='best')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'route_map.png')
    fig.savefig(out_path, dpi=cfg.PLOT_DPI, bbox_inches='tight')
    print(f"Matplotlib plot saved to {os.path.abspath(out_path)}")
    plt.close(fig)

    # Keep script alive so user can observe in CARLA
    print(f"\nWaiting {args.duration}s – observe in CARLA window (Ctrl+C to exit early)...")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    print("Done.")


if __name__ == '__main__':
    main()
