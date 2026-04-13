"""
CARLA debug drawing overlay.
Renders real-time information on the simulation window.
"""

import carla
import math
import config as cfg


class DebugDraw:
    """Draws debug information in the CARLA world."""

    def __init__(self, world):
        self.world = world

    def draw_info_text(self, vehicle, planner_name, target_speed,
                        curvature, energy, distance=0.0,
                        episode_id=None, spawn_index=None,
                        elapsed_time=0.0, route_progress=0.0):
        """
        Draw an info overlay near the vehicle showing planner state.

        Args:
            vehicle: CARLA vehicle actor
            planner_name: name of the current planner
            target_speed: current target speed from planner
            curvature: current road curvature
            energy: cumulative energy
            distance: cumulative distance traveled (meters)
            episode_id: current episode number (0-based)
            spawn_index: current spawn point index
            elapsed_time: elapsed wall-clock time (seconds)
            route_progress: fraction of route completed [0, 1]
        """
        vel = vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # Speed tracking error for color coding
        speed_error = abs(speed - target_speed)

        # Color based on speed tracking quality
        if speed_error < 1.0:
            text_color = carla.Color(r=100, g=255, b=100)   # green
        elif speed_error < 3.0:
            text_color = carla.Color(r=255, g=255, b=100)   # yellow
        else:
            text_color = carla.Color(r=255, g=100, b=100)   # red

        loc = vehicle.get_location()
        text_loc = carla.Location(x=loc.x, y=loc.y, z=loc.z + cfg.DEBUG_TEXT_Z_OFFSET)

        progress_pct = route_progress * 100.0

        # Build episode/spawn label
        ep_label = ""
        if episode_id is not None:
            ep_label += f"Ep: {episode_id + 1}"
        if spawn_index is not None:
            ep_label += f"  Spawn: {spawn_index}"

        info_text = (
            f"Planner: {planner_name}\n"
            f"Speed: {speed:.2f} / {target_speed:.2f} m/s  "
            f"(err: {speed_error:.2f})\n"
            f"Curvature: {curvature:.4f}\n"
            f"Energy: {energy:.2f}   Dist: {distance:.1f} m\n"
            f"Time: {elapsed_time:.1f}s   "
            f"Progress: {progress_pct:.0f}%"
        )
        if ep_label:
            info_text += f"\n{ep_label}"

        self.world.debug.draw_string(
            text_loc,
            info_text,
            draw_shadow=True,
            color=text_color,
            life_time=cfg.DEBUG_LIFE_TIME
        )

    def draw_waypoints(self, waypoints, current_idx, look_ahead=None):
        """
        Draw waypoints on the road.

        Args:
            waypoints: list of CARLA waypoints
            current_idx: current target waypoint index
            look_ahead: how many waypoints ahead to draw
        """
        look_ahead = look_ahead if look_ahead is not None else cfg.DEBUG_LOOK_AHEAD
        start = max(0, current_idx - cfg.DEBUG_LOOK_BEHIND)
        end = min(len(waypoints), current_idx + look_ahead)

        for i in range(start, end):
            wp = waypoints[i]
            loc = wp.transform.location
            loc.z += cfg.DEBUG_WP_Z_OFFSET  # raise slightly above road

            if i == current_idx:
                # Current target: green
                color = carla.Color(r=0, g=255, b=0)
                size = 0.15
            elif i < current_idx:
                # Passed: gray
                color = carla.Color(r=128, g=128, b=128)
                size = 0.08
            else:
                # Upcoming: blue
                color = carla.Color(r=0, g=100, b=255)
                size = 0.1

            self.world.debug.draw_point(
                loc, size=size, color=color, life_time=cfg.DEBUG_LIFE_TIME
            )

    def draw_trajectory(self, waypoints, current_idx, look_ahead=None):
        """
        Draw trajectory lines connecting waypoints.
        """
        look_ahead = look_ahead if look_ahead is not None else cfg.DEBUG_LOOK_AHEAD
        start = current_idx
        end = min(len(waypoints), current_idx + look_ahead)

        for i in range(start, end - 1):
            loc1 = waypoints[i].transform.location
            loc2 = waypoints[i + 1].transform.location
            loc1.z += cfg.DEBUG_TRAJECTORY_Z_OFFSET
            loc2.z += cfg.DEBUG_TRAJECTORY_Z_OFFSET

            self.world.debug.draw_line(
                loc1, loc2,
                thickness=cfg.DEBUG_TRAJECTORY_THICKNESS,
                color=carla.Color(r=255, g=200, b=0),
                life_time=cfg.DEBUG_LIFE_TIME
            )

    def draw_all(self, vehicle, planner_name, target_speed, curvature,
                  energy, waypoints, current_wp_idx, distance=0.0,
                  episode_id=None, spawn_index=None,
                  elapsed_time=0.0):
        """Draw all debug overlays at once."""
        route_progress = current_wp_idx / max(len(waypoints) - 1, 1)

        self.draw_info_text(vehicle, planner_name, target_speed,
                             curvature, energy,
                             distance=distance,
                             episode_id=episode_id,
                             spawn_index=spawn_index,
                             elapsed_time=elapsed_time,
                             route_progress=route_progress)
        self.draw_waypoints(waypoints, current_wp_idx)
        self.draw_trajectory(waypoints, current_wp_idx)
