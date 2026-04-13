"""
CARLA environment wrapper for speed planning experiments.
Handles connection, vehicle spawning, waypoint routing, state extraction,
and applying control commands.
"""

import carla
import math
import numpy as np
import time
import config as cfg


class CarlaEnv:
    """CARLA environment for speed planning comparison."""

    def __init__(self, host=None, port=None, town=None,
                 dt=None, max_steps=None):
        self.host = host if host is not None else cfg.CARLA_HOST
        self.port = port if port is not None else cfg.CARLA_PORT
        self.town = town
        self.dt = dt if dt is not None else cfg.DT
        self.max_steps = max_steps if max_steps is not None else cfg.MAX_STEPS

        self.client = None
        self.world = None
        self.vehicle = None
        self.bp_lib = None
        self.map = None

        self.waypoints = []
        self.current_wp_idx = 0
        self.step_count = 0

        self._spectator = None
        self._prev_location = None

    def connect(self):
        """Connect to CARLA server and use the currently loaded map."""
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(cfg.CARLA_TIMEOUT)

        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()
        print(f"  Using map: {self.map.name}")

        # Set synchronous mode with fixed timestep
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(settings)

        self.world.tick()

        self._spectator = self.world.get_spectator()

    def _build_route(self, start_wp, total_distance=None, spacing=None):
        """Build a fixed route from a starting waypoint by following the road."""
        total_distance = total_distance if total_distance is not None else cfg.ROUTE_TOTAL_DISTANCE
        spacing = spacing if spacing is not None else cfg.ROUTE_SPACING
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

    def _compute_curvature(self, idx):
        """Compute curvature at waypoint index using three-point method."""
        if idx <= 0 or idx >= len(self.waypoints) - 1:
            return 0.0

        p0 = self.waypoints[idx - 1].transform.location
        p1 = self.waypoints[idx].transform.location
        p2 = self.waypoints[idx + 1].transform.location

        # Vectors
        dx1 = p1.x - p0.x
        dy1 = p1.y - p0.y
        dx2 = p2.x - p1.x
        dy2 = p2.y - p1.y

        # Cross product magnitude (approximate curvature)
        cross = abs(dx1 * dy2 - dy1 * dx2)
        d1 = math.sqrt(dx1**2 + dy1**2) + 1e-6
        d2 = math.sqrt(dx2**2 + dy2**2) + 1e-6

        curvature = 2.0 * cross / (d1 * d2 * (d1 + d2) + 1e-6)
        return curvature

    def get_state(self):
        """
        Extract low-dimensional state:
          [speed, curvature, distance_to_waypoint]
        """
        if self.vehicle is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # Distance to current target waypoint
        veh_loc = self.vehicle.get_location()
        wp_loc = self.waypoints[self.current_wp_idx].transform.location
        dist = math.sqrt(
            (veh_loc.x - wp_loc.x)**2 +
            (veh_loc.y - wp_loc.y)**2
        )

        curvature = self._compute_curvature(self.current_wp_idx)

        return np.array([speed, curvature, dist], dtype=np.float32)

    def _advance_waypoint(self):
        """Advance waypoint index if close enough to current target."""
        if self.current_wp_idx >= len(self.waypoints) - 1:
            return True  # route finished

        veh_loc = self.vehicle.get_location()
        wp_loc = self.waypoints[self.current_wp_idx].transform.location
        dist = math.sqrt(
            (veh_loc.x - wp_loc.x)**2 +
            (veh_loc.y - wp_loc.y)**2
        )

        if dist < cfg.WAYPOINT_ADVANCE_THRESHOLD:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                return True

        return False

    def get_waypoint_direction(self):
        """Get steering direction toward the next waypoint."""
        veh_transform = self.vehicle.get_transform()
        veh_loc = veh_transform.location
        veh_fwd = veh_transform.get_forward_vector()

        target_wp = self.waypoints[min(self.current_wp_idx + 1,
                                        len(self.waypoints) - 1)]
        target_loc = target_wp.transform.location

        # Direction vector to target
        dx = target_loc.x - veh_loc.x
        dy = target_loc.y - veh_loc.y
        dist = math.sqrt(dx**2 + dy**2) + 1e-6

        # Cross product for steering direction
        cross = veh_fwd.x * dy - veh_fwd.y * dx
        # Dot product for alignment
        dot = veh_fwd.x * dx + veh_fwd.y * dy

        steer = math.atan2(cross, dot)
        # Clamp to [-1, 1]
        steer = max(-1.0, min(1.0, steer * cfg.STEERING_GAIN))

        return steer

    def reset(self, spawn_index=None):
        """Reset the environment: destroy old vehicle, spawn new one, build route.

        Args:
            spawn_index: optional spawn point index. If None, uses cfg.SPAWN_POINT_INDEX.
        """
        self._cleanup()

        # Spawn vehicle at a chosen spawn point
        spawn_points = self.map.get_spawn_points()
        idx = spawn_index if spawn_index is not None else cfg.SPAWN_POINT_INDEX

        if idx >= len(spawn_points):
            print(f"  Warning: spawn index {idx} out of range "
                  f"(max {len(spawn_points) - 1}), using 0")
            idx = 0

        spawn_transform = spawn_points[idx]

        bp = self.bp_lib.filter(cfg.VEHICLE_BLUEPRINT)[0]
        self.vehicle = self.world.spawn_actor(bp, spawn_transform)

        # Let physics settle
        for _ in range(10):
            self.world.tick()

        # Build route from the spawn location
        start_wp = self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        self.waypoints = self._build_route(start_wp, total_distance=cfg.ROUTE_TOTAL_DISTANCE,
                                            spacing=cfg.ROUTE_SPACING)
        self.current_wp_idx = 0
        self.step_count = 0
        self._prev_location = self.vehicle.get_location()

        # Tick once to initialize
        self.world.tick()

        return self.get_state()

    def step(self, throttle, brake, steer):
        """
        Apply control and advance simulation.
        Returns: (state, done, info)
        """
        control = carla.VehicleControl()
        control.throttle = float(np.clip(throttle, 0.0, 1.0))
        control.brake = float(np.clip(brake, 0.0, 1.0))
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.hand_brake = False
        control.manual_gear_shift = False

        self.vehicle.apply_control(control)
        self.world.tick()

        self.step_count += 1

        # Advance waypoint
        route_done = self._advance_waypoint()

        # Check done conditions
        done = False
        if route_done:
            done = True
        if self.step_count >= self.max_steps:
            done = True

        state = self.get_state()

        # Compute displacement since last step
        current_location = self.vehicle.get_location()
        displacement = 0.0
        if self._prev_location is not None:
            dx = current_location.x - self._prev_location.x
            dy = current_location.y - self._prev_location.y
            displacement = math.sqrt(dx**2 + dy**2)
        self._prev_location = current_location

        # Update spectator for visual following
        self._update_spectator()

        info = {
            'waypoint_idx': self.current_wp_idx,
            'total_waypoints': len(self.waypoints),
            'route_done': route_done,
            'displacement': displacement,
        }

        return state, done, info

    def _update_spectator(self):
        """Move spectator camera to follow the vehicle from behind (chase cam)."""
        if self._spectator and self.vehicle:
            transform = self.vehicle.get_transform()
            fwd = transform.get_forward_vector()
            # Position camera behind and above the vehicle
            cam_loc = transform.location + carla.Location(
                x=-fwd.x * cfg.SPECTATOR_DISTANCE,
                y=-fwd.y * cfg.SPECTATOR_DISTANCE,
                z=cfg.SPECTATOR_HEIGHT
            )
            # Look toward the vehicle (use vehicle's yaw, pitched slightly down)
            cam_rot = carla.Rotation(
                pitch=cfg.SPECTATOR_PITCH,
                yaw=transform.rotation.yaw,
                roll=0
            )
            self._spectator.set_transform(carla.Transform(cam_loc, cam_rot))

    def _cleanup(self):
        """Destroy existing vehicle."""
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

    def close(self):
        """Clean up and restore async mode."""
        self._cleanup()
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

    def get_curvature(self):
        """Get curvature at current waypoint for logging."""
        return self._compute_curvature(self.current_wp_idx)
