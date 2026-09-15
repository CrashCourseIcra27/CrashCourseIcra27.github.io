#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Low grip cut in with spinout (Town04).

A vehicle cuts across the ego's lane on a reduced-grip surface and spins out.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional

import carla
import py_trees

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import Criterion, CollisionTest
from srunner.scenariomanager.timer import GameTime


def get_value_parameter(config, name, p_type, default):
    """Read a scalar custom XML parameter."""
    other_parameters = dict(config.other_parameters) if hasattr(config, "other_parameters") else {}
    if hasattr(config, "scenario") and hasattr(config.scenario, "other_parameters"):
        other_parameters.update(config.scenario.other_parameters)
    if name in other_parameters:
        raw = other_parameters[name]
        if isinstance(raw, dict):
            return p_type(raw["value"])
        return p_type(raw)
    return default


def get_bool_parameter(config, name, default):
    """Read a boolean custom XML parameter."""
    other_parameters = dict(config.other_parameters) if hasattr(config, "other_parameters") else {}
    if hasattr(config, "scenario") and hasattr(config.scenario, "other_parameters"):
        other_parameters.update(config.scenario.other_parameters)
    if name not in other_parameters:
        return default
    raw = other_parameters[name]
    value = str(raw["value"] if isinstance(raw, dict) else raw).strip().lower()
    return value in ("1", "true", "yes", "y", "on")


class _EgoSpeedKeeper(py_trees.behaviour.Behaviour):
    def __init__(self, ego, min_speed_kmh=0.0, throttle=1.0, name="EgoSpeedKeeper"):
        super().__init__(name)
        self._ego = ego
        self._min_mps = float(min_speed_kmh) / 3.6
        self._throttle = float(throttle)

    def update(self):
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.RUNNING
        velocity = self._ego.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        if speed < self._min_mps:
            control = self._ego.get_control()
            control.brake = 0.0
            control.throttle = self._throttle
            self._ego.apply_control(control)
        return py_trees.common.Status.RUNNING


def clamp_steer(value: float) -> float:
    """Clamp steering to CARLA's valid range."""
    return max(-1.0, min(1.0, float(value)))


def planar_distance(location_a: carla.Location, location_b: carla.Location) -> float:
    """2D distance between two CARLA locations."""
    return math.hypot(location_a.x - location_b.x, location_a.y - location_b.y)


DISALLOWED_AMBIENT_MODELS = {
    "vehicle.yamaha.yzf",
    "vehicle.harley-davidson.low_rider",
    "vehicle.kawasaki.ninja",
    "vehicle.gazelle.omafiets",
    "vehicle.diamondback.century",
    "vehicle.bh.crossbike",
}


class RecorderControl(py_trees.behaviour.Behaviour):
    """Start or stop the CARLA recorder."""

    def __init__(self, start: bool, recorder_path: str = "", name: str = "RecorderControl"):
        super().__init__(name)
        self._start = start
        self._recorder_path = recorder_path

    def update(self):
        client = CarlaDataProvider.get_client()
        if client is None:
            return py_trees.common.Status.FAILURE

        try:
            if self._start:
                client.start_recorder(self._recorder_path)
            else:
                client.stop_recorder()
        except RuntimeError:
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.SUCCESS


class ApplyVehicleControlOnce(py_trees.behaviour.Behaviour):
    """Apply a one-shot control update."""

    def __init__(self, vehicle: carla.Vehicle, throttle: float = 0.0, brake: float = 0.0,
                 steer: float = 0.0, hand_brake: bool = False, name: str = "ApplyVehicleControlOnce"):
        super().__init__(name)
        self._vehicle = vehicle
        self._throttle = float(throttle)
        self._brake = float(brake)
        self._steer = float(steer)
        self._hand_brake = bool(hand_brake)

    def update(self):
        if self._vehicle is None or not self._vehicle.is_alive:
            return py_trees.common.Status.SUCCESS

        control = carla.VehicleControl()
        control.throttle = self._throttle
        control.brake = self._brake
        control.steer = self._steer
        control.hand_brake = self._hand_brake
        self._vehicle.apply_control(control)
        return py_trees.common.Status.SUCCESS


class CollisionFlagCriterion(Criterion):
    """Criterion that succeeds only if the ego collision sensor fired."""

    def __init__(self, actor: carla.Actor, scenario_ref, optional=False, name="CollisionFlagCriterion"):
        super().__init__(name, actor, optional=optional)
        self._scenario_ref = scenario_ref
        self.success_value = 1
        self.actual_value = 0
        self.units = "bool"

    def update(self):
        self.actual_value = 1 if self._scenario_ref._impact_detected else 0  # pylint: disable=protected-access
        if self.actual_value:
            self.test_status = "SUCCESS"
        else:
            self.test_status = "RUNNING"
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self.test_status != "SUCCESS":
            self.test_status = "FAILURE"
        super().terminate(new_status)


class PacedIdle(py_trees.behaviour.Behaviour):
    """Idle for a duration while maintaining wall-clock pacing."""

    def __init__(self, scenario_ref, duration: float, name: str = "PacedIdle"):
        super().__init__(name)
        self._scenario = scenario_ref
        self._duration = float(duration)
        self._start_time = None

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        now = GameTime.get_time()
        elapsed = 0.0 if self._start_time is None else max(0.0, now - self._start_time)
        self._scenario._pace_to_wall_clock(now)  # pylint: disable=protected-access
        if elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class HoldScenarioActorsOnImpact(py_trees.behaviour.Behaviour):
    """Keep scenario actors visible briefly, then clean them up."""

    def __init__(self, scenario_ref, actors, name="HoldScenarioActorsOnImpact"):
        super().__init__(name)
        self._scenario = scenario_ref
        self._actors = actors
        self._held = False
        self._hold_start = None
        self._cleaned = False

    def update(self):
        if not self._scenario._impact_detected:  # pylint: disable=protected-access
            return py_trees.common.Status.RUNNING

        if not self._held:
            self._hold_start = time.monotonic()
            self._held = True

        for actor in self._actors:
            if actor is None or not actor.is_alive:
                continue
            try:
                actor.set_autopilot(False)
                actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                actor.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=1.0, hand_brake=True,
                ))
            except RuntimeError:
                pass

        elapsed = time.monotonic() - self._hold_start
        if elapsed < self._scenario._post_impact_hold:  # pylint: disable=protected-access
            return py_trees.common.Status.RUNNING

        if not self._cleaned:
            for actor in self._actors:
                if actor is None or not actor.is_alive:
                    continue
                try:
                    actor.destroy()
                except RuntimeError:
                    pass
            self._cleaned = True
        return py_trees.common.Status.SUCCESS


class SystemTimeDelay(py_trees.behaviour.Behaviour):
    """Wait for wall-clock time (not sim time)."""

    def __init__(self, duration: float, name: str = "SystemTimeDelay"):
        super().__init__(name)
        self._duration = max(0.0, float(duration))
        self._start_wall_time = None

    def initialise(self):
        self._start_wall_time = time.monotonic()

    def update(self):
        if self._start_wall_time is None:
            self._start_wall_time = time.monotonic()
        if (time.monotonic() - self._start_wall_time) >= self._duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class RunHighwayPitManeuver(py_trees.behaviour.Behaviour):
    """
    Time-phased direct control for the adversary and supporting traffic.

    The ego vehicle is expected to be externally controlled by an agent such as TF++.
    """

    def __init__(self, scenario_ref, name: str = "RunHighwayPitManeuver"):
        super().__init__(name)
        self._scenario = scenario_ref
        self._start_time = None
        self._start_wall_time = None
        self._merge_started_at = None

    def initialise(self):
        self._start_time = GameTime.get_time()
        self._start_wall_time = time.monotonic()
        self._scenario._wall_start_time = self._start_wall_time  # pylint: disable=protected-access
        self._merge_started_at = None

    def update(self):
        ego = self._scenario.ego_vehicles[0] if self._scenario.ego_vehicles else None
        adversary = self._scenario._adversary  # pylint: disable=protected-access

        if ego is None or adversary is None:
            return py_trees.common.Status.FAILURE
        if not ego.is_alive or not adversary.is_alive:
            return py_trees.common.Status.SUCCESS

        if self._scenario._impact_detected:  # pylint: disable=protected-access
            return py_trees.common.Status.SUCCESS

        now = GameTime.get_time()
        elapsed = 0.0 if self._start_time is None else max(0.0, now - self._start_time)

        self._scenario._pace_to_wall_clock(elapsed)  # pylint: disable=protected-access
        # Spectator camera updates are intentionally disabled for this scenario.
        # (Keep spectator params parsed for backward compatibility with existing XMLs.)
        self._scenario._apply_gap_traffic_controls()  # pylint: disable=protected-access

        if self._merge_started_at is None:
            gap = self._scenario._compute_adversary_rear_gap()  # pylint: disable=protected-access
            if elapsed >= self._scenario._min_merge_delay and gap <= self._scenario._merge_trigger_gap:  # pylint: disable=protected-access
                self._merge_started_at = elapsed

        merge_elapsed = None if self._merge_started_at is None else max(0.0, elapsed - self._merge_started_at)
        adversary_control = self._scenario._build_adversary_control(elapsed, merge_elapsed)  # pylint: disable=protected-access
        adversary.apply_control(adversary_control)

        if elapsed >= self._scenario._max_run_time:  # pylint: disable=protected-access
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING


class LowGripCutInWithSpinoutTown04(BasicScenario):
    """
    Ego drives straight under external agent control.
    A slightly-ahead car in the left lane cuts across and initiates the collision.
    """

    timeout = 90

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=90):
        self.timeout = timeout
        self._world = world
        self._wall_start_time = None

        self._adversary: Optional[carla.Vehicle] = None
        self._ego_gap_lead: Optional[carla.Vehicle] = None
        self._adjacent_gap_lead: Optional[carla.Vehicle] = None
        self._collision_sensor = None
        self._background_actors: List[carla.Actor] = []
        self._impact_detected = False
        self._impact_log: List[str] = []

        self._ambient_vehicle_count = int(get_value_parameter(config, "ambient_vehicle_count", float, 40))
        self._ego_lane_follow_gain = get_value_parameter(config, "ego_lane_follow_gain", float, 0.55)
        self._ego_heading_gain = get_value_parameter(config, "ego_heading_gain", float, 1.15)
        self._ego_min_speed_kmh = get_value_parameter(config, "ego_min_speed_kmh", float, 0.0)
        self._ego_force_throttle = get_value_parameter(config, "ego_force_throttle", float, 1.0)
        self._adversary_cruise_speed = (
            get_value_parameter(config, "adversary_cruise_speed_kmh", float, 95.0) / 3.6
        )
        self._adversary_cut_speed = (
            get_value_parameter(config, "adversary_cut_speed_kmh", float, 105.0) / 3.6
        )
        self._adversary_cut_steer = get_value_parameter(config, "adversary_cut_steer", float, 0.42)
        self._adversary_countersteer = get_value_parameter(config, "adversary_countersteer", float, -0.12)
        self._adversary_countersteer_speed = (
            get_value_parameter(config, "adversary_countersteer_speed_kmh", float, 90.0) / 3.6
        )
        self._adversary_settle_steer = get_value_parameter(config, "adversary_settle_steer", float, 0.05)
        self._adversary_settle_speed = (
            get_value_parameter(config, "adversary_settle_speed_kmh", float, 60.0) / 3.6
        )
        self._recovery_steer_gain = get_value_parameter(config, "recovery_steer_gain", float, 0.55)
        self._recovery_oscillation_period = get_value_parameter(
            config, "recovery_oscillation_period", float, 0.35)
        self._recovery_decay = get_value_parameter(config, "recovery_decay", float, 1.8)
        self._recovery_panic_window = get_value_parameter(config, "recovery_panic_window", float, 1.0)
        self._recovery_panic_multiplier = get_value_parameter(
            config, "recovery_panic_multiplier", float, 1.2)
        self._recovery_floor_gain = get_value_parameter(config, "recovery_floor_gain", float, 0.35)
        self._recovery_snap_gain = get_value_parameter(config, "recovery_snap_gain", float, 0.45)
        self._recovery_snap_sharpness = get_value_parameter(config, "recovery_snap_sharpness", float, 2.2)
        self._min_merge_delay = get_value_parameter(config, "min_merge_delay", float, 1.0)
        self._merge_trigger_gap = get_value_parameter(config, "merge_trigger_gap", float, 4.5)
        self._cut_duration = get_value_parameter(config, "cut_duration", float, 2.3)
        self._countersteer_duration = get_value_parameter(config, "countersteer_duration", float, 0.6)
        self._settle_duration = get_value_parameter(config, "settle_duration", float, 0.8)
        self._max_run_time = get_value_parameter(config, "max_run_time", float, 15.0)
        self._post_impact_delay = get_value_parameter(config, "post_impact_delay", float, 0.5)
        self._post_impact_hold = get_value_parameter(config, "post_impact_hold", float, 5.0)
        self._gap_vehicle_model = str(get_value_parameter(config, "gap_vehicle_model", str, "vehicle.tesla.model3"))

        # Unused legacy parameters (kept parsed for backward-compatible XML files).
        # They currently do not affect scenario behavior.
        _ = get_value_parameter(config, "adversary_release_brake", float, 0.0)
        _ = str(get_value_parameter(config, "recorder_path", str, "accident_scenario.log"))

        self._ego_lead_vehicle_model = str(
            get_value_parameter(config, "ego_lead_vehicle_model", str, "vehicle.carlamotors.european_hgv")
        )
        self._adjacent_lead_vehicle_model = str(
            get_value_parameter(config, "adjacent_lead_vehicle_model", str, self._gap_vehicle_model)
        )

        self._ego_lead_vehicle_distance = get_value_parameter(
            config, "ego_lead_vehicle_distance", float, 14.0
        )
        self._adjacent_lead_vehicle_distance = get_value_parameter(
            config, "adjacent_lead_vehicle_distance", float, 14.0
        )
        self._gap_lane_offset_delta = get_value_parameter(config, "gap_lane_offset_delta", float, 0.0)
        self._ego_lead_vehicle_speed = get_value_parameter(
            config, "ego_lead_vehicle_speed_kmh", float, 80.0
        ) / 3.6
        self._adjacent_lead_vehicle_speed = get_value_parameter(
            config, "adjacent_lead_vehicle_speed_kmh", float, 72.0
        ) / 3.6
        self._gap_vehicle_max_steer = get_value_parameter(config, "gap_vehicle_max_steer", float, 0.22)

        self._spectator_distance = get_value_parameter(config, "spectator_distance", float, 15.0)
        self._spectator_lateral_offset = get_value_parameter(config, "spectator_lateral_offset", float, 5.0)
        self._spectator_height = get_value_parameter(config, "spectator_height", float, 8.0)
        self._spectator_pitch = get_value_parameter(config, "spectator_pitch", float, -20.0)

        self._head_start_distance = get_value_parameter(config, "head_start_distance", float, 7.0)
        self._lane_offset = get_value_parameter(config, "lane_offset", float, 3.5)

        self._ambient_spawn_radius = get_value_parameter(config, "ambient_spawn_radius", float, 120.0)
        self._ambient_exclusion_radius = get_value_parameter(config, "ambient_exclusion_radius", float, 18.0)
        self._ambient_alignment_max_deg = get_value_parameter(config, "ambient_alignment_max_deg", float, 55.0)
        self._road_friction_scale = get_value_parameter(config, "road_friction_scale", float, 0.55)
        self._realtime_factor = get_value_parameter(config, "realtime_factor", float, 1.0)
        self._end_on_collision = get_bool_parameter(config, "end_on_collision", True)

        super().__init__(
            name="LowGripCutInWithSpinoutTown04",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            terminate_on_failure=True,
            criteria_enable=criteria_enable,
        )

    def _initialize_environment(self, world):
        """Apply weather plus an explicit road-friction trigger for real skid behavior."""
        super()._initialize_environment(world)

        # Stabilize contact solving to avoid unrealistic "zero-g" launches on impact.
        try:
            settings = world.get_settings()
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
            world.apply_settings(settings)
        except RuntimeError:
            pass

        friction_scale = max(0.1, float(self._road_friction_scale))
        friction_bp = world.get_blueprint_library().find("static.trigger.friction")
        extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
        friction_bp.set_attribute("friction", str(friction_scale))
        friction_bp.set_attribute("extent_x", str(extent.x))
        friction_bp.set_attribute("extent_y", str(extent.y))
        friction_bp.set_attribute("extent_z", str(extent.z))

        transform = carla.Transform()
        transform.location = carla.Location(-10000.0, -10000.0, 0.0)
        try:
            world.spawn_actor(friction_bp, transform)
        except RuntimeError:
            pass

    def _initialize_actors(self, config):
        if not self.ego_vehicles:
            raise ValueError("LowGripCutInWithSpinoutTown04 requires one ego vehicle")
        if len(config.other_actors) < 1:
            raise ValueError("LowGripCutInWithSpinoutTown04 requires one <other_actor> adversary")

        ego_transform = self.ego_vehicles[0].get_transform()
        initial_adversary_transform = self._build_initial_adversary_spawn(ego_transform)
        self._adversary = CarlaDataProvider.request_new_actor(
            config.other_actors[0].model,
            initial_adversary_transform,
            rolename=config.other_actors[0].rolename,
            autopilot=False,
            color=config.other_actors[0].color,
        )
        if self._adversary is None:
            raise ValueError("Unable to spawn the adversary")
        self.other_actors.append(self._adversary)
        self._configure_vehicle_physics(self.ego_vehicles[0])
        self._configure_vehicle_physics(self._adversary)

        self._snap_primary_actors_to_collision_setup()
        self._spawn_gap_traffic()
        self._attach_collision_sensor()
        self._spawn_ambient_traffic()

    def _configure_vehicle_physics(self, vehicle: Optional[carla.Vehicle]):
        """Apply stability-oriented CARLA physics flags without changing control tuning."""
        if vehicle is None or not vehicle.is_alive:
            return

        try:
            vehicle.set_simulate_physics(True)
        except RuntimeError:
            return

        try:
            physics = vehicle.get_physics_control()
        except RuntimeError:
            return

        changed = False
        if hasattr(physics, "use_sweep_wheel_collision") and not physics.use_sweep_wheel_collision:
            physics.use_sweep_wheel_collision = True
            changed = True

        if changed:
            try:
                vehicle.apply_physics_control(physics)
            except RuntimeError:
                pass

    def _build_initial_adversary_spawn(self, ego_transform):
        """Build a safe non-overlapping spawn before snapping onto the adjacent lane."""
        forward = ego_transform.rotation.get_forward_vector()
        right = ego_transform.rotation.get_right_vector()
        location = ego_transform.location + carla.Location(
            x=forward.x * self._head_start_distance,
            y=forward.y * self._head_start_distance,
            z=0.0,
        )
        location -= carla.Location(
            x=right.x * 4.0,
            y=right.y * 4.0,
            z=0.0,
        )
        location.z += 0.3
        return carla.Transform(location, ego_transform.rotation)

    def _snap_primary_actors_to_collision_setup(self):
        """
        Align the adversary relative to the ego's current lane direction.

        Important:
        - We do NOT move the ego anymore.
        - The ego is expected to come from the route/agent setup.
        """
        world_map = CarlaDataProvider.get_map()
        ego = self.ego_vehicles[0]

        seed_wp = world_map.get_waypoint(
            ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if seed_wp is None:
            raise ValueError("Unable to find a driving lane for the ego start")

        ego_location = ego.get_location()
        lane_transform = seed_wp.transform

        forward = lane_transform.rotation.get_forward_vector()
        right = lane_transform.rotation.get_right_vector()

        adversary_location = ego_location + carla.Location(
            x=forward.x * self._head_start_distance,
            y=forward.y * self._head_start_distance,
            z=0.0,
        )
        adversary_location -= carla.Location(
            x=right.x * self._lane_offset,
            y=right.y * self._lane_offset,
            z=0.0,
        )
        adversary_location.z = ego_location.z + 0.3

        adversary_transform = carla.Transform(adversary_location, lane_transform.rotation)
        self._adversary.set_transform(adversary_transform)

    def _attach_collision_sensor(self):
        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            bp,
            carla.Transform(),
            attach_to=self.ego_vehicles[0],
        )
        self.other_actors.append(self._collision_sensor)

        def _on_collision(event):
            self._impact_detected = True
            self._impact_log.append(
                "collision_with={} impulse=({:.2f},{:.2f},{:.2f})".format(
                    event.other_actor.type_id,
                    event.normal_impulse.x,
                    event.normal_impulse.y,
                    event.normal_impulse.z,
                )
            )

        self._collision_sensor.listen(_on_collision)

    def _build_forward_from_waypoint(self, base_wp, distance):
        """Return a transform ahead of a specific lane waypoint."""
        if base_wp is None:
            return None

        candidates = base_wp.next(max(1.0, float(distance)))
        if not candidates:
            remaining = max(1.0, float(distance))
            cursor = base_wp
            while remaining > 0.5:
                step = min(8.0, remaining)
                next_wps = cursor.next(step)
                if not next_wps:
                    break
                cursor = next_wps[0]
                remaining -= step
            target_wp = cursor
        else:
            target_wp = candidates[0]

        transform = target_wp.transform
        transform.location.z += 0.5
        return transform

    @staticmethod
    def _find_adjacent_driving_waypoint(base_wp, prefer_left=True):
        """Find the nearest adjacent driving lane from a base waypoint."""
        if base_wp is None:
            return None

        walker = base_wp
        for _ in range(4):
            walker = walker.get_left_lane() if prefer_left else walker.get_right_lane()
            if walker is None:
                return None
            if walker.lane_type == carla.LaneType.Driving:
                return walker
        return None

    def _spawn_gap_traffic(self):
        """Spawn lead vehicles ahead of ego and in the adjacent lane."""
        ego = self.ego_vehicles[0] if self.ego_vehicles else None
        adversary = self._adversary
        if ego is None or adversary is None:
            return

        map_obj = CarlaDataProvider.get_map()
        ego_wp = None if map_obj is None else map_obj.get_waypoint(
            ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        prefer_left = self._lane_offset > 0.0
        adjacent_wp = self._find_adjacent_driving_waypoint(ego_wp, prefer_left=prefer_left)
        if adjacent_wp is None:
            adjacent_wp = None if map_obj is None else map_obj.get_waypoint(
                adversary.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

        adjacent_lead_distance = self._adjacent_lead_vehicle_distance + self._gap_lane_offset_delta
        ego_lead_transform = self._build_forward_from_waypoint(ego_wp, self._ego_lead_vehicle_distance)
        adjacent_lead_transform = self._build_forward_from_waypoint(adjacent_wp, adjacent_lead_distance)

        if ego_lead_transform is not None and adjacent_lead_transform is not None:
            lead_spacing = planar_distance(ego_lead_transform.location, adjacent_lead_transform.location)
            if lead_spacing < max(6.0, self._ambient_exclusion_radius * 0.5):
                adjacent_lead_transform = self._build_forward_from_waypoint(
                    adjacent_wp, adjacent_lead_distance + max(8.0, abs(self._gap_lane_offset_delta))
                )

        if ego_lead_transform is not None:
            ego_lead_model = self._ego_lead_vehicle_model
            self._ego_gap_lead = CarlaDataProvider.request_new_actor(
                ego_lead_model,
                ego_lead_transform,
                rolename="gap_lead_ego_lane",
                autopilot=False,
                color="180,180,180",
            )
            if self._ego_gap_lead is None and ego_lead_model != self._gap_vehicle_model:
                self._ego_gap_lead = CarlaDataProvider.request_new_actor(
                    self._gap_vehicle_model,
                    ego_lead_transform,
                    rolename="gap_lead_ego_lane",
                    autopilot=False,
                    color="160,160,160",
                )
            if self._ego_gap_lead is not None:
                self._configure_vehicle_physics(self._ego_gap_lead)
                self.other_actors.append(self._ego_gap_lead)

        if adjacent_lead_transform is not None:
            adjacent_lead_model = self._adjacent_lead_vehicle_model
            self._adjacent_gap_lead = CarlaDataProvider.request_new_actor(
                adjacent_lead_model,
                adjacent_lead_transform,
                rolename="gap_lead_adjacent_lane",
                autopilot=False,
                color="145,145,145",
            )
            if self._adjacent_gap_lead is None and adjacent_lead_model != self._gap_vehicle_model:
                self._adjacent_gap_lead = CarlaDataProvider.request_new_actor(
                    self._gap_vehicle_model,
                    adjacent_lead_transform,
                    rolename="gap_lead_adjacent_lane",
                    autopilot=False,
                    color="145,145,145",
                )
            if self._adjacent_gap_lead is not None:
                self._configure_vehicle_physics(self._adjacent_gap_lead)
                self.other_actors.append(self._adjacent_gap_lead)

    def _update_spectator(self):
        ego = self.ego_vehicles[0] if self.ego_vehicles else None
        if ego is None or not ego.is_alive:
            return

        spectator = self._world.get_spectator()
        transform = ego.get_transform()
        forward = transform.rotation.get_forward_vector()
        right = transform.rotation.get_right_vector()
        location = transform.location - carla.Location(
            x=forward.x * self._spectator_distance,
            y=forward.y * self._spectator_distance,
            z=0.0,
        )
        location += carla.Location(
            x=right.x * self._spectator_lateral_offset,
            y=right.y * self._spectator_lateral_offset,
            z=self._spectator_height,
        )
        rotation = carla.Rotation(
            pitch=self._spectator_pitch,
            yaw=transform.rotation.yaw,
            roll=0.0,
        )
        spectator.set_transform(carla.Transform(location, rotation))

    def _pace_to_wall_clock(self, sim_elapsed):
        """Slow the synchronous scenario loop so sim time does not outrun wall time."""
        if self._realtime_factor <= 0.0:
            return
        if self._wall_start_time is None:
            self._wall_start_time = time.monotonic()

        expected_wall_elapsed = sim_elapsed / self._realtime_factor
        current_wall_elapsed = time.monotonic() - self._wall_start_time
        sleep_time = expected_wall_elapsed - current_wall_elapsed
        if sleep_time > 0.0:
            time.sleep(min(sleep_time, 0.1))

    def _spawn_ambient_traffic(self):
        if self._ambient_vehicle_count <= 0:
            return

        tm = CarlaDataProvider.get_client().get_trafficmanager(CarlaDataProvider.get_traffic_manager_port())
        if hasattr(tm, "set_global_distance_to_leading_vehicle"):
            tm.set_global_distance_to_leading_vehicle(2.0)
        if hasattr(tm, "global_percentage_speed_difference"):
            tm.global_percentage_speed_difference(0.0)

        ego_transform = self.ego_vehicles[0].get_transform()
        protected_actors = [self.ego_vehicles[0], self._adversary]
        protected_locations = [
            actor.get_transform().location for actor in protected_actors
            if actor is not None and actor.is_alive
        ]

        map_obj = CarlaDataProvider.get_map()
        spawn_points = map_obj.get_spawn_points() if map_obj is not None else []
        ambient_spawn_radius = max(1.0, self._ambient_spawn_radius)
        core_exclusion_radius = max(2.5, min(self._ambient_exclusion_radius, 4.0))
        blueprint_library = self._world.get_blueprint_library()
        allowed_ambient_models = [
            blueprint.id for blueprint in blueprint_library.filter("vehicle.*")
            if blueprint.id not in DISALLOWED_AMBIENT_MODELS
        ]
        if not allowed_ambient_models:
            return

        def _yaw_delta_deg(yaw_a, yaw_b):
            delta = (yaw_a - yaw_b + 180.0) % 360.0 - 180.0
            return abs(delta)

        candidates = []
        for spawn_point in spawn_points:
            distance_to_ego = planar_distance(spawn_point.location, ego_transform.location)
            if distance_to_ego > ambient_spawn_radius:
                continue
            if any(
                planar_distance(spawn_point.location, protected_location) < core_exclusion_radius
                for protected_location in protected_locations
            ):
                continue
            if _yaw_delta_deg(spawn_point.rotation.yaw, ego_transform.rotation.yaw) > self._ambient_alignment_max_deg:
                continue
            candidates.append((distance_to_ego, spawn_point))

        candidates.sort(key=lambda item: item[0])

        for index, (_, spawn_point) in enumerate(candidates[:self._ambient_vehicle_count]):
            try:
                actor = CarlaDataProvider.request_new_actor(
                    allowed_ambient_models[index % len(allowed_ambient_models)],
                    spawn_point,
                    rolename="ambient_{:03d}".format(index),
                    autopilot=True,
                    random_location=False,
                    tick=False,
                )
            except Exception:
                actor = None

            if actor is None:
                continue

            self._background_actors.append(actor)
            self._configure_vehicle_physics(actor)
            self.other_actors.append(actor)
            try:
                if hasattr(tm, "distance_to_leading_vehicle"):
                    tm.distance_to_leading_vehicle(actor, 2.0)
                if hasattr(tm, "vehicle_percentage_speed_difference"):
                    tm.vehicle_percentage_speed_difference(actor, 0.0)
                if hasattr(tm, "ignore_lights_percentage"):
                    tm.ignore_lights_percentage(actor, 0.0)
                if hasattr(tm, "ignore_signs_percentage"):
                    tm.ignore_signs_percentage(actor, 0.0)
            except RuntimeError:
                pass

    @staticmethod
    def _speed_mps(actor: Optional[carla.Vehicle]) -> float:
        if actor is None or not actor.is_alive:
            return 0.0
        vel = actor.get_velocity()
        return math.sqrt((vel.x ** 2) + (vel.y ** 2) + (vel.z ** 2))

    def _apply_speed_hold(
        self,
        actor: Optional[carla.Vehicle],
        control: carla.VehicleControl,
        target_speed_mps: float,
        throttle_gain: float = 0.15,
        brake_gain: float = 0.25,
        max_brake: float = 0.8,
    ):
        """
        Convert target speed to throttle/brake.
        Uses a lightweight P controller to avoid hard throttle-only behavior.
        """
        target = max(0.0, float(target_speed_mps))
        current = self._speed_mps(actor)
        error = target - current
        deadband = 0.3

        if error > deadband:
            control.throttle = min(1.0, error * throttle_gain)
            control.brake = 0.0
        elif error < -deadband:
            control.throttle = 0.0
            control.brake = min(max_brake, (-error) * brake_gain)
        else:
            control.throttle = 0.0
            control.brake = 0.0

    def _build_lane_follow_control(self, actor, target_speed_mps, max_steer):
        """Simple lane following + target speed hold used for support traffic."""
        control = carla.VehicleControl()
        steer_cmd = 0.0
        if actor is not None and actor.is_alive:
            map_obj = CarlaDataProvider.get_map()
            waypoint = None if map_obj is None else map_obj.get_waypoint(
                actor.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is not None:
                actor_transform = actor.get_transform()
                wp_transform = waypoint.transform
                right = wp_transform.rotation.get_right_vector()
                delta = actor_transform.location - wp_transform.location
                lateral_error = (delta.x * right.x) + (delta.y * right.y)

                heading_delta = (actor_transform.rotation.yaw - wp_transform.rotation.yaw + 180.0) % 360.0 - 180.0
                heading_error = heading_delta / 90.0

                steer_cmd = (
                    (-lateral_error * self._ego_lane_follow_gain) +
                    (-heading_error * self._ego_heading_gain)
                )

        control.steer = max(-max_steer, min(max_steer, steer_cmd))
        self._apply_speed_hold(actor, control, target_speed_mps, throttle_gain=0.22, brake_gain=0.35, max_brake=0.5)
        return control

    def _apply_gap_traffic_controls(self):
        """Keep the lead vehicles traveling steadily ahead."""
        if self._ego_gap_lead is not None and self._ego_gap_lead.is_alive:
            self._ego_gap_lead.apply_control(
                    self._build_lane_follow_control(
                        self._ego_gap_lead, self._ego_lead_vehicle_speed, self._gap_vehicle_max_steer
                    )
                )
        if self._adjacent_gap_lead is not None and self._adjacent_gap_lead.is_alive:
            self._adjacent_gap_lead.apply_control(
                self._build_lane_follow_control(
                    self._adjacent_gap_lead, self._adjacent_lead_vehicle_speed, self._gap_vehicle_max_steer
                )
            )

    def _compute_adversary_rear_gap(self):
        """Compute ego-front to adversary-rear gap along the ego forward axis."""
        ego = self.ego_vehicles[0]
        adversary = self._adversary
        if ego is None or adversary is None or not ego.is_alive or not adversary.is_alive:
            return float("inf")

        ego_transform = ego.get_transform()
        ego_location = ego_transform.location
        adversary_location = adversary.get_transform().location
        forward = ego_transform.rotation.get_forward_vector()
        delta = adversary_location - ego_location
        rel_along = (delta.x * forward.x) + (delta.y * forward.y) + (delta.z * forward.z)

        ego_front = ego.bounding_box.extent.x if ego.bounding_box else 2.0
        adversary_rear = adversary.bounding_box.extent.x if adversary.bounding_box else 2.0
        return rel_along - ego_front - adversary_rear

    def _build_adversary_control(self, elapsed, merge_elapsed):
        _ = elapsed
        control = carla.VehicleControl()
        target_speed = self._adversary_cruise_speed

        if merge_elapsed is None:
            control.steer = 0.0
            target_speed = self._adversary_cruise_speed

        elif merge_elapsed < self._cut_duration:
            control.steer = clamp_steer(self._adversary_cut_steer)
            target_speed = self._adversary_cut_speed

        elif merge_elapsed < (self._cut_duration + self._countersteer_duration):
            counter_elapsed = merge_elapsed - self._cut_duration
            cut_steer = clamp_steer(self._adversary_cut_steer)
            base_counter = clamp_steer(self._adversary_countersteer)

            if cut_steer == 0.0:
                cut_steer = 1.0
            if base_counter == 0.0 or (base_counter * cut_steer) > 0.0:
                base_counter = -1.0 if cut_steer > 0.0 else 1.0

            flick_delay = min(0.12, self._countersteer_duration * 0.35)
            flick_gain = 0.65 * self._recovery_snap_gain
            steer = base_counter
            if counter_elapsed > flick_delay:
                flick_elapsed = counter_elapsed - flick_delay
                flick_period = max(0.08, self._recovery_oscillation_period * 0.45)
                flick_wave = math.sin((2.0 * math.pi * flick_elapsed) / flick_period)
                steer += flick_gain * math.tanh(self._recovery_snap_sharpness * flick_wave)

            control.steer = clamp_steer(steer)
            target_speed = self._adversary_countersteer_speed

        else:
            recovery_elapsed = merge_elapsed - self._cut_duration - self._countersteer_duration
            oscillation_period = max(0.1, self._recovery_oscillation_period)
            panic_multiplier = 1.0
            if recovery_elapsed < self._recovery_panic_window:
                panic_multiplier = self._recovery_panic_multiplier

            decay = math.exp(-max(0.0, recovery_elapsed) * self._recovery_decay)
            envelope = max(self._recovery_floor_gain, decay)
            phase = (2.0 * math.pi * recovery_elapsed) / oscillation_period
            smooth_oscillation = math.sin(phase)
            snap_component = math.tanh(self._recovery_snap_sharpness * smooth_oscillation)

            struggle_scale = 1.0 if recovery_elapsed < self._settle_duration else 0.75
            steer = self._adversary_settle_steer + (
                self._recovery_steer_gain * envelope * smooth_oscillation * panic_multiplier * struggle_scale
            ) + (
                self._recovery_snap_gain * envelope * snap_component * panic_multiplier * struggle_scale
            )

            control.steer = max(-1.0, min(1.0, steer))
            control.hand_brake = False
            target_speed = self._adversary_settle_speed

        self._apply_speed_hold(
            self._adversary, control, target_speed,
            throttle_gain=0.18, brake_gain=0.22, max_brake=0.65
        )

        return control

    def _create_behavior(self):
        if self._adversary is None:
            return py_trees.behaviours.Failure(name="AdversaryNotSpawned")

        sequence = py_trees.composites.Sequence(name="HighwayCutAcrossCollisionSequence")
        sequence.add_child(py_trees.behaviours.Success(name="StartImmediately"))
        maneuver = py_trees.composites.Parallel(
            "CutAcrossManeuverWithEgoSpeed",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        maneuver.add_child(RunHighwayPitManeuver(self))
        if self._ego_min_speed_kmh > 0.1:
            maneuver.add_child(_EgoSpeedKeeper(
                self.ego_vehicles[0],
                min_speed_kmh=self._ego_min_speed_kmh,
                throttle=self._ego_force_throttle,
                name="EgoSpeedKeeper",
            ))
        sequence.add_child(maneuver)
        if not self._end_on_collision:
            sequence.add_child(PacedIdle(self, self._post_impact_delay, name="PostImpactDelay"))
            sequence.add_child(ApplyVehicleControlOnce(
                self.ego_vehicles[0],
                throttle=0.0,
                brake=0.9,
                steer=0.0,
                hand_brake=False,
                name="BrakeEgoAfterImpact",
            ))
            sequence.add_child(ApplyVehicleControlOnce(
                self._adversary,
                throttle=0.0,
                brake=0.0,
                steer=0.0,
                hand_brake=True,
                name="LockCarAfterImpact",
            ))
            sequence.add_child(PacedIdle(self, self._post_impact_hold, name="HoldCrashScene"))

        if self._end_on_collision:
            collision_sequence = py_trees.composites.Sequence(
                "CollisionThenHoldTerminate", memory=True,
            )
            collision_maneuver = py_trees.composites.Parallel(
                "CollisionManeuverWithEgoSpeed",
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
            )
            collision_maneuver.add_child(RunHighwayPitManeuver(self))
            if self._ego_min_speed_kmh > 0.1:
                collision_maneuver.add_child(_EgoSpeedKeeper(
                    self.ego_vehicles[0],
                    min_speed_kmh=self._ego_min_speed_kmh,
                    throttle=self._ego_force_throttle,
                    name="EgoSpeedKeeper",
                ))
            collision_sequence.add_child(collision_maneuver)
            collision_sequence.add_child(HoldScenarioActorsOnImpact(
                self,
                [self._adversary, self._ego_gap_lead, self._adjacent_gap_lead],
                name="HoldScenarioActorsOnImpact",
            ))
            return collision_sequence

        return sequence

    def _setup_scenario_trigger(self, config):
        """Start this scenario immediately."""
        _ = config
        return None

    def _create_test_criteria(self):
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        try:
            client = CarlaDataProvider.get_client()
            if client is not None:
                client.stop_recorder()
        except Exception:
            pass

        try:
            self.remove_all_actors()
        except Exception:
            pass
