"""Unresponsive driver lateral drift (Town12).

A vehicle whose driver has become unresponsive drifts laterally across the lane line.
"""

import math

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
)
from srunner.scenariomanager.scenarioatomics.custom_atomics_unresponsive_driver_lateral_drift_town12 import (
    EgoSpeedGovernor,
    LaneCleaner,
    PeriodicStatusLogger,
    StopActorsOnEgoCollision,
)
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenarios.basic_scenario import BasicScenario


def _speed_mps(actor):
    if actor is None or not actor.is_alive:
        return 0.0
    v = actor.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


class _OneShot(py_trees.behaviour.Behaviour):
    def __init__(self, callback, name="OneShot"):
        super().__init__(name)
        self._callback = callback
        self._done = False

    def update(self):
        if not self._done:
            self._callback()
            self._done = True
        return py_trees.common.Status.SUCCESS


class _HoldUntilEgoSpeed(py_trees.behaviour.Behaviour):
    """Wait until ego reaches a minimum speed."""

    def __init__(self, ego, min_speed_kmh=30.0, name="HoldUntilEgoSpeed"):
        super().__init__(name)
        self._ego = ego
        self._min_speed = float(min_speed_kmh) / 3.6

    def update(self):
        if _speed_mps(self._ego) >= self._min_speed:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class _SteadyDrift(py_trees.behaviour.Behaviour):
    """Drive forward at constant speed while drifting laterally at a constant rate."""

    def __init__(self, actor, forward_speed_ms, lateral_speed_ms=0.5,
                 duration=6.0, lateral_sign=1.0, name="SteadyDrift"):
        super().__init__(name)
        self._actor = actor
        self._forward_speed = float(forward_speed_ms)
        self._lateral_speed = float(lateral_speed_ms)
        self._duration = float(duration)
        self._lateral_sign = float(lateral_sign)
        self._start_time = None
        self._forward = None
        self._right = None

    def initialise(self):
        self._start_time = GameTime.get_time()
        if self._actor is not None and self._actor.is_alive:
            t = self._actor.get_transform()
            self._forward = t.get_forward_vector()
            self._right = t.get_right_vector()

    def update(self):
        if self._actor is None or not self._actor.is_alive:
            return py_trees.common.Status.SUCCESS
        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0
        if elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS

        fwd = self._forward
        right = self._right
        lat = self._lateral_speed * self._lateral_sign
        vx = self._forward_speed * fwd.x + lat * right.x
        vy = self._forward_speed * fwd.y + lat * right.y
        self._actor.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
        # No steering correction — the driver is unresponsive
        self._actor.apply_control(carla.VehicleControl(
            throttle=0.4, brake=0.0, steer=0.0))
        return py_trees.common.Status.RUNNING


class _CruiseForward(py_trees.behaviour.Behaviour):
    """Cruise forward at constant speed with no lateral movement."""

    def __init__(self, actor, forward_speed_ms, duration=10.0,
                 name="CruiseForward"):
        super().__init__(name)
        self._actor = actor
        self._forward_speed = float(forward_speed_ms)
        self._duration = float(duration)
        self._start_time = None
        self._forward = None

    def initialise(self):
        self._start_time = GameTime.get_time()
        if self._actor is not None and self._actor.is_alive:
            t = self._actor.get_transform()
            self._forward = t.get_forward_vector()

    def update(self):
        if self._actor is None or not self._actor.is_alive:
            return py_trees.common.Status.SUCCESS
        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0
        if elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS

        fwd = self._forward
        vx = self._forward_speed * fwd.x
        vy = self._forward_speed * fwd.y
        self._actor.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
        self._actor.apply_control(carla.VehicleControl(
            throttle=0.4, brake=0.0, steer=0.0))
        return py_trees.common.Status.RUNNING


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    """Prevent ego from braking below a minimum speed."""

    def __init__(self, ego, min_speed_kmh=60.0, throttle=1.0,
                 name="EgoMinSpeedForcer"):
        super().__init__(name)
        self._ego = ego
        self._min_mps = float(min_speed_kmh) / 3.6
        self._throttle = float(throttle)

    def update(self):
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.RUNNING
        v = self._ego.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        if speed < self._min_mps:
            ctrl = self._ego.get_control()
            ctrl.brake = 0.0
            ctrl.throttle = self._throttle
            self._ego.apply_control(ctrl)
        return py_trees.common.Status.RUNNING


class UnresponsiveDriverLateralDriftTown12(BasicScenario):
    """
    Adjacent vehicle's driver becomes unresponsive and slowly drifts
    into ego's lane, producing a side-to-side impact.
    """
    timeout = 180

    @staticmethod
    def _get_param(op, name, default, cast=float):
        if not op or name not in op:
            return default
        raw = op[name]
        try:
            if isinstance(raw, dict):
                return cast(raw.get("value", default))
            return cast(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _walk_waypoint(waypoint, distance, forward=True):
        remaining = abs(float(distance))
        cursor = waypoint
        while remaining > 0.2 and cursor is not None:
            step = min(8.0, remaining)
            candidates = cursor.next(step) if forward else cursor.previous(step)
            if not candidates:
                break
            cursor = candidates[0]
            remaining -= step
        return cursor

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._ego = ego_vehicles[0]

        op = config.other_parameters if hasattr(config, "other_parameters") else {}
        self._activation_distance = self._get_param(op, "activation_distance_m", 120.0)
        self._drifter_spawn_ahead = self._get_param(op, "drifter_spawn_ahead_m", 6.0)
        self._drifter_cruise_speed_kmh = self._get_param(op, "drifter_cruise_speed_kmh", 72.0)
        self._drift_lateral_speed_ms = self._get_param(op, "drift_lateral_speed_ms", 0.55)
        self._drift_duration = self._get_param(op, "drift_duration_s", 7.0)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 78.0)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 60.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.12)
        self._aftermath_duration = self._get_param(op, "aftermath_duration_s", 5.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._drifter = None
        self._drifter_spawn_wp = None
        self._lateral_sign = 1.0  # drift direction toward ego; computed at spawn

        super().__init__(
            name="UnresponsiveDriverLateralDriftTown12",
            ego_vehicles=ego_vehicles, config=config, world=world,
            debug_mode=debug_mode, terminate_on_failure=True,
            criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving)

        # Find adjacent lane for drifter spawn
        spawn_lane = ego_wp.get_left_lane()
        if spawn_lane is None or spawn_lane.lane_type != carla.LaneType.Driving:
            spawn_lane = ego_wp.get_right_lane()
        if spawn_lane is None or spawn_lane.lane_type != carla.LaneType.Driving:
            raise RuntimeError("[UnresponsiveDriverLateralDriftTown12] No adjacent driving lane")

        # Spawn slightly ahead of ego
        self._drifter_spawn_wp = self._walk_waypoint(
            spawn_lane, self._drifter_spawn_ahead, forward=True) or spawn_lane

        transform = carla.Transform(
            carla.Location(
                self._drifter_spawn_wp.transform.location.x,
                self._drifter_spawn_wp.transform.location.y,
                self._drifter_spawn_wp.transform.location.z + 0.5),
            self._drifter_spawn_wp.transform.rotation)

        # Use a normal sedan
        models = ["vehicle.lincoln.mkz_2020",
                  "vehicle.dodge.charger_2020",
                  "vehicle.tesla.model3",
                  "vehicle.audi.a2"]
        self._drifter = None
        for model in models:
            actor = CarlaDataProvider.request_new_actor(
                model, transform, rolename="drifter")
            if actor is not None:
                self._drifter = actor
                break
        if self._drifter is None:
            raise RuntimeError("[UnresponsiveDriverLateralDriftTown12] Failed to spawn drifter")
        self._drifter.set_simulate_physics(True)
        self.other_actors.append(self._drifter)

        # Determine which direction is toward the ego
        drifter_right = self._drifter_spawn_wp.transform.get_right_vector()
        ego_loc = ego_wp.transform.location
        drifter_loc = self._drifter_spawn_wp.transform.location
        dx = ego_loc.x - drifter_loc.x
        dy = ego_loc.y - drifter_loc.y
        # If ego is to the right of drifter, lateral_sign = +1 (drift right)
        self._lateral_sign = 1.0 if (dx * drifter_right.x + dy * drifter_right.y) >= 0.0 else -1.0

        print(
            f"\n[UnresponsiveDriverLateralDriftTown12] Spawn\n"
            f"  Ego: road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  Drifter: road={self._drifter_spawn_wp.road_id} "
            f"lane={self._drifter_spawn_wp.lane_id} "
            f"ahead={self._drifter_spawn_ahead:.1f}m\n"
            f"  Drift: lateral={self._drift_lateral_speed_ms:.2f} m/s "
            f"duration={self._drift_duration:.1f}s "
            f"sign={self._lateral_sign:+.0f}\n"
            f"  Cruise speed: {self._drifter_cruise_speed_kmh:.0f} km/h\n",
            flush=True)

    def _launch_drifter(self):
        if self._drifter is None or not self._drifter.is_alive:
            return
        t = self._drifter.get_transform()
        fwd = t.get_forward_vector()
        speed = self._drifter_cruise_speed_kmh / 3.6
        self._drifter.set_target_velocity(
            carla.Vector3D(fwd.x * speed, fwd.y * speed, 0.0))

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego, self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateDriverMedicalDrift")

    def _create_behavior(self):
        P = py_trees.common.ParallelPolicy
        root = py_trees.composites.Sequence("MedicalDrift_Root")

        # Phase 0 — wait for ego to reach speed
        phase0 = py_trees.composites.Parallel("Phase0_WaitEgo", policy=P.SUCCESS_ON_ONE)
        phase0.add_child(_HoldUntilEgoSpeed(self._ego, min_speed_kmh=40.0))
        phase0.add_child(TimeOut(12.0, name="Phase0_Timeout"))
        root.add_child(phase0)

        # Launch drifter at cruise speed
        root.add_child(_OneShot(self._launch_drifter, name="LaunchDrifter"))

        # Phase 1 — cruise alongside ego for a few seconds (normal-looking)
        cruise_speed_ms = self._drifter_cruise_speed_kmh / 3.6
        phase1 = py_trees.composites.Parallel("Phase1_Cruise", policy=P.SUCCESS_ON_ONE)
        phase1.add_child(_CruiseForward(
            self._drifter, forward_speed_ms=cruise_speed_ms,
            duration=3.0, name="NormalCruise"))
        phase1.add_child(TimeOut(4.0, name="Phase1_Timeout"))
        root.add_child(phase1)

        # Phase 2 — slow, persistent lateral drift into ego lane (no correction)
        phase2 = py_trees.composites.Parallel("Phase2_Drift", policy=P.SUCCESS_ON_ONE)
        phase2.add_child(_SteadyDrift(
            self._drifter,
            forward_speed_ms=cruise_speed_ms,
            lateral_speed_ms=self._drift_lateral_speed_ms,
            duration=self._drift_duration,
            lateral_sign=self._lateral_sign,
            name="MedicalDrift"))
        phase2.add_child(TimeOut(self._drift_duration + 1.0, name="Phase2_Timeout"))
        root.add_child(phase2)

        # Phase 3 — aftermath (continue drifting to show driver is unresponsive)
        root.add_child(_SteadyDrift(
            self._drifter,
            forward_speed_ms=cruise_speed_ms * 0.8,
            lateral_speed_ms=self._drift_lateral_speed_ms * 0.5,
            duration=self._aftermath_duration,
            lateral_sign=self._lateral_sign,
            name="ContinuedDrift"))

        # Cleanup
        cleanup = py_trees.composites.Sequence("Cleanup")
        cleanup.add_child(ActorDestroy(self._drifter, name="DestroyDrifter"))
        root.add_child(cleanup)

        # Parallel outer wrapper
        lanes = [self._trigger_wp]
        adj = self._trigger_wp.get_left_lane()
        if adj is not None and adj.lane_type == carla.LaneType.Driving:
            lanes.append(adj)

        outer = py_trees.composites.Parallel("MedicalDrift_Outer", policy=P.SUCCESS_ON_ONE)
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego, self._drifter],
            lane_waypoints=lanes,
            center_location=self._trigger_wp.transform.location,
            radius=500.0, interval=1.0, name="ScenarioLaneCleaner"))
        outer.add_child(PeriodicStatusLogger(
            self._ego, self._drifter, None,
            interval=1.0, name="StatusLogger"))
        outer.add_child(EgoSpeedGovernor(
            self._ego, speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake, max_brake=self._ego_cap_brake,
            name="EgoSpeedGovernor"))
        outer.add_child(_EgoMinSpeedForcer(
            self._ego, min_speed_kmh=self._ego_min_speed_kmh,
            name="EgoMinSpeedForcer"))
        outer.add_child(StopActorsOnEgoCollision(
            self._ego, actors_to_stop=[self._drifter],
            name="StopOnCollision"))
        return outer

    def _create_test_criteria(self):
        return [CollisionTest(self._ego)]

    def __del__(self):
        self.remove_all_actors()
