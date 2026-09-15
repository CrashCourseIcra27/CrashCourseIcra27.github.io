"""Decelerating cut in with blocked escape (Town12).

A vehicle cuts in ahead and decelerates while the adjacent escape lane is blocked.
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
from srunner.scenariomanager.scenarioatomics.custom_atomics_decelerating_cut_in_with_blocked_escape_town12 import (
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
    def __init__(self, ego, min_speed_kmh=30.0, name="HoldUntilEgoSpeed"):
        super().__init__(name)
        self._ego = ego
        self._min_speed = float(min_speed_kmh) / 3.6

    def update(self):
        if _speed_mps(self._ego) >= self._min_speed:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class _ForcedLateralMove(py_trees.behaviour.Behaviour):
    """Drive forward while drifting laterally (exit or cutback)."""

    def __init__(self, actor, forward_speed_ms, lateral_speed_ms,
                 duration, lateral_sign=1.0, name="ForcedLateralMove"):
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
        # Slight steering in the drift direction for visual realism
        steer = 0.2 * self._lateral_sign
        self._actor.apply_control(carla.VehicleControl(
            throttle=0.5, brake=0.0, steer=steer))
        return py_trees.common.Status.RUNNING


class _CruiseForward(py_trees.behaviour.Behaviour):
    """Cruise forward at constant speed."""

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


class _GapTrafficRunner(py_trees.behaviour.Behaviour):
    def __init__(self, scenario, name="GapTrafficRunner"):
        super().__init__(name)
        self._scenario = scenario

    def update(self):
        self._scenario._tick_gap_traffic()
        return py_trees.common.Status.RUNNING


class DeceleratingCutInWithBlockedEscapeTown12(BasicScenario):
    """
    Vehicle ahead takes the exit (moves right), then abruptly cuts back
    left across the gore into ego's lane — an aborted-exit collision.
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
        self._cutter_spawn_ahead = self._get_param(op, "cutter_spawn_ahead_m", 30.0)
        self._cutter_cruise_speed_kmh = self._get_param(op, "cutter_cruise_speed_kmh", 65.0)
        self._launch_ego_speed_kmh = self._get_param(op, "launch_ego_speed_kmh", 50.0)
        self._exit_lateral_speed_ms = self._get_param(op, "exit_lateral_speed_ms", 3.0)
        self._exit_duration = self._get_param(op, "exit_duration_s", 1.3)
        self._pause_duration = self._get_param(op, "pause_on_shoulder_s", 1.5)
        self._pause_forward_factor = self._get_param(op, "pause_forward_factor", 0.75)
        self._cutback_lateral_speed_ms = self._get_param(op, "cutback_lateral_speed_ms", 5.5)
        self._cutback_duration = self._get_param(op, "cutback_duration_s", 1.2)
        self._pre_cutback_cruise_duration = self._get_param(op, "pre_cutback_cruise_duration_s", 4.0)
        self._cutback_forward_factor = self._get_param(op, "cutback_forward_factor", 0.10)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 80.0)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 65.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.12)
        self._aftermath_duration = self._get_param(op, "aftermath_duration_s", 5.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._cutter = None
        self._ego_gap_lead = None
        self._adjacent_gap_lead = None
        self._cutter_spawn_wp = None
        # Sign: +1 = drift right (exit right), -1 = drift left
        self._exit_sign = 1.0  # exit to the right
        self._cutback_sign = -1.0  # cut back to the left
        self._ego_lead_vehicle_model = str(self._get_param(
            op, "ego_lead_vehicle_model", "vehicle.carlamotors.european_hgv", cast=str))
        self._adjacent_lead_vehicle_model = str(self._get_param(
            op, "adjacent_lead_vehicle_model", "vehicle.tesla.model3", cast=str))
        self._ego_lead_vehicle_distance = self._get_param(op, "ego_lead_vehicle_distance_m", 22.0)
        self._adjacent_lead_vehicle_distance = self._get_param(op, "adjacent_lead_vehicle_distance_m", 14.0)
        self._ego_lead_vehicle_speed = self._get_param(op, "ego_lead_vehicle_speed_kmh", 56.0) / 3.6
        self._adjacent_lead_vehicle_speed = self._get_param(op, "adjacent_lead_vehicle_speed_kmh", 42.0) / 3.6
        self._gap_vehicle_max_steer = self._get_param(op, "gap_vehicle_max_steer", 0.20)

        super().__init__(
            name="DeceleratingCutInWithBlockedEscapeTown12",
            ego_vehicles=ego_vehicles, config=config, world=world,
            debug_mode=debug_mode, terminate_on_failure=True,
            criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving)

        # Prefer a visible adjacent-lane cut-in on the validated highway segment.
        right_lane_wp = ego_wp.get_left_lane()
        lateral_offset = -3.6
        self._cutback_sign = 1.0
        if right_lane_wp is None or right_lane_wp.lane_type != carla.LaneType.Driving:
            right_lane_wp = ego_wp.get_right_lane()
            lateral_offset = 3.6
            self._cutback_sign = -1.0
        if right_lane_wp is None or right_lane_wp.lane_type != carla.LaneType.Driving:
            right_lane_wp = ego_wp

        self._cutter_spawn_wp = self._walk_waypoint(
            right_lane_wp, self._cutter_spawn_ahead, forward=True) or right_lane_wp

        cutter_location = carla.Location(
            self._cutter_spawn_wp.transform.location.x,
            self._cutter_spawn_wp.transform.location.y,
            self._cutter_spawn_wp.transform.location.z + 0.5,
        )
        if self._cutter_spawn_wp == ego_wp:
            right_vec = ego_wp.transform.get_right_vector()
            cutter_location += carla.Location(
                x=right_vec.x * lateral_offset,
                y=right_vec.y * lateral_offset,
                z=0.0,
            )
        transform = carla.Transform(cutter_location, self._cutter_spawn_wp.transform.rotation)

        models = ["vehicle.dodge.charger_2020",
                  "vehicle.lincoln.mkz_2020",
                  "vehicle.tesla.model3",
                  "vehicle.audi.a2"]
        self._cutter = None
        for model in models:
            actor = CarlaDataProvider.request_new_actor(
                model, transform, rolename="gore_cutter")
            if actor is not None:
                self._cutter = actor
                break
        if self._cutter is None:
            raise RuntimeError("[DeceleratingCutInWithBlockedEscapeTown12] Failed to spawn cutter")
        self._cutter.set_simulate_physics(True)
        self.other_actors.append(self._cutter)

        self._spawn_gap_traffic(ego_wp)

        print(
            f"\n[DeceleratingCutInWithBlockedEscapeTown12] Spawn\n"
            f"  Ego: road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  Cutter: right lane, ahead={self._cutter_spawn_ahead:.0f}m "
            f"speed={self._cutter_cruise_speed_kmh:.0f} km/h\n"
            f"  Cutback sign={self._cutback_sign:+.0f} "
            f"(lateral into ego lane)\n"
            f"  Cutback: lat={self._cutback_lateral_speed_ms:.1f} m/s "
            f"dur={self._cutback_duration:.1f}s\n",
            flush=True)

    def _build_forward_transform(self, waypoint, distance):
        target_wp = self._walk_waypoint(waypoint, distance, forward=True) or waypoint
        return carla.Transform(
            carla.Location(
                target_wp.transform.location.x,
                target_wp.transform.location.y,
                target_wp.transform.location.z + 0.5,
            ),
            target_wp.transform.rotation,
        )

    def _spawn_gap_traffic(self, ego_wp):
        ego_lead_transform = self._build_forward_transform(ego_wp, self._ego_lead_vehicle_distance)
        adjacent_transform = None
        adjacent_wp = ego_wp.get_left_lane()
        if adjacent_wp is not None and adjacent_wp.lane_type == carla.LaneType.Driving:
            adjacent_transform = self._build_forward_transform(adjacent_wp, self._adjacent_lead_vehicle_distance)

        if ego_lead_transform is not None:
            self._ego_gap_lead = CarlaDataProvider.request_new_actor(
                self._ego_lead_vehicle_model,
                ego_lead_transform,
                rolename="gore_truck_lead",
                autopilot=False,
                color="180,180,180",
            )
            if self._ego_gap_lead is not None:
                self._ego_gap_lead.set_simulate_physics(True)
                self.other_actors.append(self._ego_gap_lead)

        if adjacent_transform is not None:
            self._adjacent_gap_lead = CarlaDataProvider.request_new_actor(
                self._adjacent_lead_vehicle_model,
                adjacent_transform,
                rolename="gore_adjacent_blocker",
                autopilot=False,
                color="145,145,145",
            )
            if self._adjacent_gap_lead is not None:
                self._adjacent_gap_lead.set_simulate_physics(True)
                self.other_actors.append(self._adjacent_gap_lead)

    def _steady_lane_control(self, actor, target_speed):
        if actor is None or not actor.is_alive:
            return carla.VehicleControl()
        current = _speed_mps(actor)
        delta = target_speed - current
        throttle = max(0.0, min(0.75, 0.35 + delta * 0.12))
        brake = max(0.0, min(0.4, -delta * 0.10)) if delta < -0.5 else 0.0
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0)

    def _launch_cutter(self):
        if self._cutter is None or not self._cutter.is_alive:
            return
        t = self._cutter.get_transform()
        fwd = t.get_forward_vector()
        speed = self._cutter_cruise_speed_kmh / 3.6
        self._cutter.set_target_velocity(
            carla.Vector3D(fwd.x * speed, fwd.y * speed, 0.0))

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego, self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateHighwayGoreCutoff")

    def _create_behavior(self):
        P = py_trees.common.ParallelPolicy
        cruise_speed_ms = self._cutter_cruise_speed_kmh / 3.6
        root = py_trees.composites.Sequence("GoreCutoff_Root")

        # Phase 0 — wait for ego to reach speed (cutter stays stationary)
        phase0 = py_trees.composites.Parallel("Phase0_WaitEgo", policy=P.SUCCESS_ON_ONE)
        phase0.add_child(_HoldUntilEgoSpeed(
            self._ego, min_speed_kmh=self._launch_ego_speed_kmh))
        phase0.add_child(TimeOut(10.0, name="Phase0_Timeout"))
        root.add_child(phase0)

        # Launch cutter AFTER ego is up to speed
        root.add_child(_OneShot(self._launch_cutter, name="LaunchCutter"))

        # Phase 1 — cutter cruises in right lane alongside ego
        phase1 = py_trees.composites.Parallel("Phase1_Cruise", policy=P.SUCCESS_ON_ONE)
        phase1.add_child(_CruiseForward(
            self._cutter, forward_speed_ms=cruise_speed_ms,
            duration=self._pre_cutback_cruise_duration, name="CruiseInRightLane"))
        phase1.add_child(TimeOut(
            self._pre_cutback_cruise_duration + 1.0, name="Phase1_Timeout"))
        root.add_child(phase1)

        # Phase 2 — cutter cuts HARD LEFT into ego's lane (aborted-exit swerve)
        phase2 = py_trees.composites.Parallel("Phase2_Cutback", policy=P.SUCCESS_ON_ONE)
        phase2.add_child(_ForcedLateralMove(
            self._cutter,
            forward_speed_ms=cruise_speed_ms * self._cutback_forward_factor,
            lateral_speed_ms=self._cutback_lateral_speed_ms,
            duration=self._cutback_duration,
            lateral_sign=self._cutback_sign,
            name="CutbackIntoEgoLane"))
        phase2.add_child(TimeOut(self._cutback_duration + 1.0, name="Phase2_Timeout"))
        root.add_child(phase2)

        # Phase 3 — aftermath cruise
        root.add_child(_CruiseForward(
            self._cutter, forward_speed_ms=cruise_speed_ms * 0.6,
            duration=self._aftermath_duration, name="Aftermath"))

        # Cleanup
        cleanup = py_trees.composites.Sequence("Cleanup")
        cleanup.add_child(ActorDestroy(self._cutter, name="DestroyCutter"))
        root.add_child(cleanup)

        # Parallel outer wrapper
        lanes = [self._trigger_wp]
        adj = self._trigger_wp.get_right_lane()
        if adj is not None and adj.lane_type == carla.LaneType.Driving:
            lanes.append(adj)

        outer = py_trees.composites.Parallel("GoreCutoff_Outer", policy=P.SUCCESS_ON_ONE)
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego, self._cutter],
            lane_waypoints=lanes,
            center_location=self._trigger_wp.transform.location,
            radius=500.0, interval=1.0, name="ScenarioLaneCleaner"))
        outer.add_child(PeriodicStatusLogger(
            self._ego, self._cutter, None,
            interval=1.0, name="StatusLogger"))
        outer.add_child(_GapTrafficRunner(self))
        outer.add_child(EgoSpeedGovernor(
            self._ego, speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake, name="EgoSpeedGovernor"))
        outer.add_child(_EgoMinSpeedForcer(
            self._ego, min_speed_kmh=self._ego_min_speed_kmh,
            name="EgoMinSpeedForcer"))
        outer.add_child(StopActorsOnEgoCollision(
            self._ego, actors_to_stop=[self._cutter, self._ego_gap_lead, self._adjacent_gap_lead],
            name="StopOnCollision"))
        return outer

    def _tick_gap_traffic(self):
        if self._ego_gap_lead is not None and self._ego_gap_lead.is_alive:
            self._ego_gap_lead.apply_control(
                self._steady_lane_control(self._ego_gap_lead, self._ego_lead_vehicle_speed))
        if self._adjacent_gap_lead is not None and self._adjacent_gap_lead.is_alive:
            self._adjacent_gap_lead.apply_control(
                self._steady_lane_control(self._adjacent_gap_lead, self._adjacent_lead_vehicle_speed))

    def _create_test_criteria(self):
        return [CollisionTest(self._ego)]

    def __del__(self):
        self.remove_all_actors()
