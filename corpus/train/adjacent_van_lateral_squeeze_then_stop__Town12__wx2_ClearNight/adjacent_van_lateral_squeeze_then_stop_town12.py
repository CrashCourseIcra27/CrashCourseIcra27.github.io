"""Adjacent van lateral squeeze then stop (Town12).

A van in the adjacent lane pushes laterally into the ego's lane and then comes to rest there.
"""
from __future__ import print_function

import math
import json
import operator
import os

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy, AtomicBehavior, ScenarioTimeout, WaypointFollower)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest, CollisionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    AtomicCondition, DriveDistance, TriggerVelocity)
from srunner.scenarios.basic_scenario import BasicScenario

ROAD_ID = 794

# Push-target blend: fraction of the lateral gap resolved toward the ego's
# own lane centreline (vs the pursuer's own lane centreline). 0.9 is deep
# enough to overlap the ego's body width by construction (see module
# docstring); the remaining 0.1 keeps this a graze, not 's full cut-in.
_PUSH_EGO_BIAS = 0.9

_REST_STRADDLE_BIAS = 0.65


def _value(config, name, default, low, high):
    result = float(config.other_parameters.get(name, {}).get("value", default))
    if result < low or result > high:
        raise ValueError("{} outside [{}, {}]".format(name, low, high))
    return result


def _advance(waypoint, distance_m):
    candidates = [item for item in waypoint.next(distance_m)
                  if item.road_id == waypoint.road_id
                  and item.lane_id == waypoint.lane_id
                  and item.lane_type == carla.LaneType.Driving
                  and not item.is_junction]
    if not candidates:
        raise ValueError("cannot advance {} m on road {}".format(distance_m, ROAD_ID))
    return candidates[0]


def _outboard_driving(waypoint, hops=1):
    candidate = waypoint
    for _ in range(hops):
        candidate = candidate.get_right_lane()
        if (candidate is None or candidate.lane_type != carla.LaneType.Driving
                or candidate.road_id != waypoint.road_id
                or candidate.lane_id * waypoint.lane_id <= 0):
            raise ValueError("requires {} outboard (right) driving lane(s)".format(hops))
    return candidate


def _inboard_driving(waypoint):
    candidate = waypoint.get_left_lane()
    if (candidate is None or candidate.lane_type != carla.LaneType.Driving
            or candidate.road_id != waypoint.road_id
            or candidate.lane_id * waypoint.lane_id <= 0):
        raise ValueError("requires an inboard (left) escape lane -- checked at design time "
                          "to prove the acceptance criterion before acceptance")
    return candidate


def _lane_ahead(actor_or_ego_wp, distance_m):
    """Best-effort projection forward along a waypoint's own lane; relaxed
    (lane_id match only, not road_id) so a road-boundary crossing near
    commit time does not raise -- mirrors /'s `_ego_relative_target`."""
    candidates = [item for item in actor_or_ego_wp.next(distance_m)
                  if item.lane_id == actor_or_ego_wp.lane_id
                  and item.lane_type == carla.LaneType.Driving
                  and not item.is_junction]
    if not candidates:
        raise RuntimeError("cannot project {} m ahead".format(distance_m))
    return candidates[0]


def _ego_relative_blend_target(ego, pursuer, gap_ahead_m, bias):
    """Ego-conditioned interaction clock (carla-scenario-design skill): a."""
    world_map = CarlaDataProvider.get_map()
    ego_wp = world_map.get_waypoint(ego.get_location(), project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    pursuer_wp = world_map.get_waypoint(pursuer.get_location(), project_to_road=True,
                                         lane_type=carla.LaneType.Driving)
    ego_ahead = _lane_ahead(ego_wp, gap_ahead_m)
    pursuer_ahead = _lane_ahead(pursuer_wp, gap_ahead_m)
    return carla.Location(
        x=(1.0 - bias) * pursuer_ahead.transform.location.x + bias * ego_ahead.transform.location.x,
        y=(1.0 - bias) * pursuer_ahead.transform.location.y + bias * ego_ahead.transform.location.y,
        z=ego_ahead.transform.location.z)


def _ego_relative_push_target(ego, pursuer, gap_ahead_m):
    """The push target -- blended toward the pursuer's own lane so the."""
    return _ego_relative_blend_target(ego, pursuer, gap_ahead_m, _PUSH_EGO_BIAS)


def _ego_relative_rest_target(ego, pursuer, gap_ahead_m):
    """Scenario helper."""
    return _ego_relative_blend_target(ego, pursuer, gap_ahead_m, _REST_STRADDLE_BIAS)


class PursuerAbeamEgo(AtomicCondition):
    """Scenario helper."""

    def __init__(self, ego, pursuer, offset_m, name="PursuerAbeamEgo"):
        super(PursuerAbeamEgo, self).__init__(name)
        self._ego = ego
        self._pursuer = pursuer
        self._offset_m = offset_m

    def update(self):
        ego_transform = self._ego.get_transform()
        forward = ego_transform.get_forward_vector()
        delta = self._pursuer.get_location() - ego_transform.location
        longitudinal_offset = delta.x * forward.x + delta.y * forward.y
        if abs(longitudinal_offset) <= self._offset_m:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class HoldStationary(AtomicBehavior):
    """Scenario helper."""

    def __init__(self, actor):
        super(HoldStationary, self).__init__("HoldStationary", actor)

    def update(self):
        self._actor.apply_control(carla.VehicleControl(
            throttle=0.0, brake=1.0, hand_brake=True))
        return py_trees.common.Status.RUNNING


def _emit_event(name, ego, actors, reason):
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "AdjacentVanLateralSqueezeThenStopTown12", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
                "frame": snapshot.frame, "simulation_time": snapshot.timestamp.elapsed_seconds,
                "route_progress_m": float(getattr(ego, "_route_progress_m", 0.0)), "reason": reason,
                "actors": [{"id": actor.id, "role_name": actor.attributes.get("role_name", ""),
                            "alive": actor.is_alive} for actor in actors if actor is not None]}
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    except Exception:
        pass


class EventMarker(AtomicBehavior):
    def __init__(self, event, ego, actors, reason):
        super(EventMarker, self).__init__("B03EventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


class PointSeek(AtomicBehavior):
    """Shared bounded point-seeking controller for the push and spin-out
    phases. `target_fn`, if given, is called once from `initialise()` to
    compute the target live (ego-relative push phase); `target_location`
    is used directly otherwise (fixed spin-out exit target)."""

    def __init__(self, actor, timeout_s, speed_mps, target_location=None, target_fn=None,
                 spin_bias=0.0, clear_radius_m=2.4, brake_after_s=None):
        super(PointSeek, self).__init__("PointSeek", actor)
        self._target = target_location
        self._target_fn = target_fn
        self._timeout_s = timeout_s
        self._speed_mps = speed_mps
        self._spin_bias = spin_bias
        self._clear_radius_m = clear_radius_m
        self._brake_after_s = brake_after_s
        self._started = None
        self._steer = 0.0

    def initialise(self):
        self._started = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        if self._target_fn is not None:
            self._target = self._target_fn()

    def update(self):
        elapsed = (CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
                   - self._started)
        transform = self._actor.get_transform()
        dx = self._target.x - transform.location.x
        dy = self._target.y - transform.location.y
        remaining = math.hypot(dx, dy)
        wanted_yaw = math.atan2(dy, dx)
        yaw = math.radians(transform.rotation.yaw)
        yaw_error = (wanted_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
        steer_target = max(-0.55, min(0.55, 0.8 * yaw_error + self._spin_bias))
        self._steer = max(self._steer - 0.045, min(self._steer + 0.045, steer_target))
        velocity = self._actor.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        brake = 0.0
        if self._brake_after_s is not None and elapsed > self._brake_after_s:
            brake = 0.3
        throttle = (max(0.0, min(0.45, 0.15 + 0.05 * (self._speed_mps - speed)))
                    if not brake else 0.0)
        self._actor.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=self._steer))
        if elapsed >= self._timeout_s or remaining < self._clear_radius_m:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._actor.is_alive:
            self._actor.apply_control(carla.VehicleControl(brake=0.4))
        super(PointSeek, self).terminate(new_status)


class AdjacentVanLateralSqueezeThenStopTown12(BasicScenario):
    timeout = 40

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=40):
        self._map = CarlaDataProvider.get_map()
        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._trigger_wp is None or self._trigger_wp.is_junction
                or self._trigger_wp.road_id != ROAD_ID):
            raise ValueError("requires verified Town06 road {}".format(ROAD_ID))
        self._pursuer_lane_wp = _outboard_driving(self._trigger_wp, hops=1)
        _inboard_driving(self._trigger_wp)

        self._pursuer_spawn_m = _value(config, "pursuer_spawn_distance_m", 4.0, 2.0, 8.0)
        self._pursuer_speed_mps = _value(config, "pursuer_speed_mps", 8.0, 6.0, 12.0)
        self._gate_speed_mps = _value(config, "gate_speed_mps", 7.5, 5.0, 9.0)
        self._abeam_offset_m = _value(config, "abeam_offset_m", 2.0, 1.0, 3.0)
        self._push_gap_ahead_m = _value(config, "push_gap_ahead_m", 6.0, 4.0, 10.0)
        self._push_duration_s = _value(config, "push_duration_s", 2.5, 1.5, 4.0)
        self._spin_speed_mps = _value(config, "spin_speed_mps", 4.0, 2.5, 6.5)
        self._spin_exit_timeout_s = _value(config, "spin_exit_timeout_s", 4.0, 2.5, 6.0)
        self._rest_gap_ahead_m = _value(config, "rest_gap_ahead_m", 12.0, 10.0, 15.0)
        self._max_exposure_s = _value(config, "max_exposure_s", 12.0, 8.0, 18.0)

        self._pursuer = None
        super(AdjacentVanLateralSqueezeThenStopTown12, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        pursuer_wp = _advance(self._pursuer_lane_wp, self._pursuer_spawn_m)
        self._pursuer = CarlaDataProvider.request_new_actor(
            "vehicle.mercedes.sprinter", pursuer_wp.transform,
            rolename="scenario.pursuer", color="20,20,20")
        if self._pursuer is None:
            raise RuntimeError("pursuer spawn failed")
        self.other_actors.append(self._pursuer)

    def _create_behavior(self):
        ego = self.ego_vehicles[0]

        arm_gate = py_trees.composites.Parallel(
            "B03ArmGate", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        arm_gate.add_child(TriggerVelocity(
            ego, self._gate_speed_mps, comparison_operator=operator.ge, name="B03EgoSpeedGate"))
        arm_gate.add_child(PursuerAbeamEgo(ego, self._pursuer, self._abeam_offset_m))

        arm = py_trees.composites.Parallel(
            "B03Approach", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(arm_gate)
        arm.add_child(DriveDistance(ego, 45.0, name="B03ProgressFallback"))
        arm.add_child(WaypointFollower(self._pursuer, self._pursuer_speed_mps, name="B03PursuerApproach"))

        event = py_trees.composites.Sequence("B03PushThenRestEvent")
        event.add_child(arm)
        event.add_child(EventMarker("armed", ego, [self._pursuer],
                                       "ego-relative arm condition satisfied"))
        event.add_child(EventMarker("committed", ego, [self._pursuer],
                                       "pursuer committed to the lateral push maneuver"))
        event.add_child(EventMarker("conflict_entered", ego, [self._pursuer],
                                       "sustained lateral push began"))
        event.add_child(PointSeek(
            self._pursuer, self._push_duration_s, self._pursuer_speed_mps,
            target_fn=lambda: _ego_relative_push_target(ego, self._pursuer, self._push_gap_ahead_m),
            spin_bias=0.0, clear_radius_m=1.0))
        event.add_child(EventMarker("spin_out_begins", ego, [self._pursuer],
                                       "push held, pursuer now loses control and spins toward a stop"))
        event.add_child(PointSeek(
            self._pursuer, self._spin_exit_timeout_s, self._spin_speed_mps,
            target_fn=lambda: _ego_relative_rest_target(ego, self._pursuer, self._rest_gap_ahead_m),
            spin_bias=0.35, clear_radius_m=2.4,
            brake_after_s=self._spin_exit_timeout_s * 0.6))

        hold = py_trees.composites.Parallel(
            "B03HoldOrClear", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        hold.add_child(HoldStationary(self._pursuer))
        hold.add_child(DriveDistance(ego, 85.0, name="B03EgoClears"))
        hold.add_child(ScenarioTimeout(self._max_exposure_s, self.__class__.__name__,
                                       name="B03ExposureTimeout"))
        event.add_child(hold)
        event.add_child(EventMarker("cleared", ego, [self._pursuer],
                                       "ego cleared the bounded exposure window"))
        event.add_child(ActorDestroy(self._pursuer, name="B03DestroyPursuer"))
        return event

    def _create_test_criteria(self):
        return [
            CollisionTest(self.ego_vehicles[0], other_actor=self._pursuer,
                          terminate_on_failure=True, name="B03EgoPursuerCollision"),
            ActorBlockedTest(self.ego_vehicles[0], min_speed=5.0, max_time=8.0,
                             terminate_on_failure=True, name="B03Blocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
