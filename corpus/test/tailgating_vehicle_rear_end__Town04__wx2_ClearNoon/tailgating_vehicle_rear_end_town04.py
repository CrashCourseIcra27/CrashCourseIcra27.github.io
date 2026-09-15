"""Tailgating vehicle rear end (Town04).

A tailgating vehicle closes from directly behind and strikes the ego from the rear.
"""
from __future__ import print_function

import json
import math
import os

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    AtomicBehavior, ActorDestroy, ScenarioTimeout)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest, CollisionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance, InTriggerDistanceToLocation)
from srunner.scenarios.basic_scenario import BasicScenario

ROAD_ID_AT_TRIGGER = 45


def _parameter(config, name, default, low, high):
    result = float(config.other_parameters.get(name, {}).get("value", default))
    if result < low or result > high:
        raise ValueError("{} outside [{}, {}]".format(name, low, high))
    return result


def _walk_lane(waypoint, distance_m, forward=True):
    """Advance/retreat along the SAME lane_id, staying `Driving` and
    non-junction -- deliberately NOT scoped to a single `road_id` (see
    module docstring: road 45's own pre-trigger stretch is only 26.51 m,
    shorter than spawn_gap_m, so a same-road_id-only walk would reject
    the prescribed default). A real OpenDRIVE road_id boundary that keeps
    the same lane_id/Driving/non-junction is a topology segment boundary,
    not a lane change or a junction."""
    candidates = (waypoint.next(distance_m) if forward
                  else waypoint.previous(distance_m))
    candidates = [item for item in candidates
                  if item.lane_id == waypoint.lane_id
                  and item.lane_type == carla.LaneType.Driving
                  and not item.is_junction]
    if not candidates:
        raise ValueError("cannot walk {} m ({}) on lane {}".format(
            distance_m, "forward" if forward else "backward", waypoint.lane_id))
    return candidates[0]


def _emit_event(name, ego, actors, reason):
    """Scenario helper."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "TailgatingVehicleRearEndTown04", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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


class ClearAmbient(AtomicBehavior):
    """Reissue LeaveSpaceInFront every tick to keep ambient traffic clear."""

    def __init__(self, space_m, name="ClearAmbient"):
        super(ClearAmbient, self).__init__(name)
        self._space_m = space_m

    def update(self):
        py_trees.blackboard.Blackboard().set("BA_LeaveSpaceInFront", self._space_m, overwrite=True)
        return py_trees.common.Status.RUNNING


class EventMarker(AtomicBehavior):
    def __init__(self, event, ego, actors, reason):
        super(EventMarker, self).__init__("A01aEventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


class RearPursuit(AtomicBehavior):
    """Lane-locked pursuit with AGGRESSIVE EGO-RELATIVE closing law."""

    LOOKAHEAD_M = 6.0
    STEER_RATE_LIMIT = 0.045
    STEER_CAP = 0.55
    THROTTLE_RATE_LIMIT = 0.08
    BRAKE_RATE_LIMIT = 0.10
    GAP_TIME_CONSTANT_S = 2.25
    GAP_BONUS_FLOOR_MPS = 5.0
    GAP_BONUS_CEIL_MPS = 17.0
    V_CMD_FLOOR_MPS = 8.0
    V_CMD_CEIL_MPS = 30.0

    def __init__(self, actor, ego, timeout_s, gap_time_constant_s=None,
                 gap_bonus_floor_mps=None, gap_bonus_ceil_mps=None):
        super(RearPursuit, self).__init__("RearPursuit", actor)
        self._ego = ego
        self._timeout_s = timeout_s
        self._gap_time_constant_s = gap_time_constant_s or self.GAP_TIME_CONSTANT_S
        self._gap_bonus_floor_mps = gap_bonus_floor_mps or self.GAP_BONUS_FLOOR_MPS
        self._gap_bonus_ceil_mps = gap_bonus_ceil_mps or self.GAP_BONUS_CEIL_MPS
        self._start_s = None
        self._last_steer = 0.0
        self._last_throttle = 0.0
        self._last_brake = 0.0

    def initialise(self):
        self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        self._actor.set_light_state(carla.VehicleLightState.HighBeam)

    def _lane_lookahead_location(self):
        wp = CarlaDataProvider.get_map().get_waypoint(
            self._actor.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        try:
            ahead = _walk_lane(wp, self.LOOKAHEAD_M, forward=True)
            return ahead.transform.location
        except ValueError:
            return wp.transform.location

    def update(self):
        now_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        elapsed_s = now_s - self._start_s
        transform = self._actor.get_transform()

        # -- lane-locked steering (pure lane-following, never toward ego) --
        target_loc = self._lane_lookahead_location()
        dx = target_loc.x - transform.location.x
        dy = target_loc.y - transform.location.y
        desired_yaw = math.atan2(dy, dx)
        current_yaw = math.radians(transform.rotation.yaw)
        yaw_error = (desired_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi
        desired_steer = max(-self.STEER_CAP, min(self.STEER_CAP, 0.85 * yaw_error))
        steer = max(self._last_steer - self.STEER_RATE_LIMIT,
                    min(self._last_steer + self.STEER_RATE_LIMIT, desired_steer))
        self._last_steer = steer

        ego_loc = self._ego.get_location()
        gap_m = math.hypot(ego_loc.x - transform.location.x, ego_loc.y - transform.location.y)
        ego_vel = self._ego.get_velocity()
        v_ego = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)
        gap_bonus = max(self._gap_bonus_floor_mps,
                         min(self._gap_bonus_ceil_mps, gap_m / self._gap_time_constant_s))
        v_cmd = max(self.V_CMD_FLOOR_MPS, min(self.V_CMD_CEIL_MPS, v_ego + gap_bonus))

        velocity = self._actor.get_velocity()
        speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_error = v_cmd - speed_mps
        desired_throttle = max(0.0, min(0.75, 0.20 + 0.06 * speed_error))
        desired_brake = max(0.0, min(0.5, -0.09 * speed_error))
        throttle = max(self._last_throttle - self.THROTTLE_RATE_LIMIT,
                       min(self._last_throttle + self.THROTTLE_RATE_LIMIT, desired_throttle))
        brake = max(self._last_brake - self.BRAKE_RATE_LIMIT,
                    min(self._last_brake + self.BRAKE_RATE_LIMIT, desired_brake))
        if brake > 0.02:
            throttle = 0.0
        self._last_throttle, self._last_brake = throttle, brake
        self._actor.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

        if elapsed_s >= self._timeout_s:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._actor.is_alive:
            self._actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=self._last_steer))
            self._actor.set_light_state(carla.VehicleLightState.Brake)
        super(RearPursuit, self).terminate(new_status)


class HoldStationary(AtomicBehavior):
    """Genuine brake-to-rest and hold -- full brake + handbrake, actively
    re-applied every tick, never left coasting."""

    def __init__(self, actor):
        super(HoldStationary, self).__init__("HoldStationary", actor)

    def initialise(self):
        try:
            self._actor.set_light_state(carla.VehicleLightState.Hazard)
        except AttributeError:
            pass

    def update(self):
        self._actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        return py_trees.common.Status.RUNNING


class TailgatingVehicleRearEndTown04(BasicScenario):
    timeout = 45

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=45):
        self._map = CarlaDataProvider.get_map()
        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._trigger_wp is None or self._trigger_wp.is_junction
                or self._trigger_wp.road_id != ROAD_ID_AT_TRIGGER):
            raise ValueError("requires verified Town04 road {}".format(ROAD_ID_AT_TRIGGER))

        self._spawn_gap_m = _parameter(config, "spawn_gap_m", 22.0, 15.0, 35.0)
        self._gap_time_constant_s = _parameter(config, "gap_time_constant_s", 3.0, 2.0, 5.0)
        self._gap_bonus_floor_mps = _parameter(config, "gap_bonus_floor_mps", 5.0, 3.0, 8.0)
        self._gap_bonus_ceil_mps = _parameter(config, "gap_bonus_ceil_mps", 14.0, 10.0, 18.0)
        self._v_cmd_ceil_mps = _parameter(config, "v_cmd_ceil_mps", 30.0, 25.0, 35.0)
        self._pursuit_timeout_s = _parameter(config, "pursuit_timeout_s", 12.0, 10.0, 14.0)
        self._max_exposure_s = _parameter(config, "max_exposure_s", 15.0, 10.0, 25.0)
        self._clear_distance_m = _parameter(config, "clear_distance_m", 60.0, 40.0, 90.0)
        self._ambient_clear_bubble_m = _parameter(config, "ambient_clear_bubble_m", 530.0, 0.0, 1000.0)
        self._aggressor = None
        super(TailgatingVehicleRearEndTown04, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        spawn_wp = _walk_lane(self._trigger_wp, self._spawn_gap_m, forward=False)
        self._aggressor = CarlaDataProvider.request_new_actor(
            "vehicle.tesla.model3", spawn_wp.transform,
            rolename="scenario.aggressor", color="10,10,10")
        if self._aggressor is None:
            raise RuntimeError("aggressor spawn failed")
        self.other_actors.append(self._aggressor)

    def _create_behavior(self):
        arm = py_trees.composites.Parallel(
            "A01aApproachAndArm", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._trigger_wp.transform.location, 3.0,
            name="A01aTriggerArm"))
        arm.add_child(DriveDistance(self.ego_vehicles[0], 5.0, name="A01aProgressFallback"))

        event = py_trees.composites.Sequence("A01aEventPhases")
        event.add_child(arm)
        event.add_child(EventMarker("armed", self.ego_vehicles[0], [self._aggressor],
                                        "ego-relative arm condition satisfied"))
        event.add_child(EventMarker("committed", self.ego_vehicles[0], [self._aggressor],
                                        "ego-relative rear pursuit committed"))
        event.add_child(EventMarker("conflict_entered", self.ego_vehicles[0], [self._aggressor],
                                        "lane-locked pursuit began"))
        event.add_child(RearPursuit(
            self._aggressor, self.ego_vehicles[0], self._pursuit_timeout_s,
            gap_time_constant_s=self._gap_time_constant_s,
            gap_bonus_floor_mps=self._gap_bonus_floor_mps,
            gap_bonus_ceil_mps=self._gap_bonus_ceil_mps))

        clear = py_trees.composites.Parallel(
            "A01aClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(self.ego_vehicles[0], self._clear_distance_m, name="A01aEgoClears"))
        clear.add_child(ScenarioTimeout(self._max_exposure_s, self.__class__.__name__,
                                        name="A01aExposureTimeout"))
        clear.add_child(HoldStationary(self._aggressor))
        event.add_child(clear)
        event.add_child(EventMarker("cleared", self.ego_vehicles[0], [self._aggressor],
                                        "ego cleared the bounded exposure window"))
        event.add_child(ActorDestroy(self._aggressor, name="A01aDestroyAggressor"))

        root = py_trees.composites.Parallel(
            "A01aWithAmbientClear", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(event)
        root.add_child(ClearAmbient(self._ambient_clear_bubble_m, name="A01aAmbientClear"))
        return root

    def _create_test_criteria(self):
        return [
            CollisionTest(
                self.ego_vehicles[0], other_actor=self._aggressor,
                terminate_on_failure=True, name="A01aEgoAggressorCollision"),
            ActorBlockedTest(
                self.ego_vehicles[0], min_speed=5.0, max_time=35.0,
                terminate_on_failure=True, name="A01aPursuitBlocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
