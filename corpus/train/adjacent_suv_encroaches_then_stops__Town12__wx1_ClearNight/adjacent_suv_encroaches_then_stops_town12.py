"""Adjacent suv encroaches then stops (Town12).

An SUV encroaches laterally from the adjacent lane and stops close alongside the ego.
"""
from __future__ import print_function

import json
import math
import operator
import os

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    AtomicBehavior, ScenarioTimeout, WaypointFollower)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest, CollisionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance, InTimeToArrivalToLocation, InTriggerDistanceToLocation, TriggerVelocity)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront


def _emit_event(name, ego, actors, reason):
    """Scenario helper."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "AdjacentSuvEncroachesThenStopsTown12", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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


def _emit_suv_track(name, actor, extra):
    """Additive instrumentation, no behavior change -- structured per-tick
    SUV-lane telemetry, mirroring 's `_emit_debris_event`/'s
    `_emit_rig_track`."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "AdjacentSuvEncroachesThenStopsTown12", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
                "actors": [{"id": actor.id, "role_name": actor.attributes.get("role_name", ""),
                            "alive": actor.is_alive}] if actor is not None else []}
        item.update(extra)
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    except Exception:
        pass


class EventMarker(AtomicBehavior):
    def __init__(self, event, ego, actors, reason):
        super(EventMarker, self).__init__("B07EventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


class HoldStationary(AtomicBehavior):
    """Scenario helper."""

    def __init__(self, actor):
        super(HoldStationary, self).__init__("HoldStationary", actor)

    def update(self):
        self._actor.apply_control(carla.VehicleControl(
            throttle=0.0, brake=1.0, hand_brake=True))
        return py_trees.common.Status.RUNNING


class SustainedClearAmbient(AtomicBehavior):
    """Scenario helper."""

    def __init__(self, actor, radius_m):
        super(SustainedClearAmbient, self).__init__("SustainedClearAmbient", actor)
        self._radius_m = radius_m

    def update(self):
        return py_trees.common.Status.RUNNING


def _value(config, name, default, low, high):
    value = float(config.other_parameters.get(name, {}).get("value", default))
    if value < low or value > high:
        raise ValueError("{} outside [{}, {}]".format(name, low, high))
    return value


def _advance(waypoint, distance_m):
    candidates = [item for item in waypoint.next(distance_m)
                  if item.road_id == waypoint.road_id
                  and item.lane_id == waypoint.lane_id
                  and item.lane_type == carla.LaneType.Driving
                  and not item.is_junction]
    if not candidates:
        raise ValueError("cannot advance {} m on road 38".format(distance_m))
    return candidates[0]


def _adjacent_outboard(waypoint):
    candidate = waypoint.get_right_lane()
    if (candidate is None or candidate.lane_type != carla.LaneType.Driving
            or candidate.road_id != waypoint.road_id
            or candidate.lane_id * waypoint.lane_id <= 0):
        raise ValueError("requires an outboard adjacent driving lane")
    return candidate


def _encroach_target_from_live(actor, ahead_m, lateral_fraction):
    """Scenario helper."""
    world_map = CarlaDataProvider.get_map()
    live_wp = world_map.get_waypoint(
        actor.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    if live_wp is None:
        raise RuntimeError("could not resolve the SUV's live waypoint at commit")
    conflict_wp = _advance(live_wp, ahead_m)
    left = conflict_wp.transform.get_right_vector() * -1.0
    offset_m = lateral_fraction * conflict_wp.lane_width
    return carla.Location(
        x=conflict_wp.transform.location.x + left.x * offset_m,
        y=conflict_wp.transform.location.y + left.y * offset_m,
        z=conflict_wp.transform.location.z)


class SuvAggressive(AtomicBehavior):
    """Scenario helper."""

    def __init__(self, actor, ahead_m, lateral_fraction, cruise_speed_mps,
                 encroach_duration_s, timeout_s):
        super(SuvAggressive, self).__init__("SuvAggressive", actor)
        self._ahead_m = ahead_m
        self._lateral_fraction = lateral_fraction
        self._cruise_speed_mps = cruise_speed_mps
        self._encroach_duration_s = encroach_duration_s
        self._timeout_s = timeout_s
        self._target = None
        self._start_s = None
        self._last_steer = 0.0

    def initialise(self):
        self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        self._target = _encroach_target_from_live(
            self._actor, self._ahead_m, self._lateral_fraction)

    def update(self):
        now_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        elapsed_s = now_s - self._start_s
        transform = self._actor.get_transform()
        dx = self._target.x - transform.location.x
        dy = self._target.y - transform.location.y
        remaining = math.hypot(dx, dy)
        yaw_rad = math.radians(transform.rotation.yaw)
        forward_x, forward_y = math.cos(yaw_rad), math.sin(yaw_rad)
        overrun = (dx * forward_x + dy * forward_y) < 0.0
        near_target = remaining < 2.6
        near_timeout = elapsed_s > (self._encroach_duration_s - 1.0)

        _emit_suv_track("suv_track", self._actor, {
            "x": transform.location.x, "y": transform.location.y,
            "elapsed_s": elapsed_s, "remaining_m": remaining, "overrun": overrun})

        if overrun or near_target or near_timeout:
            steer = self._last_steer
            throttle = 0.0
            brake = 1.0
            hand_brake = True
        else:
            desired_yaw = math.atan2(dy, dx)
            yaw_error = (desired_yaw - yaw_rad + math.pi) % (2 * math.pi) - math.pi
            desired_steer = max(-0.20, min(0.20, 0.55 * yaw_error))
            steer = max(self._last_steer - 0.035, min(self._last_steer + 0.035, desired_steer))
            velocity = self._actor.get_velocity()
            speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            speed_error = self._cruise_speed_mps - speed_mps
            throttle = max(0.0, min(0.5, 0.18 + 0.05 * speed_error))
            brake = max(0.0, min(0.2, -0.05 * speed_error))
            if brake > 0.02:
                throttle = 0.0
            hand_brake = False
        self._last_steer = steer
        self._actor.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=steer, hand_brake=hand_brake))
        if (elapsed_s >= self._encroach_duration_s or elapsed_s >= self._timeout_s
                or (elapsed_s > 1.0 and near_target)):
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._actor.is_alive:
            self._actor.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        super(SuvAggressive, self).terminate(new_status)


class AdjacentSuvEncroachesThenStopsTown12(BasicScenario):
    timeout = 50

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=50):
        self._map = CarlaDataProvider.get_map()
        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._trigger_wp is None or self._trigger_wp.is_junction
                or self._trigger_wp.road_id != 738):
            raise ValueError("requires verified Town04 road 38")
        self._suv_lane_wp = _adjacent_outboard(self._trigger_wp)
        self._suv_spawn_m = _value(config, "actor_spawn_distance_m", 16.0, 12.0, 22.0)
        self._arm_distance_m = _value(config, "arm_distance_m", 12.0, 10.0, 18.0)
        self._suv_speed_mps = _value(config, "actor_speed_mps", 9.0, 7.0, 12.0)
        self._conflict_m = _value(config, "conflict_distance_m", 31.0, 25.0, 35.0)
        self._encroach_duration_s = _value(config, "encroach_duration_s", 3.5, 2.5, 5.0)
        self._arm_ttc_s = _value(config, "arm_ttc_s", 5.0, 3.5, 6.0)
        self._gate_speed_mps = _value(config, "gate_speed_mps", 7.5, 5.0, 9.0)
        self._max_exposure_s = _value(config, "max_exposure_s", 20.0, 12.0, 28.0)
        self._suv = None
        super(AdjacentSuvEncroachesThenStopsTown12, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        suv_wp = _advance(self._suv_lane_wp, self._suv_spawn_m)
        self._suv = CarlaDataProvider.request_new_actor(
            "vehicle.nissan.patrol", suv_wp.transform,
            rolename="scenario.suv", color="18,18,20")
        if self._suv is None:
            raise RuntimeError("SUV spawn failed")
        self.other_actors.append(self._suv)

        # Conflict location for TTC measurement (ego's arm condition)
        conflict_wp = _advance(self._suv_lane_wp, self._conflict_m)
        left = conflict_wp.transform.get_right_vector() * -1.0
        offset_m = 0.85 * conflict_wp.lane_width
        self._encroach_location = carla.Location(
            x=conflict_wp.transform.location.x + left.x * offset_m,
            y=conflict_wp.transform.location.y + left.y * offset_m,
            z=conflict_wp.transform.location.z)

        # SUV's live-relative drift corridor length
        self._encroach_ahead_m = max(self._conflict_m - self._suv_spawn_m, 8.0)
        self._encroach_lateral_fraction = 0.85

    def _create_behavior(self):
        ego = self.ego_vehicles[0]

        gate = py_trees.composites.Parallel(
            "B07EgoSpeedGate", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        gate.add_child(TriggerVelocity(
            ego, self._gate_speed_mps, comparison_operator=operator.ge, name="B07EgoSpeedGate"))
        gate.add_child(WaypointFollower(self._suv, self._suv_speed_mps, name="B07SuvPreRoll"))
        gate.add_child(DriveDistance(ego, 30.0, name="B07GateProgressFallback"))

        # TTC-based arm with distance and progress fallbacks
        arm = py_trees.composites.Parallel(
            "B07ApproachAndArm", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(InTimeToArrivalToLocation(
            ego, self._arm_ttc_s, self._encroach_location,
            name="B07TTCArm"))
        arm.add_child(InTriggerDistanceToLocation(
            ego, self._encroach_location, self._arm_distance_m,
            name="B07DistanceBackstop"))
        arm.add_child(DriveDistance(ego, 45.0, name="B07ProgressFallback"))
        arm.add_child(WaypointFollower(self._suv, self._suv_speed_mps, name="B07SuvApproach"))

        ego_location = ego.get_location()
        ambient_clearance_m = ego_location.distance(self._encroach_location) + 15.0

        event = py_trees.composites.Sequence("B07EventPhases")
        event.add_child(LeaveSpaceInFront(ambient_clearance_m, name="B07ClearAmbientApproach"))
        event.add_child(gate)
        event.add_child(arm)
        event.add_child(EventMarker("armed", ego, [self._suv],
                                       "ego-relative arm condition satisfied"))
        event.add_child(EventMarker("committed", ego, [self._suv],
                                       "SUV aggressive encroachment maneuver committed"))
        event.add_child(EventMarker("conflict_entered", ego, [self._suv],
                                       "SUV lateral encroachment began"))
        event.add_child(SuvAggressive(
            self._suv, self._encroach_ahead_m, self._encroach_lateral_fraction,
            self._suv_speed_mps, self._encroach_duration_s, 8.0))
        clear = py_trees.composites.Parallel(
            "B07ClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(ego, 75.0, name="B07EgoClears"))
        clear.add_child(ScenarioTimeout(self._max_exposure_s, self.__class__.__name__,
                                        name="B07ExposureTimeout"))
        clear.add_child(HoldStationary(self._suv))
        event.add_child(clear)
        event.add_child(EventMarker("cleared", ego, [self._suv],
                                       "ego cleared the bounded exposure window"))
        return event

    def _create_test_criteria(self):
        return [
            CollisionTest(
                self.ego_vehicles[0], other_actor=self._suv,
                terminate_on_failure=True, name="B07EgoSuvCollision"),
            ActorBlockedTest(
                self.ego_vehicles[0], min_speed=5.0, max_time=15.0,
                terminate_on_failure=True, name="B07SuvBlocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
