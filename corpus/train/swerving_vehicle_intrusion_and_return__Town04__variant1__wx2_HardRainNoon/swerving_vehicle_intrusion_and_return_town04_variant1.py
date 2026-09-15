"""Swerving vehicle intrusion and return (Town04).

A sedan alongside spins into the ego's lane and then rebounds back toward its own lane.
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
    DriveDistance, InTimeToArrivalToLocation)
from srunner.scenarios.basic_scenario import BasicScenario


def _value(config, name, default, low, high):
    value = float(config.other_parameters.get(name, {}).get("value", default))
    if value < low or value > high:
        raise ValueError("{} outside [{}, {}]".format(name, low, high))
    return value


def _next(waypoint, distance_m):
    candidates = [candidate for candidate in waypoint.next(distance_m)
                  if candidate.road_id == waypoint.road_id
                  and candidate.lane_id == waypoint.lane_id
                  and candidate.lane_type == carla.LaneType.Driving
                  and not candidate.is_junction]
    if not candidates:
        raise ValueError("cannot advance {} m".format(distance_m))
    return candidates[0]


def _neighbor(waypoint):
    for candidate in (waypoint.get_left_lane(), waypoint.get_right_lane()):
        if (candidate is not None and candidate.lane_type == carla.LaneType.Driving
                and candidate.road_id == waypoint.road_id
                and candidate.lane_id * waypoint.lane_id > 0):
            return candidate
    raise ValueError("needs a parallel sedan lane")


def _emit_event(name, ego, actors, reason):
    """Scenario helper."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "SwervingVehicleIntrusionAndReturnTown04Variant1", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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
        super(EventMarker, self).__init__("A08EventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


class IcySpinAndRebound(AtomicBehavior):
    """Bounded steering loss sends the sedan left, then back across the roadway."""

    def __init__(self, actor, left_target, rebound_target, timeout_s):
        super(IcySpinAndRebound, self).__init__("IcySpinAndRebound", actor)
        self._left_target = left_target
        self._rebound_target = rebound_target
        self._timeout_s = timeout_s
        self._started = None
        self._steer = 0.0

    def initialise(self):
        self._started = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        try:
            self._actor.set_light_state(carla.VehicleLightState.Hazard)
        except AttributeError:
            pass

    def update(self):
        elapsed = (CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
                   - self._started)
        target = self._left_target if elapsed < 1.8 else self._rebound_target
        transform = self._actor.get_transform()
        dx = target.x - transform.location.x
        dy = target.y - transform.location.y
        remaining = math.hypot(dx, dy)
        wanted = math.atan2(dy, dx)
        yaw = math.radians(transform.rotation.yaw)
        yaw_error = (wanted - yaw + math.pi) % (2.0 * math.pi) - math.pi
        spin_bias = 0.28 if elapsed < 1.1 else (-0.25 if elapsed < 2.8 else 0.0)
        wanted_steer = max(-0.52, min(0.52, 0.84 * yaw_error + spin_bias))
        self._steer = max(self._steer - 0.035,
                          min(self._steer + 0.035, wanted_steer))
        velocity = self._actor.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        brake = 0.18 if elapsed > 1.6 else 0.0
        throttle = max(0.0, min(0.35, 0.12 + 0.035 * (8.0 - speed))) if not brake else 0.0
        self._actor.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=self._steer))
        if elapsed >= self._timeout_s or (elapsed > 2.2 and remaining < 2.6):
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._actor.is_alive:
            self._actor.apply_control(carla.VehicleControl(brake=0.6))
        super(IcySpinAndRebound, self).terminate(new_status)


class StopSemi(AtomicBehavior):
    """Scenario helper."""

    def __init__(self, actor):
        super(StopSemi, self).__init__("StopSemi", actor)

    def update(self):
        if self._actor is not None and self._actor.is_alive:
            self._actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        return py_trees.common.Status.SUCCESS


class SwervingVehicleIntrusionAndReturnTown04Variant1(BasicScenario):
    timeout = 42

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=42):
        world_map = CarlaDataProvider.get_map()
        self._trigger_wp = world_map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if self._trigger_wp is None or self._trigger_wp.is_junction:
            raise ValueError("trigger must be a non-junction driving waypoint")
        self._sedan_lane = _neighbor(self._trigger_wp)
        self._sedan_spawn_m = _value(config, "sedan_spawn_distance_m", 17.0, 12.0, 25.0)
        self._semi_spawn_m = _value(config, "semi_spawn_distance_m", 165.0, 160.0, 200.0)
        self._arm_ttc_s = _value(config, "arm_ttc_s", 2.0, 1.0, 2.5)
        self._creep_speed_mps = _value(config, "creep_speed_mps", 2.0, 1.0, 3.0)
        self._sedan_speed = _value(config, "sedan_speed_mps", 8.0, 5.0, 12.0)
        self._semi_speed = _value(config, "semi_speed_mps", 7.0, 4.0, 11.0)
        self._max_exposure_s = _value(config, "max_exposure_s", 12.0, 8.0, 18.0)
        self._sedan = self._semi = None
        super(SwervingVehicleIntrusionAndReturnTown04Variant1, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        sedan_wp = _next(self._sedan_lane, self._sedan_spawn_m)
        semi_wp = _next(self._sedan_lane, self._semi_spawn_m)
        self._sedan = CarlaDataProvider.request_new_actor(
            "vehicle.lincoln.mkz_2020", sedan_wp.transform,
            rolename="scenario.ice_sedan", color="18,18,24")
        self._semi = CarlaDataProvider.request_new_actor(
            "vehicle.carlamotors.european_hgv", semi_wp.transform,
            rolename="scenario.distant_semi", color="235,235,235")
        if self._sedan is None or self._semi is None:
            raise RuntimeError("vehicle spawn failed")
        self.other_actors.extend([self._sedan, self._semi])

    def _create_behavior(self):
        left_target = _next(self._trigger_wp, 23.0).transform.location
        rebound_target = _next(self._sedan_lane, 28.0).transform.location

        arm = py_trees.composites.Parallel(
            "A08Approach", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._arm_ttc_s, left_target,
            comparison_operator=operator.lt, name="A08LiveTTCArm"))
        # Generous safety-net fallback only (never expected to fire in
        # normal operation): prevents an indefinite hang if the ego somehow
        # never reaches the TTC threshold before nearing route end.
        arm.add_child(DriveDistance(self.ego_vehicles[0], 100.0, name="A08ProgressArmFallback"))
        arm.add_child(WaypointFollower(self._sedan, self._creep_speed_mps, name="A08SedanApproach"))
        event = py_trees.composites.Sequence("A08IceEvent")
        event.add_child(arm)
        event.add_child(EventMarker(
            "armed", self.ego_vehicles[0], [self._sedan, self._semi],
            "approach arm condition satisfied"))
        event.add_child(EventMarker(
            "committed", self.ego_vehicles[0], [self._sedan], "icy spin-and-rebound committed"))
        event.add_child(EventMarker(
            "conflict_entered", self.ego_vehicles[0], [self._sedan],
            "sedan spin maneuver began"))
        event.add_child(StopSemi(self._semi))
        event.add_child(IcySpinAndRebound(self._sedan, left_target, rebound_target, 5.5))
        clear = py_trees.composites.Parallel(
            "A08ClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(self.ego_vehicles[0], 95.0, name="A08EgoClears"))
        clear.add_child(ScenarioTimeout(self._max_exposure_s, self.__class__.__name__,
                                        name="A08ExposureTimeout"))
        event.add_child(clear)
        event.add_child(EventMarker(
            "cleared", self.ego_vehicles[0], [self._sedan, self._semi],
            "ego cleared the bounded exposure window"))
        return event

    def _create_test_criteria(self):
        return [
            CollisionTest(self.ego_vehicles[0], other_actor=self._sedan,
                          terminate_on_failure=True, name="A08EgoIcySedanCollision"),
            ActorBlockedTest(self.ego_vehicles[0], min_speed=5.0, max_time=8.0,
                             terminate_on_failure=True, name="A08IceBlocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
