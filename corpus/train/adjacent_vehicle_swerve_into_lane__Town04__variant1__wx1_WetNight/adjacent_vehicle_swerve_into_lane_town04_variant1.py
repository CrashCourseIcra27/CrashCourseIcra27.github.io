"""Adjacent vehicle swerve into lane (Town04).

A vehicle in the adjacent lane veers suddenly into the ego's lane and commits to the intrusion.
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
    AtomicBehavior, ScenarioTimeout, WaypointFollower)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest, CollisionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance, InTimeToArrivalToLocation)
from srunner.scenarios.basic_scenario import BasicScenario

PRE_ARM_CRUISE_SPEED_MPS = 4.0


def _parameter(config, name, default, low, high):
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
        raise ValueError("cannot advance {} m on road 35".format(distance_m))
    return candidates[0]


def _adjacent_inboard(waypoint):
    candidate = waypoint.get_left_lane()
    if (candidate is None or candidate.lane_type != carla.LaneType.Driving
            or candidate.road_id != waypoint.road_id
            or candidate.lane_id * waypoint.lane_id <= 0):
        raise ValueError("requires an inboard adjacent driving lane")
    return candidate


def _emit_event(name, ego, actors, reason):
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "AdjacentVehicleSwerveIntoLaneTown04Variant1", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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
        super(EventMarker, self).__init__("A03EventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


class BlowoutIntrusion(AtomicBehavior):
    """Bounded weave followed by one curved physical intrusion."""

    def __init__(self, actor, conflict_location, speed_mps, weave_s, timeout_s):
        super(BlowoutIntrusion, self).__init__("BlowoutIntrusion", actor)
        self._conflict = conflict_location
        self._speed_mps = speed_mps
        self._weave_s = weave_s
        self._timeout_s = timeout_s
        self._start_s = None
        self._last_steer = 0.0

    def initialise(self):
        self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds

    def update(self):
        now_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        elapsed_s = now_s - self._start_s
        transform = self._actor.get_transform()
        dx = self._conflict.x - transform.location.x
        dy = self._conflict.y - transform.location.y
        distance_m = math.hypot(dx, dy)
        if elapsed_s < self._weave_s:
            desired_steer = 0.20 * math.sin(2.0 * math.pi * elapsed_s / self._weave_s)
        else:
            self._actor.set_light_state(carla.VehicleLightState.RightBlinker)
            desired_yaw = math.atan2(dy, dx)
            current_yaw = math.radians(transform.rotation.yaw)
            yaw_error = (desired_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
            desired_steer = max(-0.52, min(0.52, 0.82 * yaw_error))
        steer = max(self._last_steer - 0.045,
                    min(self._last_steer + 0.045, desired_steer))
        self._last_steer = steer
        velocity = self._actor.get_velocity()
        speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        speed_error = self._speed_mps - speed_mps
        throttle = max(0.0, min(0.52, 0.14 + 0.05 * speed_error))
        brake = max(0.0, min(0.35, -0.07 * speed_error))
        if brake > 0.02:
            throttle = 0.0
        self._actor.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=steer))
        if distance_m < 2.2 or elapsed_s >= self._timeout_s:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._actor.is_alive:
            self._actor.apply_control(carla.VehicleControl(brake=0.55))
        super(BlowoutIntrusion, self).terminate(new_status)


class AdjacentVehicleSwerveIntoLaneTown04Variant1(BasicScenario):
    timeout = 40

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=40):
        self._map = CarlaDataProvider.get_map()
        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._trigger_wp is None or self._trigger_wp.is_junction
                or self._trigger_wp.road_id != 35):
            raise ValueError("requires verified Town04 road 35")
        self._actor_lane_wp = _adjacent_inboard(self._trigger_wp)
        self._actor_spawn_m = _parameter(config, "actor_spawn_distance_m", 37.0, 30.0, 45.0)
        self._arm_distance_m = _parameter(config, "arm_distance_m", 26.0, 15.0, 28.0)
        self._arm_ttc_s = _parameter(config, "arm_ttc_s", 2.4, 1.5, 3.2)
        self._actor_speed_mps = _parameter(config, "actor_speed_mps", 12.0, 5.0, 13.0)
        self._weave_s = _parameter(config, "weave_duration_s", 0.4, 0.3, 0.8)
        self._conflict_m = _parameter(config, "conflict_distance_m", 60.0, 50.0, 70.0)
        self._max_exposure_s = _parameter(config, "max_exposure_s", 14.0, 6.0, 20.0)
        self._actor = None
        super(AdjacentVehicleSwerveIntoLaneTown04Variant1, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        actor_wp = _advance(self._actor_lane_wp, self._actor_spawn_m)
        self._actor = CarlaDataProvider.request_new_actor(
            "vehicle.lincoln.mkz_2017", actor_wp.transform,
            rolename="scenario.sedan", color="20,20,25")
        if self._actor is None:
            raise RuntimeError("sedan spawn failed")
        self.other_actors.append(self._actor)

    def _create_behavior(self):
        conflict = _advance(self._trigger_wp, self._conflict_m).transform.location

        arm = py_trees.composites.Parallel(
            "A03ApproachAndArm", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._arm_ttc_s, conflict,
            comparison_operator=operator.lt, name="A03LiveTTCArm"))
        arm_progress_fallback_m = max(self._conflict_m - 5.0, 50.0)
        arm.add_child(DriveDistance(
            self.ego_vehicles[0], arm_progress_fallback_m, name="A03ProgressFallback"))
        arm.add_child(WaypointFollower(
            self._actor, PRE_ARM_CRUISE_SPEED_MPS, name="A03AdjacentCreep"))

        event = py_trees.composites.Sequence("A03EventPhases")
        event.add_child(arm)
        event.add_child(EventMarker("armed", self.ego_vehicles[0], [self._actor],
                                       "ego-relative arm condition satisfied"))
        event.add_child(EventMarker("committed", self.ego_vehicles[0], [self._actor],
                                       "adjacent-lane instability committed"))
        event.add_child(EventMarker("conflict_entered", self.ego_vehicles[0], [self._actor],
                                       "physical lane-intrusion maneuver began"))
        event.add_child(BlowoutIntrusion(
            self._actor, conflict, self._actor_speed_mps,
            self._weave_s, 7.0))
        clear = py_trees.composites.Parallel(
            "A03ClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(
            self.ego_vehicles[0], 90.0, name="A03EgoClears"))
        clear.add_child(ScenarioTimeout(
            self._max_exposure_s, self.__class__.__name__, name="A03ExposureTimeout"))
        event.add_child(clear)
        event.add_child(EventMarker("cleared", self.ego_vehicles[0], [self._actor],
                                       "ego cleared the bounded exposure window"))
        return event

    def _create_test_criteria(self):
        return [
            CollisionTest(
                self.ego_vehicles[0], other_actor=self._actor,
                terminate_on_failure=True, name="A03EgoIntrudingSedanCollision"),
            ActorBlockedTest(
                self.ego_vehicles[0], min_speed=5.0, max_time=8.0,
                terminate_on_failure=True, name="A03IntrusionBlocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
