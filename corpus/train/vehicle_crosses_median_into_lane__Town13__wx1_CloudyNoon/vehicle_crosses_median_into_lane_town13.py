"""Vehicle crosses median into lane (Town13).

An SUV crosses from the median shoulder into the ego's lane.
"""
from __future__ import print_function

import math
import json
import os

import carla
import py_trees
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    AtomicBehavior, ScenarioTimeout, WaypointFollower)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest, CollisionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance, InTriggerDistanceToVehicle)
from srunner.scenarios.basic_scenario import BasicScenario

ROAD_ID = 1216


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


def _outboard_escape_driving(waypoint):
    """the acceptance criterion machine-checked gate: the suv's entire swept path is the median."""
    candidate = waypoint.get_right_lane()
    if (candidate is None or candidate.lane_type != carla.LaneType.Driving
            or candidate.road_id != waypoint.road_id
            or candidate.lane_id * waypoint.lane_id <= 0):
        raise ValueError("requires an outboard (right) escape lane, independent of the suv's swept path")
    return candidate


def _shoulder_preroll_plan(shoulder_wp, span_m, step_m=2.0):
    """Scenario helper."""
    plan = [(shoulder_wp, RoadOption.LANEFOLLOW)]
    cur = shoulder_wp
    travelled = 0.0
    while travelled < span_m:
        nxts = cur.next(step_m)
        if not nxts:
            raise ValueError("shoulder pre-roll: dead end after {:.1f} m".format(travelled))
        cur = nxts[0]
        if (cur.lane_type != shoulder_wp.lane_type or cur.road_id != shoulder_wp.road_id
                or cur.lane_id != shoulder_wp.lane_id or cur.is_junction):
            raise ValueError("shoulder pre-roll: left the shoulder lane after {:.1f} m".format(travelled))
        plan.append((cur, RoadOption.LANEFOLLOW))
        travelled += step_m
    return plan


def _median_shoulder_waypoint(ego_lane_wp, forward_m):
    """Walk forward on the ego's own lane, then step left onto the median
    shoulder -- the suv's spawn side, matching the source ('leaves the
    roadway to the left...onto the grassy shoulder/median')."""
    forward_wp = _advance(ego_lane_wp, forward_m)
    shoulder = forward_wp.get_left_lane()
    if shoulder is None or shoulder.lane_type != carla.LaneType.Shoulder:
        raise ValueError("requires a median shoulder lane inboard of the ego's own lane")
    return shoulder


def _ego_relative_target(ego, gap_ahead_m):
    """Ego-conditioned interaction clock (carla-scenario-design skill):."""
    ego_map = CarlaDataProvider.get_map()
    ego_wp = ego_map.get_waypoint(ego.get_location(), project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
    candidates = [item for item in ego_wp.next(gap_ahead_m)
                  if item.lane_id == ego_wp.lane_id
                  and item.lane_type == carla.LaneType.Driving
                  and not item.is_junction]
    if not candidates:
        raise RuntimeError("cannot project an ego-relative target {} m ahead".format(gap_ahead_m))
    return candidates[0].transform.location


def _emit_event(name, ego, actors, reason):
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "VehicleCrossesMedianIntoLaneTown13", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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
        super(EventMarker, self).__init__("B01EventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


class MedianLaunchCross(AtomicBehavior):
    """Bounded point-seek: crosses from the median shoulder into the ego's
    lane at an ego-relative landing point, then hands off (this atomic
    completes once close to the target or on timeout; a separate bounded
    WaypointFollower drives the post-landing continue phase)."""

    def __init__(self, actor, ego, gap_ahead_m, timeout_s, cruise_speed_mps=10.0):
        super(MedianLaunchCross, self).__init__("MedianLaunchCross", actor)
        self._ego = ego
        self._gap_ahead_m = gap_ahead_m
        self._timeout_s = timeout_s
        self._cruise_speed_mps = cruise_speed_mps
        self._target = None
        self._started = None
        self._steer = 0.0

    def initialise(self):
        self._started = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        self._target = _ego_relative_target(self._ego, self._gap_ahead_m)
        self._actor.set_light_state(
            carla.VehicleLightState(
                carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker))

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
        steer_target = max(-0.55, min(0.55, 0.8 * yaw_error))
        self._steer = max(self._steer - 0.05, min(self._steer + 0.05, steer_target))
        velocity = self._actor.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        throttle = max(0.0, min(0.55, 0.20 + 0.05 * (self._cruise_speed_mps - speed)))
        self._actor.apply_control(carla.VehicleControl(
            throttle=throttle, brake=0.0, steer=self._steer))
        if elapsed >= self._timeout_s or remaining < 2.0:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        super(MedianLaunchCross, self).terminate(new_status)


class VehicleCrossesMedianIntoLaneTown13(BasicScenario):
    timeout = 42

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=42):
        self._map = CarlaDataProvider.get_map()
        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._trigger_wp is None or self._trigger_wp.is_junction
                or self._trigger_wp.road_id != ROAD_ID):
            raise ValueError("requires verified Town04 road {}".format(ROAD_ID))
        _median_shoulder_waypoint(self._trigger_wp, 1.0)  # fails loud if the median shoulder isn't there
        _outboard_escape_driving(self._trigger_wp)

        self._arm_distance_m = _value(config, "arm_distance_m", 14.0, 10.0, 20.0)
        self._suv_spawn_m = _value(config, "suv_spawn_distance_m", 30.0, 18.0, 34.0)
        self._suv_gap_ahead_m = _value(config, "suv_gap_ahead_m", 14.0, 8.0, 20.0)
        self._suv_cross_speed_mps = _value(config, "suv_cross_speed_mps", 10.0, 7.0, 13.0)
        self._suv_continue_speed_mps = _value(config, "suv_continue_speed_mps", 4.5, 3.0, 6.0)
        self._suv_launch_timeout_s = _value(config, "suv_launch_timeout_s", 5.5, 4.0, 8.0)
        self._max_exposure_s = _value(config, "max_exposure_s", 12.0, 8.0, 18.0)
        self._suv_preroll_speed_mps = _value(config, "suv_preroll_speed_mps", 6.0, 4.0, 8.0)

        self._suv = None
        self._suv_preroll_plan = None
        super(VehicleCrossesMedianIntoLaneTown13, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        suv_wp = _median_shoulder_waypoint(self._trigger_wp, self._suv_spawn_m)
        suv_spawn_transform = carla.Transform(
            carla.Location(x=suv_wp.transform.location.x, y=suv_wp.transform.location.y,
                           z=suv_wp.transform.location.z + 0.3),
            suv_wp.transform.rotation)
        self._suv = CarlaDataProvider.request_new_actor(
            "vehicle.dodge.charger_2020", suv_spawn_transform,
            rolename="scenario.pursuit_suv", color="235,235,225")
        if self._suv is None:
            raise RuntimeError("suv spawn failed")
        self.other_actors.append(self._suv)
        self._suv_preroll_plan = _shoulder_preroll_plan(suv_wp, 44.0)

    def _create_behavior(self):
        arm = py_trees.composites.Parallel(
            "B01Approach", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self._suv, self._arm_distance_m, name="B01DistanceArm"))
        arm.add_child(DriveDistance(self.ego_vehicles[0], 72.0, name="B01ProgressFallback"))
        arm.add_child(WaypointFollower(
            self._suv, self._suv_preroll_speed_mps, plan=self._suv_preroll_plan,
            name="B01SuvPreRoll"))

        event = py_trees.composites.Sequence("B01LaunchEvent")
        event.add_child(arm)
        event.add_child(EventMarker("armed", self.ego_vehicles[0], [self._suv],
                                       "ego-relative arm condition satisfied"))
        event.add_child(EventMarker("committed", self.ego_vehicles[0], [self._suv],
                                       "suv committed to the median-launch cross"))
        event.add_child(EventMarker("conflict_entered", self.ego_vehicles[0], [self._suv],
                                       "median-launch cross began"))
        event.add_child(MedianLaunchCross(
            self._suv, self.ego_vehicles[0], self._suv_gap_ahead_m,
            self._suv_launch_timeout_s, self._suv_cross_speed_mps))
        event.add_child(EventMarker("cleared", self.ego_vehicles[0], [self._suv],
                                       "suv committed to ego's lane and continued"))
        # The suv never freezes: it keeps moving forward in its landing lane
        # for the remainder of the exposure window (bounded resume), at a
        # speed clearly below the scripted reference's fixed ~8.33 m/s
        # (30 km/h) cruise so a non-reactive driver that does not brake
        # keeps closing the small ego-relative gap left by the cross,
        # rather than the suv pulling away.
        continue_and_clear = py_trees.composites.Parallel(
            "B01ContinueOrClear", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        continue_and_clear.add_child(WaypointFollower(
            self._suv, self._suv_continue_speed_mps, name="B01SuvContinue"))
        continue_and_clear.add_child(DriveDistance(self.ego_vehicles[0], 85.0, name="B01EgoClears"))
        continue_and_clear.add_child(ScenarioTimeout(
            self._max_exposure_s, self.__class__.__name__, name="B01ExposureTimeout"))
        event.add_child(continue_and_clear)
        return event

    def _create_test_criteria(self):
        return [
            CollisionTest(self.ego_vehicles[0], other_actor=self._suv,
                          terminate_on_failure=True, name="B01EgoSuvCollision"),
            ActorBlockedTest(self.ego_vehicles[0], min_speed=5.0, max_time=8.0,
                             terminate_on_failure=True, name="B01Blocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
