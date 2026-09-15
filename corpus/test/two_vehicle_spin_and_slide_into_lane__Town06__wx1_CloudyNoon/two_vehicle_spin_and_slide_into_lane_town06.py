"""Two vehicle spin and slide into lane (Town06).

Two vehicles lose traction in sequence and slide across the lanes into the ego's path.
"""
from __future__ import print_function

import math
import json
import os

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    AtomicBehavior, ScenarioTimeout, WaypointFollower)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest, CollisionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance, InTriggerDistanceToVehicle)
from srunner.scenarios.basic_scenario import BasicScenario

ROAD_ID = 37


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


def _left_driving(waypoint):
    candidate = waypoint.get_left_lane()
    if (candidate is None or candidate.lane_type != carla.LaneType.Driving
            or candidate.road_id != waypoint.road_id
            or candidate.lane_id * waypoint.lane_id <= 0):
        raise ValueError("requires an inboard (left) adjacent driving lane")
    return candidate


def _right_driving(waypoint, hops=1):
    candidate = waypoint
    for _ in range(hops):
        candidate = candidate.get_right_lane()
        if (candidate is None or candidate.lane_type != carla.LaneType.Driving
                or candidate.road_id != waypoint.road_id
                or candidate.lane_id * waypoint.lane_id <= 0):
            raise ValueError("requires {} outboard (right) adjacent driving lane(s)".format(hops))
    return candidate


def _emit_event(name, ego, actors, reason):
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "TwoVehicleSpinAndSlideIntoLaneTown06", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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
        super(EventMarker, self).__init__("A09EventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


class SustainedClearAmbient(AtomicBehavior):
    """Reissues `LeaveSpaceInFront`'s blackboard instruction every tick,."""

    def __init__(self, space, name="SustainedClearAmbient"):
        super(SustainedClearAmbient, self).__init__(name)
        self._space = space

    def update(self):
        py_trees.blackboard.Blackboard().set("BA_LeaveSpaceInFront", self._space, overwrite=True)
        return py_trees.common.Status.RUNNING


class TractionLossSlide(AtomicBehavior):
    """Bounded steering-loss point-seek: spin bias, then track toward a target."""

    def __init__(self, actor, target_location, timeout_s, cruise_speed_mps=8.0):
        super(TractionLossSlide, self).__init__("TractionLossSlide", actor)
        self._target = target_location
        self._timeout_s = timeout_s
        self._cruise_speed_mps = cruise_speed_mps
        self._started = None
        self._steer = 0.0

    def initialise(self):
        self._started = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
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
        spin_bias = 0.30 if elapsed < 0.9 else (-0.22 if elapsed < 2.4 else 0.0)
        wanted_steer = max(-0.6, min(0.6, 0.85 * yaw_error + spin_bias))
        self._steer = max(self._steer - 0.04, min(self._steer + 0.04, wanted_steer))
        velocity = self._actor.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        near_target = remaining < 2.6
        near_timeout = elapsed > (self._timeout_s - 1.0)
        brake = 0.25 if (near_target or near_timeout) else 0.0
        throttle = (max(0.0, min(0.5, 0.15 + 0.05 * (self._cruise_speed_mps - speed)))
                    if not brake else 0.0)
        self._actor.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=self._steer))
        if elapsed >= self._timeout_s or (elapsed > 2.0 and remaining < 2.6):
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._actor.is_alive:
            self._actor.apply_control(carla.VehicleControl(brake=0.5))
        super(TractionLossSlide, self).terminate(new_status)


class TwoVehicleSpinAndSlideIntoLaneTown06(BasicScenario):
    timeout = 48

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=48):
        self._map = CarlaDataProvider.get_map()
        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._trigger_wp is None or self._trigger_wp.is_junction
                or self._trigger_wp.road_id != ROAD_ID):
            raise ValueError("requires verified Town06 road {}".format(ROAD_ID))
        self._inner_lane_wp = _left_driving(self._trigger_wp)
        self._outer_lane_wp = _right_driving(self._trigger_wp, hops=3)

        self._arm_distance_m = _value(config, "arm_distance_m", 14.0, 10.0, 20.0)
        self._sedan_spawn_m = _value(config, "sedan_spawn_distance_m", 18.0, 14.0, 45.0)
        self._sedan_target_m = _value(config, "sedan_target_distance_m", 13.0, 9.0, 40.0)
        self._sedan_timeout_s = _value(config, "sedan_slide_timeout_s", 5.0, 3.5, 7.0)
        self._suv_arm_distance_m = _value(config, "suv_arm_distance_m", 16.0, 12.0, 22.0)
        self._suv_spawn_m = _value(config, "suv_spawn_distance_m", 70.0, 60.0, 85.0)
        self._suv_target_m = _value(config, "suv_target_distance_m", 58.0, 48.0, 68.0)
        self._suv_progress_fallback_m = _value(config, "suv_progress_fallback_m", 50.0, 40.0, 70.0)
        self._suv_timeout_s = _value(config, "suv_slide_timeout_s", 6.0, 4.0, 9.0)
        self._max_exposure_s = _value(config, "max_exposure_s", 12.0, 8.0, 18.0)

        self._sedan = None
        self._suv = None
        super(TwoVehicleSpinAndSlideIntoLaneTown06, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        sedan_wp = _advance(self._trigger_wp, self._sedan_spawn_m)
        suv_wp = _advance(self._outer_lane_wp, self._suv_spawn_m)
        self._sedan = CarlaDataProvider.request_new_actor(
            "vehicle.lincoln.mkz_2017", sedan_wp.transform,
            rolename="scenario.sedan", color="15,15,20")
        self._suv = CarlaDataProvider.request_new_actor(
            "vehicle.nissan.patrol_2021", suv_wp.transform,
            rolename="scenario.suv", color="180,30,30")
        if self._sedan is None or self._suv is None:
            raise RuntimeError("actor spawn failed")
        self.other_actors.extend([self._sedan, self._suv])

    def _create_behavior(self):
        sedan_target = _advance(self._inner_lane_wp, self._sedan_target_m).transform.location
        suv_target = _advance(self._trigger_wp, self._suv_target_m).transform.location

        arm_a = py_trees.composites.Parallel(
            "A09ArmSedan", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm_a.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self._sedan, self._arm_distance_m, name="A09SedanDistanceArm"))
        arm_a.add_child(DriveDistance(self.ego_vehicles[0], 26.0, name="A09SedanProgressFallback"))

        arm_b = py_trees.composites.Parallel(
            "A09ArmSuv", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm_b.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self._suv, self._suv_arm_distance_m, name="A09SuvDistanceArm"))
        arm_b.add_child(DriveDistance(
            self.ego_vehicles[0], self._suv_progress_fallback_m, name="A09SuvProgressFallback"))

        ego_location = self.ego_vehicles[0].get_location()
        ambient_clearance_m = ego_location.distance(suv_target) + 90.0 + 20.0

        event = py_trees.composites.Sequence("A09TwoPhaseSlide")
        event.add_child(arm_a)
        event.add_child(EventMarker("armed", self.ego_vehicles[0], [self._sedan],
                                       "ego-relative arm condition satisfied for sedan"))
        event.add_child(EventMarker("armed_sedan", self.ego_vehicles[0], [self._sedan],
                                       "ego-relative arm condition satisfied for sedan"))
        event.add_child(EventMarker("committed", self.ego_vehicles[0], [self._sedan],
                                       "sedan traction-loss slide committed"))
        event.add_child(EventMarker("conflict_entered", self.ego_vehicles[0], [self._sedan],
                                       "sedan traction-loss slide began"))
        event.add_child(EventMarker("conflict_entered_sedan", self.ego_vehicles[0], [self._sedan],
                                       "sedan traction-loss slide began"))
        event.add_child(TractionLossSlide(self._sedan, sedan_target, self._sedan_timeout_s))
        event.add_child(EventMarker("cleared_sedan", self.ego_vehicles[0], [self._sedan],
                                       "sedan slide phase complete"))

        event.add_child(arm_b)
        event.add_child(EventMarker("armed_suv", self.ego_vehicles[0], [self._suv],
                                       "ego-relative arm condition satisfied for suv"))
        event.add_child(EventMarker("conflict_entered_suv", self.ego_vehicles[0], [self._suv],
                                       "suv traction-loss slide began"))
        event.add_child(TractionLossSlide(self._suv, suv_target, self._suv_timeout_s))
        event.add_child(EventMarker("cleared_suv", self.ego_vehicles[0], [self._suv],
                                       "suv slide phase complete"))

        clear = py_trees.composites.Parallel(
            "A09ClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(self.ego_vehicles[0], 90.0, name="A09EgoClears"))
        clear.add_child(ScenarioTimeout(self._max_exposure_s, self.__class__.__name__,
                                        name="A09ExposureTimeout"))
        event.add_child(clear)

        root = py_trees.composites.Parallel(
            "A09RootWithSustainedAmbientClear", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(event)
        root.add_child(SustainedClearAmbient(ambient_clearance_m))
        return root

    def _create_test_criteria(self):
        return [
            CollisionTest(self.ego_vehicles[0], other_actor=self._sedan,
                          terminate_on_failure=True, name="A09EgoSedanCollision"),
            CollisionTest(self.ego_vehicles[0], other_actor=self._suv,
                          terminate_on_failure=True, name="A09EgoSuvCollision"),
            ActorBlockedTest(self.ego_vehicles[0], min_speed=5.0, max_time=8.0,
                             terminate_on_failure=True, name="A09Blocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
