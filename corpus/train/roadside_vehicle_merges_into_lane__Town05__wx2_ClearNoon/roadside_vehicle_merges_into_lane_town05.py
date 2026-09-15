"""Roadside vehicle merges into lane (Town05).

A vehicle waiting at the roadside lurches forward and merges into the ego's lane.
"""
from __future__ import print_function

import math

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    AtomicBehavior, ScenarioTimeout)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance, InTriggerDistanceToVehicle)
from srunner.scenarios.basic_scenario import BasicScenario


def _number(config, name, default, low, high):
    value = float(config.other_parameters.get(name, {}).get("value", default))
    if value < low or value > high:
        raise ValueError("{}={} outside [{}, {}]".format(name, value, low, high))
    return value


def _follow_road(waypoint, distance_m):
    candidates = [candidate for candidate in waypoint.next(distance_m)
                  if candidate.road_id == waypoint.road_id
                  and candidate.lane_id == waypoint.lane_id
                  and candidate.lane_type == carla.LaneType.Driving
                  and not candidate.is_junction]
    if not candidates:
        raise ValueError("service road ends before {} m".format(distance_m))
    return candidates[0]


def _offset_transform(waypoint, right_offset_m):
    transform = waypoint.transform
    right = transform.get_right_vector()
    return carla.Transform(
        carla.Location(
            x=transform.location.x + right.x * right_offset_m,
            y=transform.location.y + right.y * right_offset_m,
            z=transform.location.z + 0.10),
        carla.Rotation(pitch=transform.rotation.pitch,
                       yaw=transform.rotation.yaw,
                       roll=transform.rotation.roll))


class BurnoutThenLurch(AtomicBehavior):
    """Explicit brake-held throttle cue, then a bounded physical lane entry."""

    def __init__(self, actor, target, burnout_s, timeout_s):
        super(BurnoutThenLurch, self).__init__("BurnoutThenLurch", actor)
        self._target = target
        self._burnout_s = burnout_s
        self._timeout_s = timeout_s
        self._start_s = None
        self._last_steer = 0.0

    def initialise(self):
        self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        self._actor.set_light_state(carla.VehicleLightState.Brake)

    def update(self):
        now_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        elapsed_s = now_s - self._start_s
        if elapsed_s < self._burnout_s:
            self._actor.apply_control(carla.VehicleControl(
                throttle=0.68, brake=0.78, steer=0.0))
            return py_trees.common.Status.RUNNING

        self._actor.set_light_state(carla.VehicleLightState.LeftBlinker)
        transform = self._actor.get_transform()
        dx = self._target.x - transform.location.x
        dy = self._target.y - transform.location.y
        distance_m = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        current_yaw = math.radians(transform.rotation.yaw)
        yaw_error = (desired_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi
        desired_steer = max(-0.55, min(0.55, 0.95 * yaw_error))
        steer = max(self._last_steer - 0.05,
                    min(self._last_steer + 0.05, desired_steer))
        self._last_steer = steer
        velocity = self._actor.get_velocity()
        speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        target_speed_mps = 8.0
        throttle = max(0.0, min(0.62, 0.20 + 0.07 * (target_speed_mps - speed_mps)))
        brake = max(0.0, min(0.35, 0.10 * (speed_mps - target_speed_mps)))
        if brake > 0.02:
            throttle = 0.0
        self._actor.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=steer))
        if distance_m < 1.8 or elapsed_s >= self._timeout_s:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._actor.is_alive:
            self._actor.apply_control(carla.VehicleControl(brake=0.45))
            self._actor.set_light_state(carla.VehicleLightState.Brake)
        super(BurnoutThenLurch, self).terminate(new_status)


class RoadsideVehicleMergesIntoLaneTown05(BasicScenario):
    timeout = 35

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=35):
        self._map = CarlaDataProvider.get_map()
        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._trigger_wp is None or self._trigger_wp.is_junction
                or self._trigger_wp.road_id != 19):
            raise ValueError("requires verified Town05 service road 19")
        self._actor_spawn_m = _number(config, "actor_spawn_distance_m", 40.0, 30.0, 48.0)
        self._shoulder_offset_m = _number(config, "shoulder_offset_m", 3.6, 3.2, 4.3)
        self._arm_distance_m = _number(config, "arm_distance_m", 22.0, 14.0, 30.0)
        self._burnout_s = _number(config, "burnout_duration_s", 0.8, 0.4, 1.5)
        self._target_m = _number(config, "lurch_target_distance_m", 55.0, 48.0, 65.0)
        self._max_exposure_s = _number(config, "max_exposure_s", 10.0, 6.0, 15.0)
        self._actor = None
        super(RoadsideVehicleMergesIntoLaneTown05, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        shoulder_wp = _follow_road(self._trigger_wp, self._actor_spawn_m)
        spawn_transform = _offset_transform(shoulder_wp, self._shoulder_offset_m)
        self._actor = CarlaDataProvider.request_new_actor(
            "vehicle.tesla.model3", spawn_transform,
            rolename="scenario.sedan", color="245,245,245")
        if self._actor is None:
            raise RuntimeError("shoulder sedan spawn failed")
        self._actor.apply_control(carla.VehicleControl(brake=0.55, hand_brake=True))
        self.other_actors.append(self._actor)

    def _create_behavior(self):
        arm = py_trees.composites.Parallel(
            "A04EgoRelativeArm", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self._actor, self._arm_distance_m,
            name="A04DistanceArm"))
        arm.add_child(DriveDistance(
            self.ego_vehicles[0], 24.0, name="A04ProgressFallback"))
        target = _follow_road(self._trigger_wp, self._target_m).transform.location
        event = py_trees.composites.Sequence("A04EventPhases")
        event.add_child(arm)
        event.add_child(BurnoutThenLurch(
            self._actor, target, self._burnout_s, 5.5))
        clear = py_trees.composites.Parallel(
            "A04ClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(
            self.ego_vehicles[0], 55.0, name="A04EgoClears"))
        clear.add_child(ScenarioTimeout(
            self._max_exposure_s, self.__class__.__name__, name="A04ExposureTimeout"))
        event.add_child(clear)
        return event

    def _create_test_criteria(self):
        return [CollisionTest(
            self.ego_vehicles[0], other_actor=self._actor,
            terminate_on_failure=True, name="A04EgoActorCollision")]

    def __del__(self):
        self.remove_all_actors()
