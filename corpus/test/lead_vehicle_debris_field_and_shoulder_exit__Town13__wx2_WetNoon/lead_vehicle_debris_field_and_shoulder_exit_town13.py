"""Lead vehicle debris field and shoulder exit (Town13).

A high-speed SUV ahead scatters a field of debris across the lane, then exits to the shoulder.
"""
from __future__ import print_function

import math
import json
import os

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter, AtomicBehavior, ScenarioTimeout, WaypointFollower)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest, CollisionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance, InTriggerDistanceToVehicle)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront


def _read_value(config, name, default, minimum, maximum):
    value = float(config.other_parameters.get(name, {}).get("value", default))
    if not minimum <= value <= maximum:
        raise ValueError("parameter {}={} outside [{}, {}]".format(
            name, value, minimum, maximum))
    return value


def _road_step(waypoint, distance_m):
    options = waypoint.next(distance_m)
    options = [item for item in options if item.road_id == waypoint.road_id
               and item.lane_id == waypoint.lane_id
               and item.lane_type == carla.LaneType.Driving
               and not item.is_junction]
    if not options:
        raise ValueError("road ends before {} m".format(distance_m))
    return options[0]


def _emit_event(name, ego, actors, reason):
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "LeadVehicleDebrisFieldAndShoulderExitTown13", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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
        super(EventMarker, self).__init__("A02EventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


class SettleFrames(AtomicBehavior):
    def __init__(self, duration_s):
        super(SettleFrames, self).__init__("SettleFrames")
        self._duration_s = duration_s
        self._start_s = None

    def initialise(self):
        self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds

    def update(self):
        now_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        return (py_trees.common.Status.SUCCESS
                if now_s - self._start_s >= self._duration_s
                else py_trees.common.Status.RUNNING)


class SuvFailureExit(AtomicBehavior):
    """Short bounded weave followed by a fixed, map-relative shoulder exit."""

    def __init__(self, actor, shoulder_target, cruise_speed_mps, timeout_s):
        super(SuvFailureExit, self).__init__("SuvFailureExit", actor)
        self._target = shoulder_target
        self._cruise_speed_mps = cruise_speed_mps
        self._timeout_s = timeout_s
        self._start_s = None
        self._last_steer = 0.0

    def initialise(self):
        self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        lights = (carla.VehicleLightState.RightBlinker
                  | carla.VehicleLightState.Brake)
        self._actor.set_light_state(carla.VehicleLightState(lights))

    def update(self):
        now_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        elapsed_s = now_s - self._start_s
        transform = self._actor.get_transform()
        dx = self._target.x - transform.location.x
        dy = self._target.y - transform.location.y
        distance_m = math.hypot(dx, dy)
        if elapsed_s < 0.75:
            desired_steer = 0.08 * math.sin(elapsed_s * math.pi * 4.0)
        else:
            desired_yaw = math.atan2(dy, dx)
            current_yaw = math.radians(transform.rotation.yaw)
            error = (desired_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi
            desired_steer = max(-0.22, min(0.22, error * 0.45))
        steer = max(self._last_steer - 0.02,
                    min(self._last_steer + 0.02, desired_steer))
        self._last_steer = steer
        velocity = self._actor.get_velocity()
        speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        target_speed_mps = max(8.0, self._cruise_speed_mps - 1.6 * elapsed_s)
        error_mps = target_speed_mps - speed_mps
        throttle = max(0.0, min(0.45, 0.12 + 0.045 * error_mps))
        brake = max(0.0, min(0.48, -0.07 * error_mps))
        if brake > 0.02:
            throttle = 0.0
        self._actor.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=steer))
        if distance_m < 2.0 or elapsed_s >= self._timeout_s:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._actor.is_alive:
            self._actor.apply_control(carla.VehicleControl(brake=0.45))
        super(SuvFailureExit, self).terminate(new_status)


class LeadVehicleDebrisFieldAndShoulderExitTown13(BasicScenario):
    timeout = 45

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=45):
        self._map = CarlaDataProvider.get_map()
        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._trigger_wp is None or self._trigger_wp.is_junction
                or self._trigger_wp.road_id != 510):
            raise ValueError("requires the verified Town06 road 40 outer lane")
        self._suv_spawn_m = _read_value(config, "suv_spawn_distance_m", 50.0, 35.0, 65.0)
        self._arm_distance_m = _read_value(config, "arm_distance_m", 34.0, 25.0, 55.0)
        self._suv_speed_mps = _read_value(config, "suv_speed_mps", 26.0, 14.0, 30.0)
        self._debris_start_m = _read_value(config, "debris_start_distance_m", 50.0, 45.0, 70.0)
        self._debris_spacing_m = _read_value(config, "debris_spacing_m", 2.0, 1.0, 3.0)
        self._max_exposure_s = _read_value(config, "max_exposure_s", 12.0, 8.0, 20.0)
        self._suv = None
        self._debris = []
        self._debris_transforms = []
        super(LeadVehicleDebrisFieldAndShoulderExitTown13, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        suv_wp = _road_step(self._trigger_wp, self._suv_spawn_m)
        self._suv = CarlaDataProvider.request_new_actor(
            "vehicle.jeep.wrangler_rubicon", suv_wp.transform,
            rolename="scenario.suv", color="18,18,22")
        if self._suv is None:
            raise RuntimeError("SUV spawn failed")
        self.other_actors.append(self._suv)

        debris_layout = tuple(
            (row, fraction)
            for row in (0.0, 1.0, 2.0)
            for fraction in (-1.05, -0.80, -0.55, -0.30, -0.05, 0.20, 0.45)
        )
        for index, (row, fraction) in enumerate(debris_layout):
            road_wp = _road_step(
                self._trigger_wp, self._debris_start_m + row * self._debris_spacing_m)
            transform = road_wp.transform
            right = transform.get_right_vector()
            target = carla.Transform(
                carla.Location(
                    x=transform.location.x + right.x * road_wp.lane_width * fraction,
                    y=transform.location.y + right.y * road_wp.lane_width * fraction,
                    z=transform.location.z + 0.05),
                carla.Rotation(pitch=transform.rotation.pitch,
                               yaw=transform.rotation.yaw + 90.0 * (index - 1),
                               roll=transform.rotation.roll))
            hidden = carla.Transform(
                carla.Location(x=target.location.x, y=target.location.y,
                               z=target.location.z - 40.0), target.rotation)
            prop = CarlaDataProvider.request_new_actor(
                "static.prop.barrel", hidden, rolename="scenario.debris")
            if prop is None:
                raise RuntimeError("debris {} spawn failed".format(index))
            prop.set_simulate_physics(False)
            self.other_actors.append(prop)
            self._debris.append(prop)
            self._debris_transforms.append(target)

    def _create_behavior(self):
        arm = py_trees.composites.Parallel(
            "A02ApproachAndArm", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self._suv, self._arm_distance_m,
            name="A02DistanceArm"))
        arm.add_child(DriveDistance(
            self.ego_vehicles[0], 16.0, name="A02ProgressFallback"))
        arm.add_child(WaypointFollower(
            self._suv, self._suv_speed_mps, name="A02SuvApproach"))

        activate = py_trees.composites.Parallel(
            "A02ActivateDebris", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        for index, actor in enumerate(self._debris):
            activate.add_child(ActorTransformSetter(
                actor, self._debris_transforms[index], physics=True,
                name="A02ActivateDebris{}".format(index)))

        suv_wp = _road_step(self._trigger_wp, self._suv_spawn_m + 24.0)
        right = suv_wp.transform.get_right_vector()
        shoulder_target = carla.Location(
            x=suv_wp.transform.location.x + right.x * (suv_wp.lane_width * 0.72),
            y=suv_wp.transform.location.y + right.y * (suv_wp.lane_width * 0.72),
            z=suv_wp.transform.location.z)

        ego_location = self.ego_vehicles[0].get_location()
        ambient_clearance_m = ego_location.distance(suv_wp.transform.location) + 20.0

        event = py_trees.composites.Sequence("A02EventPhases")
        event.add_child(LeaveSpaceInFront(ambient_clearance_m, name="A02ClearAmbientApproach"))
        event.add_child(arm)
        event.add_child(EventMarker("armed", self.ego_vehicles[0],
                                       [self._suv] + self._debris, "ego-relative arm condition satisfied"))
        event.add_child(EventMarker("committed", self.ego_vehicles[0],
                                       [self._suv] + self._debris, "SUV failure maneuver committed"))
        event.add_child(activate)
        event.add_child(SettleFrames(0.20))
        event.add_child(EventMarker("conflict_entered", self.ego_vehicles[0],
                                       [self._suv] + self._debris, "debris and shoulder-exit conflict began"))
        event.add_child(SuvFailureExit(
            self._suv, shoulder_target, self._suv_speed_mps, 4.5))
        clear = py_trees.composites.Parallel(
            "A02ClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(
            self.ego_vehicles[0], 100.0, name="A02EgoClears"))
        clear.add_child(ScenarioTimeout(
            self._max_exposure_s, self.__class__.__name__, name="A02ExposureTimeout"))
        event.add_child(clear)
        event.add_child(EventMarker("cleared", self.ego_vehicles[0],
                                       [self._suv] + self._debris, "ego cleared the bounded exposure window"))
        return event

    def _create_test_criteria(self):
        return [
            CollisionTest(
                self.ego_vehicles[0], terminate_on_failure=True,
                name="A02EgoDebrisOrVehicleCollision"),
            ActorBlockedTest(
                self.ego_vehicles[0], min_speed=5.0, max_time=8.0,
                terminate_on_failure=True, name="A02DebrisFieldBlocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
