"""Braking lead vehicle with lane debris (Town12).

A heavy lead vehicle brakes hard while debris is strewn across the ego's lane.
"""
from __future__ import print_function

import json
import math
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


def _number(config, name, default, low, high):
    result = float(config.other_parameters.get(name, {}).get("value", default))
    if result < low or result > high:
        raise ValueError("{} outside [{}, {}]".format(name, low, high))
    return result


def _step(waypoint, distance_m):
    candidates = [item for item in waypoint.next(distance_m)
                  if item.road_id == waypoint.road_id
                  and item.lane_id == waypoint.lane_id
                  and item.lane_type == carla.LaneType.Driving
                  and not item.is_junction]
    if not candidates:
        raise ValueError("cannot advance {} m".format(distance_m))
    return candidates[0]


def _parallel_lane(waypoint):
    for candidate in (waypoint.get_right_lane(), waypoint.get_left_lane()):
        if (candidate is not None and candidate.lane_type == carla.LaneType.Driving
                and candidate.road_id == waypoint.road_id
                and candidate.lane_id * waypoint.lane_id > 0):
            return candidate
    raise ValueError("needs adjacent passenger-car traffic")


def _emit_event(name, ego, actors, reason):
    """Scenario helper."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "BrakingLeadVehicleWithLaneDebrisTown12Variant1", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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
        super(EventMarker, self).__init__("A06EventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


class PickupBlowoutDeceleration(AtomicBehavior):
    """Small destabilization followed by progressive braking in-lane."""

    def __init__(self, actor, timeout_s=5.0):
        super(PickupBlowoutDeceleration, self).__init__(
            "PickupBlowoutDeceleration", actor)
        self._timeout_s = timeout_s
        self._start_s = None
        self._brake = 0.0

    def initialise(self):
        self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        self._actor.set_light_state(carla.VehicleLightState.Brake)

    def update(self):
        elapsed = (CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
                   - self._start_s)
        steer = 0.10 * math.sin(min(elapsed, 1.4) * math.pi * 2.2) if elapsed < 1.4 else 0.0
        self._brake = min(0.72, self._brake + 0.035)
        self._actor.apply_control(carla.VehicleControl(steer=steer, brake=self._brake))
        velocity = self._actor.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        if speed < 0.25 or elapsed >= self._timeout_s:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._actor.is_alive:
            self._actor.apply_control(carla.VehicleControl(brake=0.65))
        super(PickupBlowoutDeceleration, self).terminate(new_status)


class BrakingLeadVehicleWithLaneDebrisTown12Variant1(BasicScenario):
    timeout = 45

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=45):
        world_map = CarlaDataProvider.get_map()
        self._trigger_wp = world_map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if self._trigger_wp is None or self._trigger_wp.is_junction:
            raise ValueError("trigger must be a non-junction driving waypoint")
        self._traffic_lane = _parallel_lane(self._trigger_wp)
        self._pickup_spawn = _number(config, "pickup_spawn_distance_m", 22.0, 17.0, 30.0)
        self._arm_m = _number(config, "arm_distance_m", 16.0, 12.0, 22.0)
        self._pickup_speed = _number(config, "pickup_speed_mps", 9.0, 6.0, 14.0)
        self._traffic_speed = _number(config, "traffic_speed_mps", 10.0, 7.0, 15.0)
        self._debris_start = _number(config, "debris_start_distance_m", 28.0, 24.0, 36.0)
        self._max_exposure_s = _number(config, "max_exposure_s", 12.0, 8.0, 18.0)
        self._pickup = None
        self._traffic = []
        self._debris = []
        self._debris_targets = []
        super(BrakingLeadVehicleWithLaneDebrisTown12Variant1, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        pickup_wp = _step(self._trigger_wp, self._pickup_spawn)
        self._pickup = CarlaDataProvider.request_new_actor(
            "vehicle.nissan.patrol", pickup_wp.transform,
            rolename="scenario.pickup", color="12,12,16")
        if self._pickup is None:
            raise RuntimeError("pickup spawn failed")
        self.other_actors.append(self._pickup)
        for index, distance in enumerate((18.0, 40.0)):
            traffic_wp = _step(self._traffic_lane, distance)
            model = "vehicle.seat.leon" if index == 0 else "vehicle.audi.a2"
            actor = CarlaDataProvider.request_new_actor(
                model, traffic_wp.transform, rolename="scenario.traffic")
            if actor is None:
                raise RuntimeError("adjacent traffic spawn failed")
            self._traffic.append(actor)
            self.other_actors.append(actor)

        layout = tuple((row, lateral) for row in (0.0, 1.8, 3.6)
                       for lateral in (-0.42, -0.21, 0.0, 0.21, 0.42))
        models = ("static.prop.dirtdebris01", "static.prop.dirtdebris02",
                  "static.prop.dirtdebris03")
        for index, (row, fraction) in enumerate(layout):
            debris_wp = _step(self._trigger_wp, self._debris_start + row)
            transform = debris_wp.transform
            right = transform.get_right_vector()
            target = carla.Transform(carla.Location(
                x=transform.location.x + right.x * debris_wp.lane_width * fraction,
                y=transform.location.y + right.y * debris_wp.lane_width * fraction,
                z=transform.location.z + 0.04),
                carla.Rotation(yaw=transform.rotation.yaw + (index % 3) * 37.0))
            hidden = carla.Transform(carla.Location(
                x=target.location.x, y=target.location.y, z=target.location.z - 35.0),
                target.rotation)
            prop = CarlaDataProvider.request_new_actor(
                models[index % len(models)], hidden, rolename="scenario.debris")
            if prop is None:
                raise RuntimeError("debris spawn failed")
            prop.set_simulate_physics(False)
            self._debris.append(prop)
            self._debris_targets.append(target)
            self.other_actors.append(prop)

    def _create_behavior(self):
        arm = py_trees.composites.Parallel(
            "A06Approach", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self._pickup, self._arm_m, name="A06DistanceArm"))
        arm.add_child(DriveDistance(self.ego_vehicles[0], 14.0, name="A06ProgressArm"))
        arm.add_child(WaypointFollower(self._pickup, self._pickup_speed, name="A06PickupApproach"))
        for index, actor in enumerate(self._traffic):
            arm.add_child(WaypointFollower(
                actor, self._traffic_speed, name="A06Traffic{}Approach".format(index)))

        activate = py_trees.composites.Parallel(
            "A06ReleaseDebris", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        for index, actor in enumerate(self._debris):
            activate.add_child(ActorTransformSetter(
                actor, self._debris_targets[index], physics=True,
                name="A06ReleaseDebris{}".format(index)))

        event = py_trees.composites.Sequence("A06BlowoutEvent")
        event.add_child(arm)
        event.add_child(EventMarker(
            "armed", self.ego_vehicles[0], [self._pickup] + self._traffic,
            "approach arm condition satisfied"))
        event.add_child(EventMarker(
            "committed", self.ego_vehicles[0], [self._pickup], "debris release committed"))
        event.add_child(activate)
        event.add_child(EventMarker(
            "conflict_entered", self.ego_vehicles[0], [self._pickup] + self._debris,
            "debris cloud released into ego corridor"))
        incident = py_trees.composites.Parallel(
            "A06PickupAndTraffic", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        incident.add_child(PickupBlowoutDeceleration(self._pickup))
        for index, actor in enumerate(self._traffic):
            incident.add_child(WaypointFollower(
                actor, self._traffic_speed, name="A06Traffic{}Continues".format(index)))
        event.add_child(incident)
        clear = py_trees.composites.Parallel(
            "A06ClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(self.ego_vehicles[0], 100.0, name="A06EgoClears"))
        clear.add_child(ScenarioTimeout(
            self._max_exposure_s, self.__class__.__name__, name="A06ExposureTimeout"))
        event.add_child(clear)
        event.add_child(EventMarker(
            "cleared", self.ego_vehicles[0], [self._pickup] + self._traffic,
            "ego cleared the bounded exposure window"))
        return event

    def _create_test_criteria(self):
        return [
            CollisionTest(self.ego_vehicles[0], terminate_on_failure=True,
                          name="A06EgoCollision"),
            ActorBlockedTest(self.ego_vehicles[0], min_speed=5.0, max_time=8.0,
                             terminate_on_failure=True, name="A06DebrisCloudBlocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
