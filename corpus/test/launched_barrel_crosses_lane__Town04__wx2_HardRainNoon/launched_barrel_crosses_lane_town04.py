"""Launched barrel crosses lane (Town04).

A barrel is launched from a pickup in the adjacent lane and tumbles across the ego's lane.
"""
from __future__ import print_function

import json
import math
import os

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy, ActorTransformSetter, AtomicBehavior, ScenarioTimeout, WaypointFollower)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest, CollisionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance, InTriggerDistanceToLocation, InTriggerDistanceToVehicle)
from srunner.scenarios.basic_scenario import BasicScenario


def _emit_event(name, ego, actors, reason):
    """Scenario helper."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "LaunchedBarrelCrossesLaneTown04", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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


def _emit_debris_event(name, ego, debris, extra):
    """the acceptance criterion step 1 (additive instrumentation): like `_emit_event`, but for."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "LaunchedBarrelCrossesLaneTown04", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
                "frame": snapshot.frame, "simulation_time": snapshot.timestamp.elapsed_seconds,
                "actors": [{"id": debris.id, "role_name": debris.attributes.get("role_name", ""),
                            "alive": debris.is_alive}] if debris is not None else []}
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
        super(EventMarker, self).__init__("B10EventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


def _param(config, name, default, low, high):
    value = float(config.other_parameters.get(name, {}).get("value", default))
    if value < low or value > high:
        raise ValueError("{} outside [{}, {}]".format(name, low, high))
    return value


def _ahead(waypoint, distance_m):
    choices = [item for item in waypoint.next(distance_m)
               if item.road_id == waypoint.road_id
               and item.lane_id == waypoint.lane_id
               and item.lane_type == carla.LaneType.Driving
               and not item.is_junction]
    if not choices:
        raise ValueError("lacks {} m of driving lane ahead".format(distance_m))
    return choices[0]


def _median_lane(waypoint):
    """Same-direction driving lane one step toward the median (get_left_lane)."""
    candidate = waypoint.get_left_lane()
    if (candidate is None or candidate.lane_type != carla.LaneType.Driving
            or candidate.road_id != waypoint.road_id
            or candidate.lane_id * waypoint.lane_id <= 0):
        raise ValueError("needs a median-side same-direction driving lane")
    return candidate


class ProjectileLaunch(AtomicBehavior):
    """A bounded rigid-body force/torque impulse launches the debris across
    ego's corridor, then it coasts to rest under natural physics -- the same
    class of technique validated for 's wheel surrogate, applied here with
    an independent trigger cause and geometry."""

    def __init__(self, actor, across_vector, forward_vector, duration_s=2.6):
        super(ProjectileLaunch, self).__init__("ProjectileLaunch", actor)
        self._across = across_vector
        self._forward = forward_vector
        self._duration_s = duration_s
        self._start_s = None

    def initialise(self):
        self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds

    def update(self):
        elapsed_s = (CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
                     - self._start_s)
        if elapsed_s < 0.75:
            ramp = min(1.0, elapsed_s / 0.3)
            self._actor.add_force(carla.Vector3D(
                x=ramp * (2950.0 * self._across.x + 950.0 * self._forward.x),
                y=ramp * (2950.0 * self._across.y + 950.0 * self._forward.y),
                z=0.0))
            self._actor.add_torque(carla.Vector3D(
                x=985.0 * ramp * self._forward.x,
                y=985.0 * ramp * self._forward.y, z=0.0))
        return (py_trees.common.Status.SUCCESS if elapsed_s >= self._duration_s
                else py_trees.common.Status.RUNNING)


class DebrisTelemetry(AtomicBehavior):
    """the acceptance criterion step 1 (additive instrumentation only, no behavior change --."""

    SETTLE_SPEED_MPS = 0.3
    SETTLE_HOLD_S = 0.5
    SAMPLE_INTERVAL_S = 0.5

    def __init__(self, ego, actor, shared_state):
        super(DebrisTelemetry, self).__init__("DebrisTelemetry", actor)
        self._ego = ego
        self._state = shared_state

    def update(self):
        if self._actor is None or not self._actor.is_alive:
            return py_trees.common.Status.RUNNING
        now = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        velocity = self._actor.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        transform = self._actor.get_transform()
        world_map = CarlaDataProvider.get_map()
        wp = world_map.get_waypoint(transform.location, project_to_road=True,
                                     lane_type=carla.LaneType.Any)

        last_sample_s = self._state.get("last_sample_s")
        if last_sample_s is None or (now - last_sample_s) >= self.SAMPLE_INTERVAL_S:
            self._state["last_sample_s"] = now
            _emit_debris_event("debris_track", self._ego, self._actor, {
                "x": transform.location.x, "y": transform.location.y, "z": transform.location.z,
                "speed_mps": speed,
                "road_id": wp.road_id if wp else None,
                "lane_id": wp.lane_id if wp else None,
                "is_junction": bool(wp.is_junction) if wp else None,
            })

        if speed < self.SETTLE_SPEED_MPS:
            if self._state.get("settle_since_s") is None:
                self._state["settle_since_s"] = now
            elif (not self._state.get("settled")
                    and (now - self._state["settle_since_s"]) >= self.SETTLE_HOLD_S):
                self._state["settled"] = True
                _emit_debris_event("debris_settled", self._ego, self._actor, {
                    "x": transform.location.x, "y": transform.location.y, "z": transform.location.z,
                    "road_id": wp.road_id if wp else None,
                    "lane_id": wp.lane_id if wp else None,
                    "lane_type": str(wp.lane_type) if wp else None,
                    "is_junction": bool(wp.is_junction) if wp else None,
                })
        else:
            self._state["settle_since_s"] = None
        return py_trees.common.Status.RUNNING


class LaunchedBarrelCrossesLaneTown04(BasicScenario):
    timeout = 48

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=48):
        world_map = CarlaDataProvider.get_map()
        self._trigger_wp = world_map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if self._trigger_wp is None or self._trigger_wp.is_junction:
            raise ValueError("trigger must be a non-junction driving waypoint")
        self._pickup_lane = _median_lane(self._trigger_wp)
        self._pickup_spawn_m = _param(config, "pickup_spawn_distance_m", 20.0, 14.0, 30.0)
        self._debris_distance_m = _param(config, "debris_distance_m", 38.0, 28.0, 50.0)
        self._arm_m = _param(config, "arm_distance_m", 14.0, 10.0, 20.0)
        self._pickup_speed = _param(config, "pickup_speed_mps", 9.5, 6.0, 14.0)
        self._impact_gap_m = _param(config, "impact_gap_m", 5.0, 3.0, 8.0)
        self._max_exposure_s = _param(config, "max_exposure_s", 12.0, 8.0, 18.0)
        self._pickup = None
        self._debris = None
        self._debris_target = None
        self._debris_telemetry_state = {}
        super(LaunchedBarrelCrossesLaneTown04, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        pickup_wp = _ahead(self._pickup_lane, self._pickup_spawn_m)
        self._pickup = CarlaDataProvider.request_new_actor(
            "vehicle.nissan.patrol_2021", pickup_wp.transform,
            rolename="scenario.pickup", color="205,60,15")
        if self._pickup is None:
            raise RuntimeError("pickup spawn failed")
        self.other_actors.append(self._pickup)

        debris_wp = _ahead(self._pickup_lane, self._debris_distance_m)
        transform = debris_wp.transform
        self._debris_target = carla.Transform(
            carla.Location(x=transform.location.x, y=transform.location.y,
                           z=transform.location.z + 0.35),
            carla.Rotation(pitch=90.0, yaw=transform.rotation.yaw, roll=0.0))
        hidden = carla.Transform(carla.Location(
            x=self._debris_target.location.x, y=self._debris_target.location.y,
            z=self._debris_target.location.z - 35.0), self._debris_target.rotation)
        self._debris = CarlaDataProvider.request_new_actor(
            "static.prop.barrel", hidden, rolename="scenario.projectile")
        if self._debris is None:
            raise RuntimeError("debris surrogate spawn failed")
        self._debris.set_simulate_physics(False)
        self.other_actors.append(self._debris)

    def _create_behavior(self):
        arm = py_trees.composites.Parallel(
            "B10Approach", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self._pickup, self._arm_m, name="B10DistanceArm"))
        arm.add_child(DriveDistance(self.ego_vehicles[0], 14.0, name="B10ProgressArm"))
        arm.add_child(WaypointFollower(self._pickup, min(2.0, self._pickup_speed),
                                       name="B10PickupIdleCreep"))

        closing = py_trees.composites.Parallel(
            "B10Closing", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        closing.add_child(InTriggerDistanceToLocation(
            self._pickup, self._debris_target.location, self._impact_gap_m,
            name="B10ImpactGapReached"))
        closing.add_child(ScenarioTimeout(8.0, self.__class__.__name__, name="B10ClosingSafety"))
        closing.add_child(WaypointFollower(self._pickup, self._pickup_speed, name="B10PickupHolds"))

        event = py_trees.composites.Sequence("B10ProjectileEvent")
        event.add_child(arm)
        event.add_child(EventMarker("armed", self.ego_vehicles[0], [self._pickup, self._debris],
                                       "ego-relative arm condition satisfied"))
        event.add_child(closing)
        event.add_child(EventMarker("committed", self.ego_vehicles[0], [self._pickup, self._debris],
                                       "pickup reached impact gap, projectile launch committed"))
        event.add_child(ActorTransformSetter(
            self._debris, self._debris_target, physics=True, name="B10ActivateDebris"))
        event.add_child(EventMarker("conflict_entered", self.ego_vehicles[0], [self._pickup, self._debris],
                                       "debris projectile launch across ego's corridor began"))

        pickup_lane_transform = self._pickup_lane.transform
        incident = py_trees.composites.Parallel(
            "B10LaunchAndContinue", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        incident.add_child(ProjectileLaunch(
            self._debris, pickup_lane_transform.get_right_vector(),
            pickup_lane_transform.get_forward_vector()))
        incident.add_child(WaypointFollower(self._pickup, self._pickup_speed, name="B10PickupContinuesOn"))
        incident.add_child(DebrisTelemetry(
            self.ego_vehicles[0], self._debris, self._debris_telemetry_state))
        event.add_child(incident)

        clear = py_trees.composites.Parallel(
            "B10ClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(self.ego_vehicles[0], 95.0, name="B10EgoClears"))
        clear.add_child(ScenarioTimeout(
            self._max_exposure_s, self.__class__.__name__, name="B10ExposureTimeout"))
        clear.add_child(WaypointFollower(self._pickup, self._pickup_speed, name="B10PickupCreepResume"))
        clear.add_child(DebrisTelemetry(
            self.ego_vehicles[0], self._debris, self._debris_telemetry_state))
        event.add_child(clear)
        event.add_child(EventMarker("cleared", self.ego_vehicles[0], [self._pickup, self._debris],
                                       "ego cleared the bounded exposure window"))
        event.add_child(ActorDestroy(self._debris, name="B10DestroyDebris"))
        event.add_child(ActorDestroy(self._pickup, name="B10DestroyPickup"))
        return event

    def _create_test_criteria(self):
        return [
            CollisionTest(self.ego_vehicles[0], terminate_on_failure=True,
                          name="B10EgoCollision"),
            ActorBlockedTest(self.ego_vehicles[0], min_speed=5.0, max_time=8.0,
                             terminate_on_failure=True, name="B10ProjectileBlocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
