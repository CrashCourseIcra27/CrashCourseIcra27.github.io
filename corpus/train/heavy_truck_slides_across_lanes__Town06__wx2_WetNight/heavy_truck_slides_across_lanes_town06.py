"""Heavy truck slides across lanes (Town06).

A high-profile truck ahead slides broadside across the adjacent lanes.
"""
from __future__ import print_function

import json
import math
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


def _param(config, name, default, low, high):
    value = float(config.other_parameters.get(name, {}).get("value", default))
    if value < low or value > high:
        raise ValueError("{} outside [{}, {}]".format(name, low, high))
    return value


def _forward(waypoint, distance_m):
    choices = [candidate for candidate in waypoint.next(distance_m)
               if candidate.road_id == waypoint.road_id
               and candidate.lane_id == waypoint.lane_id
               and candidate.lane_type == carla.LaneType.Driving
               and not candidate.is_junction]
    if not choices:
        raise ValueError("lacks {} m of straight driving lane".format(distance_m))
    return choices[0]


def _left_driving(waypoint, hops=1):
    candidate = waypoint
    for _ in range(hops):
        candidate = candidate.get_left_lane()
        if (candidate is None or candidate.lane_type != carla.LaneType.Driving
                or candidate.road_id != waypoint.road_id
                or candidate.lane_id * waypoint.lane_id <= 0):
            raise ValueError("requires {} inboard (left) adjacent driving lane(s)".format(hops))
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
    """Scenario helper."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "HeavyTruckSlidesAcrossLanesTown06", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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
        super(EventMarker, self).__init__("A07EventMarker_" + event, actors[0] if actors else ego)
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


class CommitOnTTC(AtomicBehavior):
    """SUCCESS once the ego's LIVE time-to-collision to a fixed location."""

    def __init__(self, ego, target_location, commit_ttc_s, floor_speed_mps=4.0,
                 name="CommitOnTTC"):
        super(CommitOnTTC, self).__init__(name, ego)
        self._ego = ego
        self._target = target_location
        self._commit_ttc_s = commit_ttc_s
        self._floor = floor_speed_mps

    def update(self):
        location = self._ego.get_location()
        distance_m = location.distance(self._target)
        velocity = self._ego.get_velocity()
        speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        ttc_s = distance_m / max(speed_mps, self._floor)
        if ttc_s <= self._commit_ttc_s:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class BroadsideSlide(AtomicBehavior):
    """Bounded steering-loss point-seek: spin bias, then track toward a."""

    def __init__(self, actor, target_location, timeout_s, cruise_speed_mps=8.5):
        super(BroadsideSlide, self).__init__("BroadsideSlide", actor)
        self._target = target_location
        self._timeout_s = timeout_s
        self._cruise_speed_mps = cruise_speed_mps
        self._started = None
        self._steer = 0.0

    def initialise(self):
        self._started = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        # carla.VehicleLightState has no `Hazard` member in this CARLA 0.9.15
        # Python API build -- LeftBlinker|RightBlinker together are the
        # standard hazard-flasher substitute, matching 's own convention
        # exactly (visual cue only, not read by any acceptance criterion).
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
        super(BroadsideSlide, self).terminate(new_status)


class HeavyTruckSlidesAcrossLanesTown06(BasicScenario):
    """A high-profile rigid truck replaces CARLA's unavailable travel trailer."""
    timeout = 45

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=45):
        world_map = CarlaDataProvider.get_map()
        self._trigger_wp = world_map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if self._trigger_wp is None or self._trigger_wp.is_junction:
            raise ValueError("trigger must be a non-junction driving waypoint")
        self._rig_lane = _right_driving(self._trigger_wp, hops=2)
        self._escape_lane = _left_driving(self._trigger_wp, hops=1)  # verified only; not driven
        self._rig_spawn_m = _param(config, "rig_spawn_distance_m", 35.0, 25.0, 45.0)
        self._arm_m = _param(config, "arm_distance_m", 20.0, 15.0, 28.0)
        self._rig_speed = _param(config, "rig_speed_mps", 8.5, 5.0, 10.0)
        self._commit_ttc_s = _param(config, "commit_ttc_s", 2.25, 2.0, 2.5)
        self._slide_corridor_m = _param(config, "slide_corridor_distance_m", 35.0, 25.0, 45.0)
        self._slide_timeout_s = _param(config, "slide_timeout_s", 5.0, 3.5, 7.0)
        self._max_exposure_s = _param(config, "max_exposure_s", 14.0, 8.0, 20.0)
        self._rig = None
        super(HeavyTruckSlidesAcrossLanesTown06, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        rig_wp = _forward(self._rig_lane, self._rig_spawn_m)
        self._rig = CarlaDataProvider.request_new_actor(
            "vehicle.carlamotors.european_hgv", rig_wp.transform,
            rolename="scenario.high_profile_rig", color="175,175,180")
        if self._rig is None:
            raise RuntimeError("high-profile rig spawn failed")
        self.other_actors.append(self._rig)

    def _create_behavior(self):
        slide_corridor = _forward(
            self._trigger_wp, self._slide_corridor_m).transform.location

        arm = py_trees.composites.Parallel(
            "A07Approach", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self._rig, self._arm_m, name="A07DistanceArm"))
        arm.add_child(DriveDistance(self.ego_vehicles[0], 28.0, name="A07ProgressArm"))

        commit_fallback_m = max(self._slide_corridor_m - 8.0, 27.0)
        commit_gate = py_trees.composites.Parallel(
            "A07CommitGate", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        commit_gate.add_child(CommitOnTTC(
            self.ego_vehicles[0], slide_corridor, self._commit_ttc_s,
            name="A07LiveTTCCommit"))
        commit_gate.add_child(DriveDistance(
            self.ego_vehicles[0], commit_fallback_m, name="A07CommitProgressFallback"))

        ego_location = self.ego_vehicles[0].get_location()
        ambient_clearance_m = ego_location.distance(slide_corridor) + 100.0 + 20.0

        event = py_trees.composites.Sequence("A07WindEvent")
        event.add_child(arm)
        event.add_child(EventMarker(
            "armed", self.ego_vehicles[0], [self._rig], "approach arm condition satisfied"))
        event.add_child(commit_gate)
        event.add_child(EventMarker(
            "committed", self.ego_vehicles[0], [self._rig],
            "broadside slide commit: live ego TTC to the slide corridor reached target"))
        event.add_child(EventMarker(
            "conflict_entered", self.ego_vehicles[0], [self._rig],
            "rig broadside slide began"))
        event.add_child(BroadsideSlide(
            self._rig, slide_corridor, self._slide_timeout_s,
            cruise_speed_mps=self._rig_speed))
        clear = py_trees.composites.Parallel(
            "A07ClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(self.ego_vehicles[0], 100.0, name="A07EgoClears"))
        clear.add_child(ScenarioTimeout(self._max_exposure_s, self.__class__.__name__,
                                        name="A07ExposureTimeout"))
        event.add_child(clear)
        event.add_child(EventMarker(
            "cleared", self.ego_vehicles[0], [self._rig], "ego cleared the bounded exposure window"))

        root = py_trees.composites.Parallel(
            "A07RootWithSustainedAmbientClear", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(event)
        root.add_child(SustainedClearAmbient(ambient_clearance_m))
        return root

    def _create_test_criteria(self):
        return [
            CollisionTest(self.ego_vehicles[0], other_actor=self._rig,
                          terminate_on_failure=True, name="A07EgoRigCollision"),
            ActorBlockedTest(self.ego_vehicles[0], min_speed=5.0, max_time=8.0,
                             terminate_on_failure=True, name="A07RigWreckageBlocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
