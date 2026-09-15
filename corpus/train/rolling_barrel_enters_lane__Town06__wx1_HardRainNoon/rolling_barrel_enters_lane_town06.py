"""Rolling barrel enters lane (Town06).

A barrel breaks loose from traffic ahead and rolls into the ego's lane.
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
    DriveDistance)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront


def _value(config, name, default, low, high):
    value = float(config.other_parameters.get(name, {}).get("value", default))
    if value < low or value > high:
        raise ValueError("{} outside [{}, {}]".format(name, low, high))
    return value


def _ahead(waypoint, distance_m):
    options = [item for item in waypoint.next(distance_m)
               if item.road_id == waypoint.road_id
               and item.lane_id == waypoint.lane_id
               and item.lane_type == carla.LaneType.Driving
               and not item.is_junction]
    if not options:
        raise ValueError("road ends before {} m".format(distance_m))
    return options[0]


def _adjacent_same_direction(waypoint):
    for item in (waypoint.get_left_lane(), waypoint.get_right_lane()):
        if (item is not None and item.lane_type == carla.LaneType.Driving
                and item.road_id == waypoint.road_id
                and item.lane_id * waypoint.lane_id > 0):
            return item
    raise ValueError("requires an adjacent same-direction lane")


def _emit_event(name, ego, actors, reason):
    """Scenario helper."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "RollingBarrelEntersLaneTown06", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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


def _emit_rig_track(name, actor, extra):
    """Scenario helper."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "RollingBarrelEntersLaneTown06", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
                "frame": snapshot.frame, "simulation_time": snapshot.timestamp.elapsed_seconds,
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
        super(EventMarker, self).__init__("A05EventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


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
        _emit_rig_track("ttc_track", self._ego, {
            "distance_to_crossing_m": distance_m, "ego_speed_mps": speed_mps, "ttc_s": ttc_s,
            "ego_x": location.x, "ego_y": location.y})
        if ttc_s <= self._commit_ttc_s:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class GapKeepingFollower(WaypointFollower):
    """Scenario helper."""

    def __init__(self, actor, ego, origin, forward_unit, off_m,
                 base_speed_mps=9.0, gain=0.4, floor_mps=4.0, ceiling_mps=12.0,
                 name="GapKeepingFollower"):
        super(GapKeepingFollower, self).__init__(actor, base_speed_mps, name=name)
        self._ego = ego
        self._origin = origin
        self._forward = forward_unit
        self._off = off_m
        self._base = base_speed_mps
        self._gain = gain
        self._floor = floor_mps
        self._ceiling = ceiling_mps

    def _progress_m(self, actor):
        loc = actor.get_location()
        return ((loc.x - self._origin.x) * self._forward.x
                 + (loc.y - self._origin.y) * self._forward.y)

    def update(self):
        ego_prog = self._progress_m(self._ego)
        actor_prog = self._progress_m(self._actor)
        speed_mps = self._base + self._gain * ((ego_prog + self._off) - actor_prog)
        speed_mps = max(self._floor, min(self._ceiling, speed_mps))
        local_planner = self._local_planner_dict.get(self._actor)
        if local_planner is not None and local_planner != "Walker":
            local_planner.set_speed(speed_mps * 3.6)
        return super(GapKeepingFollower, self).update()


class RollingWheelSurrogate(AtomicBehavior):
    """Bounded, critically-damped force release toward the ego's lane, then."""

    def __init__(self, actor, across_vector, forward_vector, target_lateral_m,
                 duration_s=4.0, k=1500.0, c=1400.0, force_cap_n=2400.0, forward_force_n=300.0,
                 settle_pos_m=0.08, settle_vel_mps=0.05):
        super(RollingWheelSurrogate, self).__init__("RollingWheelSurrogate", actor)
        self._across = across_vector
        self._forward = forward_vector
        self._target_lateral_m = target_lateral_m
        self._duration_s = duration_s
        self._k = k
        self._c = c
        self._force_cap_n = force_cap_n
        self._forward_force_n = forward_force_n
        self._settle_pos_m = settle_pos_m
        self._settle_vel_mps = settle_vel_mps
        self._start_s = None
        self._start_loc = None
        self._converged = False

    def initialise(self):
        if self._start_loc is None:
            self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
            self._start_loc = self._actor.get_location()

    def update(self):
        if self._converged:
            return py_trees.common.Status.SUCCESS
        now_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        elapsed_s = now_s - self._start_s
        loc = self._actor.get_location()
        traveled = ((loc.x - self._start_loc.x) * self._across.x
                    + (loc.y - self._start_loc.y) * self._across.y)
        remaining = self._target_lateral_m - traveled
        velocity = self._actor.get_velocity()
        velocity_along = velocity.x * self._across.x + velocity.y * self._across.y
        lateral_force = self._k * remaining - self._c * velocity_along
        lateral_force = max(-self._force_cap_n, min(self._force_cap_n, lateral_force))
        forward_force = self._forward_force_n if elapsed_s < 1.0 else 0.0
        self._actor.add_force(carla.Vector3D(
            x=lateral_force * self._across.x + forward_force * self._forward.x,
            y=lateral_force * self._across.y + forward_force * self._forward.y,
            z=0.0))
        converged = abs(remaining) <= self._settle_pos_m and abs(velocity_along) <= self._settle_vel_mps
        _emit_rig_track("wheel_settle_track", self._actor, {
            "elapsed_s": elapsed_s, "remaining_m": remaining, "velocity_along_mps": velocity_along,
            "converged": bool(converged), "wheel_x": loc.x, "wheel_y": loc.y})
        if converged or elapsed_s >= self._duration_s:
            self._converged = True
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class WheelImpactResponse(AtomicBehavior):
    """Target hatchback yaws and brakes after the wheel enters its corridor."""

    def __init__(self, actor, across_vector, max_lateral_m=0.8, timeout_s=4.0):
        super(WheelImpactResponse, self).__init__("WheelImpactResponse", actor)
        self._across = across_vector
        self._max_lateral_m = max_lateral_m
        self._timeout_s = timeout_s
        self._start_s = None
        self._start_loc = None
        self._steer = 0.0
        self._brake = 0.0
        self._converged = False
        self._final_control_applied = False

    def initialise(self):
        if self._start_loc is None:
            self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
            self._start_loc = self._actor.get_location()
            self._actor.set_light_state(carla.VehicleLightState.Brake)

    def update(self):
        if self._converged:
            return py_trees.common.Status.SUCCESS
        now_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        elapsed_s = now_s - self._start_s
        loc = self._actor.get_location()
        traveled = ((loc.x - self._start_loc.x) * self._across.x
                    + (loc.y - self._start_loc.y) * self._across.y)
        remaining_budget = max(0.0, self._max_lateral_m - traveled)
        target_steer = min(0.38, 0.38 * remaining_budget / self._max_lateral_m)
        if target_steer > self._steer:
            self._steer = min(target_steer, self._steer + 0.035)
        else:
            self._steer = max(target_steer, self._steer - 0.07)
        self._brake = min(0.70, self._brake + 0.05)
        self._actor.apply_control(carla.VehicleControl(
            steer=self._steer, brake=self._brake))
        speed = self._actor.get_velocity()
        speed_mps = math.sqrt(speed.x ** 2 + speed.y ** 2 + speed.z ** 2)
        if speed_mps < 0.3 or elapsed_s >= self._timeout_s:
            self._converged = True
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if not self._final_control_applied:
            self._final_control_applied = True
            if self._actor.is_alive:
                self._actor.apply_control(carla.VehicleControl(brake=0.65, steer=0.25))
        super(WheelImpactResponse, self).terminate(new_status)


class RollingBarrelEntersLaneTown06(BasicScenario):
    timeout = 40

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=40):
        world_map = CarlaDataProvider.get_map()
        self._trigger_wp = world_map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if self._trigger_wp is None or self._trigger_wp.is_junction:
            raise ValueError("trigger must be a non-junction driving waypoint")
        self._pickup_lane = _adjacent_same_direction(self._trigger_wp)
        self._target_spawn_m = _value(config, "target_spawn_distance_m", 74.0, 55.0, 75.0)
        self._pickup_spawn_m = _value(config, "pickup_spawn_distance_m", 80.0, 60.0, 80.0)
        self._arm_progress_m = _value(config, "arm_distance_m", 25.0, 20.0, 32.0)
        self._commit_ttc_s = _value(config, "commit_ttc_s", 3.6, 3.0, 3.6)
        self._crossing_distance_m = _value(
            config, "wheel_crossing_distance_m", 80.0, 65.0, 80.0)
        self._target_speed = _value(config, "target_speed_mps", 9.0, 5.0, 11.0)
        self._pickup_speed = _value(config, "pickup_speed_mps", 9.0, 6.0, 13.0)
        self._max_exposure_s = _value(config, "max_exposure_s", 12.0, 8.0, 18.0)
        self._target = self._pickup = self._wheel = None
        self._wheel_target = None
        self._route_origin = self._trigger_wp.transform.location
        forward = self._trigger_wp.transform.get_forward_vector()
        forward_len = math.hypot(forward.x, forward.y)
        self._route_forward = carla.Vector3D(
            x=forward.x / forward_len, y=forward.y / forward_len, z=0.0)
        super(RollingBarrelEntersLaneTown06, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        target_wp = _ahead(self._pickup_lane, self._target_spawn_m)
        pickup_wp = _ahead(self._pickup_lane, self._pickup_spawn_m)
        self._target = CarlaDataProvider.request_new_actor(
            "vehicle.audi.a2", target_wp.transform,
            rolename="scenario.target", color="18,18,22")
        self._pickup = CarlaDataProvider.request_new_actor(
            "vehicle.nissan.patrol_2021", pickup_wp.transform,
            rolename="scenario.pickup", color="225,225,225")
        if self._target is None or self._pickup is None:
            raise RuntimeError("vehicle spawn failed")
        self.other_actors.extend([self._target, self._pickup])
        transform = pickup_wp.transform

        ego_lane_here = _ahead(self._trigger_wp, self._pickup_spawn_m)
        toward_ego_x = ego_lane_here.transform.location.x - transform.location.x
        toward_ego_y = ego_lane_here.transform.location.y - transform.location.y
        toward_ego_len = math.hypot(toward_ego_x, toward_ego_y)
        self._wheel_toward_ego = carla.Vector3D(
            x=toward_ego_x / toward_ego_len, y=toward_ego_y / toward_ego_len, z=0.0)
        wheel_location = carla.Location(
            x=transform.location.x + self._wheel_toward_ego.x * 1.5,
            y=transform.location.y + self._wheel_toward_ego.y * 1.5,
            z=transform.location.z + 0.55)
        self._wheel_target = carla.Transform(
            wheel_location, carla.Rotation(
                pitch=90.0, yaw=transform.rotation.yaw, roll=0.0))
        hidden = carla.Transform(carla.Location(
            x=wheel_location.x, y=wheel_location.y, z=wheel_location.z - 35.0),
            self._wheel_target.rotation)
        self._wheel = CarlaDataProvider.request_new_actor(
            "static.prop.barrel", hidden, rolename="scenario.wheel")
        if self._wheel is None:
            raise RuntimeError("wheel surrogate spawn failed")
        self._wheel.set_simulate_physics(False)
        self.other_actors.append(self._wheel)

    def _create_behavior(self):
        crossing_location = _ahead(
            self._trigger_wp, self._crossing_distance_m).transform.location
        commit_fallback_m = max(self._crossing_distance_m - 8.0, self._arm_progress_m + 5.0)

        approach_gate = py_trees.composites.Sequence("A05ArmThenCommit")
        approach_gate.add_child(DriveDistance(
            self.ego_vehicles[0], self._arm_progress_m, name="A05ProgressArm"))
        approach_gate.add_child(EventMarker(
            "armed", self.ego_vehicles[0], [self._target, self._pickup, self._wheel],
            "ego progress arm condition satisfied"))
        approach_gate.add_child(CommitOnTTC(
            self.ego_vehicles[0], crossing_location, self._commit_ttc_s,
            name="A05LiveTTCCommit"))

        approach_and_commit = py_trees.composites.Parallel(
            "A05ApproachAndCommit", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        approach_and_commit.add_child(approach_gate)
        approach_and_commit.add_child(DriveDistance(
            self.ego_vehicles[0], commit_fallback_m, name="A05CommitProgressFallback"))
        approach_and_commit.add_child(GapKeepingFollower(
            self._target, self.ego_vehicles[0], self._route_origin, self._route_forward,
            off_m=34.0, base_speed_mps=self._target_speed, name="A05TargetApproach"))
        approach_and_commit.add_child(GapKeepingFollower(
            self._pickup, self.ego_vehicles[0], self._route_origin, self._route_forward,
            off_m=42.0, base_speed_mps=self._pickup_speed, name="A05PickupApproach"))

        event = py_trees.composites.Sequence("A05WheelEvent")
        ego_location = self.ego_vehicles[0].get_location()
        ambient_clearance_m = ego_location.distance(crossing_location) + 40.0
        event.add_child(LeaveSpaceInFront(ambient_clearance_m, name="A05ClearAmbientApproach"))
        event.add_child(approach_and_commit)
        event.add_child(EventMarker(
            "committed", self.ego_vehicles[0], [self._wheel], "wheel detachment committed"))
        event.add_child(ActorTransformSetter(
            self._wheel, self._wheel_target, physics=True, name="A05DetachWheel"))
        event.add_child(EventMarker(
            "conflict_entered", self.ego_vehicles[0], [self._wheel, self._target],
            "wheel projectile entered ego corridor"))
        incident = py_trees.composites.Parallel(
            "A05WheelAndTarget", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        pickup_transform = self._pickup_lane.transform
        incident.add_child(RollingWheelSurrogate(
            self._wheel, self._wheel_toward_ego,
            pickup_transform.get_forward_vector(), target_lateral_m=2.0,
            duration_s=4.0, k=3750.0, c=2210.0, force_cap_n=6000.0))
        incident.add_child(WheelImpactResponse(self._target, self._wheel_toward_ego))
        pickup_continue = py_trees.composites.Parallel(
            "A05PickupContinueBounded", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        pickup_continue.add_child(WaypointFollower(
            self._pickup, self._pickup_speed, name="A05PickupContinues"))
        pickup_continue.add_child(DriveDistance(
            self._pickup, 20.0, name="A05PickupContinueDistance"))
        incident.add_child(pickup_continue)
        event.add_child(incident)
        clear = py_trees.composites.Parallel(
            "A05ClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(self.ego_vehicles[0], 45.0, name="A05EgoClears"))
        clear.add_child(ScenarioTimeout(
            self._max_exposure_s, self.__class__.__name__, name="A05ExposureTimeout"))
        event.add_child(clear)
        event.add_child(EventMarker(
            "cleared", self.ego_vehicles[0], [self._wheel, self._target, self._pickup],
            "ego cleared the bounded exposure window"))
        return event

    def _create_test_criteria(self):
        return [
            CollisionTest(self.ego_vehicles[0], terminate_on_failure=True,
                          name="A05EgoCollision"),
            ActorBlockedTest(self.ego_vehicles[0], min_speed=5.0, max_time=35.0,
                             terminate_on_failure=True, name="A05WreckageBlocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
