"""Oncoming vehicle head on intrusion (Town12).

A vehicle from the opposing lane crosses the centre line and drives head-on at the ego.
"""
from __future__ import print_function

import json
import math
import os

import carla
import py_trees
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy, ActorTransformSetter, AtomicBehavior, Idle, ScenarioTimeout,
    WaypointFollower)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest, CollisionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance, InTriggerDistanceToVehicle)
from srunner.scenarios.basic_scenario import BasicScenario

ROAD_ID = 740


def _value(config, name, default, low, high):
    result = float(config.other_parameters.get(name, {}).get("value", default))
    if result < low or result > high:
        raise ValueError("{} outside [{}, {}]".format(name, low, high))
    return result


def _speed(actor):
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _advance(waypoint, distance_m):
    """Walk lane -1's own `next()` — the KNOWN-good direction."""
    candidates = [item for item in waypoint.next(distance_m)
                  if item.road_id == waypoint.road_id
                  and item.lane_id == waypoint.lane_id
                  and item.lane_type == carla.LaneType.Driving
                  and not item.is_junction]
    if not candidates:
        raise ValueError("cannot advance {} m on road {} lane {}".format(
            distance_m, waypoint.road_id, waypoint.lane_id))
    return candidates[0]


def _opposing_side_waypoint(ego_lane_wp, forward_m):
    """Walk ego lane forward, step left once to reach opposing lane."""
    forward_wp = _advance(ego_lane_wp, forward_m)
    opposing = forward_wp.get_left_lane()
    if (opposing is None or opposing.lane_type != carla.LaneType.Driving
            or opposing.road_id != forward_wp.road_id
            or opposing.lane_id * forward_wp.lane_id >= 0):
        raise ValueError("requires a directly adjacent opposite-direction driving lane")
    return opposing


def _preroll_plan(opposing_wp, span_m, step_m=2.0):
    """Scenario helper."""
    plan = [(opposing_wp, RoadOption.LANEFOLLOW)]
    cur = opposing_wp
    travelled = 0.0
    while travelled < span_m:
        nxts = cur.next(step_m)
        if not nxts:
            raise ValueError("pre-roll: dead end after {:.1f} m".format(travelled))
        cur = nxts[0]
        if (cur.lane_type != opposing_wp.lane_type or cur.road_id != opposing_wp.road_id
                or cur.lane_id != opposing_wp.lane_id or cur.is_junction):
            raise ValueError("pre-roll: left the opposing lane after {:.1f} m".format(travelled))
        plan.append((cur, RoadOption.LANEFOLLOW))
        travelled += step_m
    return plan


def _emit_event(name, ego, actors, reason):
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "OncomingVehicleHeadOnIntrusionTown12", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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


class SustainedClearAmbient(AtomicBehavior):
    """Reissue LeaveSpaceInFront every tick to keep ambient traffic clear.
    (Pattern copied from SustainedClearAmbient, adapted for .)
    """

    def __init__(self, space, name="SustainedClearAmbient"):
        super(SustainedClearAmbient, self).__init__(name)
        self._space = space

    def update(self):
        py_trees.blackboard.Blackboard().set("BA_LeaveSpaceInFront", self._space, overwrite=True)
        return py_trees.common.Status.RUNNING


class PrerollNonTerminating(WaypointFollower):
    """Pre-roll WaypointFollower that never terminates on its own — only."""

    def update(self):
        status = super(PrerollNonTerminating, self).update()
        if status != py_trees.common.Status.RUNNING:
            return py_trees.common.Status.RUNNING
        return status


class CommitDistanceCondition(AtomicBehavior):
    """Simple distance-gated commit: fire when ego-intruder planar distance."""

    def __init__(self, ego, intruder, commit_distance_m):
        super(CommitDistanceCondition, self).__init__("CommitDistanceCondition", ego)
        self._ego, self._intruder = ego, intruder
        self._commit_distance_m = commit_distance_m

    def update(self):
        ego_loc = self._ego.get_location()
        intr_loc = self._intruder.get_location()
        distance = math.hypot(intr_loc.x - ego_loc.x, intr_loc.y - ego_loc.y)
        if distance <= self._commit_distance_m:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class LiveEgoPursuit(AtomicBehavior):
    """Scenario helper."""

    LATERAL_STOP_THRESHOLD_M = 1.0
    NEAR_TARGET_M = 3.0

    def __init__(self, intruder, ego, cruise_speed_mps, brake_decel_mps2,
                 pursuit_steer_rate, pursuit_steer_cap, brake_engaged_distance_m,
                 max_pursuit_distance_m, timeout_s):
        super(LiveEgoPursuit, self).__init__("LiveEgoPursuit", intruder)
        self._intruder, self._ego = intruder, ego
        self._cruise_speed_mps = cruise_speed_mps
        self._brake_decel_mps2 = brake_decel_mps2
        self._pursuit_steer_rate = pursuit_steer_rate
        self._pursuit_steer_cap = pursuit_steer_cap
        self._brake_engaged_distance_m = brake_engaged_distance_m
        self._max_pursuit_distance_m = max_pursuit_distance_m
        self._timeout_s = timeout_s
        self._started = None
        self._steer = 0.0
        self._braking = False

    def initialise(self):
        self._started = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        self._braking = False
        self._steer = 0.0
        self._intruder.set_light_state(
            carla.VehicleLightState(
                carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker))

    def update(self):
        elapsed = (CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
                   - self._started)

        ego_loc = self._ego.get_location()
        intr_loc = self._intruder.get_location()
        distance_to_ego = math.hypot(ego_loc.x - intr_loc.x, ego_loc.y - intr_loc.y)

        transform = self._intruder.get_transform()
        dx = ego_loc.x - transform.location.x
        dy = ego_loc.y - transform.location.y
        remaining = math.hypot(dx, dy)

        # Check if intruder has traveled far enough or timeout/collision will occur
        if (remaining < self.NEAR_TARGET_M or
                elapsed > self._timeout_s or
                distance_to_ego < 1.5):
            self._braking = True

        # Brake once close enough to ego or when conditions demand it
        if distance_to_ego < self._brake_engaged_distance_m or self._braking:
            self._braking = True
            steer_target = 0.0
            self._steer = max(self._steer - self._pursuit_steer_rate,
                              min(self._steer + self._pursuit_steer_rate, steer_target))
            brake_value = min(0.7, 0.1 * self._brake_decel_mps2)
            self._intruder.apply_control(carla.VehicleControl(
                throttle=0.0, brake=brake_value, steer=self._steer, hand_brake=False))
        else:
            # LIVE PURSUIT: steer toward ego's current position (recomputed every tick)
            wanted_yaw = math.atan2(dy, dx)
            yaw = math.radians(transform.rotation.yaw)
            yaw_error = (wanted_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
            steer_target = max(-self._pursuit_steer_cap, min(self._pursuit_steer_cap, 1.0 * yaw_error))
            self._steer = max(self._steer - self._pursuit_steer_rate,
                              min(self._steer + self._pursuit_steer_rate, steer_target))
            velocity = self._intruder.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            throttle = max(0.0, min(0.6, 0.15 + 0.04 * (self._cruise_speed_mps - speed)))
            self._intruder.apply_control(carla.VehicleControl(
                throttle=throttle, brake=0.0, steer=self._steer))

        speed = _speed(self._intruder)
        if self._braking and speed < 0.3:
            return py_trees.common.Status.SUCCESS
        if elapsed >= self._timeout_s:
            return py_trees.common.Status.SUCCESS
        if distance_to_ego < 0.5:
            return py_trees.common.Status.SUCCESS  # Collision imminent, resolve
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._intruder.is_alive:
            self._intruder.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        super(LiveEgoPursuit, self).terminate(new_status)


class HoldStationary(AtomicBehavior):
    """Keep intruder at rest via repeated brake application."""

    def __init__(self, actor):
        super(HoldStationary, self).__init__("HoldStationary", actor)

    def initialise(self):
        try:
            self._actor.set_light_state(carla.VehicleLightState.Hazard)
        except AttributeError:
            pass

    def update(self):
        self._actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        return py_trees.common.Status.RUNNING


class OncomingVehicleHeadOnIntrusionTown12(BasicScenario):
    timeout = 50

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=50):
        self._map = CarlaDataProvider.get_map()
        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._trigger_wp is None or self._trigger_wp.is_junction
                or self._trigger_wp.road_id != ROAD_ID):
            raise ValueError("requires verified Town01 road {}".format(ROAD_ID))
        _opposing_side_waypoint(self._trigger_wp, 1.0)  # Verify opposing lane exists

        self._intruder_spawn_m = _value(config, "intruder_spawn_distance_m", 120.0, 100.0, 140.0)
        self._intruder_speed_mps = _value(config, "intruder_speed_mps", 11.0, 9.0, 11.0)
        self._arm_distance_m = _value(config, "arm_distance_m", 100.0, 80.0, 130.0)
        self._commit_distance_m = _value(config, "commit_distance_m", 45.0, 35.0, 55.0)
        self._max_pursuit_distance_m = _value(config, "max_pursuit_distance_m", 80.0, 60.0, 100.0)
        self._pursuit_steer_rate = _value(config, "pursuit_steer_rate", 0.12, 0.06, 0.12)
        self._pursuit_steer_cap = _value(config, "pursuit_steer_cap", 0.80, 0.50, 0.80)
        self._brake_engaged_distance_m = _value(config, "brake_engaged_distance_m", 2.0, 2.0, 10.0)
        self._brake_decel_mps2 = _value(config, "brake_decel_mps2", 6.0, 4.0, 6.0)
        self._max_exposure_s = _value(config, "max_exposure_s", 20.0, 15.0, 30.0)
        self._clear_distance_m = _value(config, "clear_distance_m", 80.0, 60.0, 100.0)
        self._ambient_clear_bubble_m = _value(config, "ambient_clear_bubble_m", 120.0, 60.0, 200.0)

        self._intruder = None
        self._intruder_spawn_transform = None
        self._preroll_plan = None
        super(OncomingVehicleHeadOnIntrusionTown12, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        opposing_wp = _opposing_side_waypoint(self._trigger_wp, self._intruder_spawn_m)
        self._intruder_spawn_transform = carla.Transform(
            carla.Location(x=opposing_wp.transform.location.x, y=opposing_wp.transform.location.y,
                           z=opposing_wp.transform.location.z + 0.3),
            opposing_wp.transform.rotation)
        # Hidden staging: spawn 500 m below, physics off
        hidden_transform = carla.Transform(
            self._intruder_spawn_transform.location - carla.Location(z=500.0),
            self._intruder_spawn_transform.rotation)
        self._intruder = CarlaDataProvider.request_new_actor(
            "vehicle.tesla.model3", hidden_transform,
            rolename="scenario.intruder", color="30,10,10")
        if self._intruder is None:
            raise RuntimeError("intruder spawn failed")
        self._intruder.set_simulate_physics(False)
        self.other_actors.append(self._intruder)
        # Pre-roll plan: 80m span to ensure intruder reaches commit point
        self._preroll_plan = _preroll_plan(opposing_wp, 80.0)

    def _create_behavior(self):
        # Main scenario sequence
        scenario_sequence = py_trees.composites.Sequence("V3B01Scenario")
        scenario_sequence.add_child(ActorTransformSetter(
            self._intruder, self._intruder_spawn_transform, physics=True,
            name="V3B01RevealIntruder"))
        scenario_sequence.add_child(Idle(0.4))

        # ARM: distance-gated
        arm = py_trees.composites.Parallel(
            "V3B01Arm", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self._intruder, self._arm_distance_m, name="V3B01DistanceArm"))
        arm.add_child(DriveDistance(
            self.ego_vehicles[0], 30.0, name="V3B01ProgressFallbackArm"))
        scenario_sequence.add_child(arm)
        scenario_sequence.add_child(EventMarker("armed", self.ego_vehicles[0], [self._intruder],
                                      "arm distance condition satisfied"))

        # COMMIT + PURSUIT: distance-gated commit, then live ego-position pursuit
        commit = py_trees.composites.Parallel(
            "V3B01Commit", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        commit.add_child(CommitDistanceCondition(
            self.ego_vehicles[0], self._intruder, self._commit_distance_m))
        commit.add_child(PrerollNonTerminating(
            self._intruder, self._intruder_speed_mps, plan=self._preroll_plan,
            name="V3B01IntruderPreroll"))
        scenario_sequence.add_child(commit)
        scenario_sequence.add_child(EventMarker("committed", self.ego_vehicles[0], [self._intruder],
                                      "commit distance condition satisfied"))

        scenario_sequence.add_child(LiveEgoPursuit(
            self._intruder, self.ego_vehicles[0], self._intruder_speed_mps,
            self._brake_decel_mps2, self._pursuit_steer_rate, self._pursuit_steer_cap,
            self._brake_engaged_distance_m, self._max_pursuit_distance_m, 10.0))
        scenario_sequence.add_child(EventMarker("conflict_entered", self.ego_vehicles[0], [self._intruder],
                                      "live pursuit phase began"))

        # HOLD AND CLEAR: intruder held at rest until ego clears
        hold_and_clear = py_trees.composites.Parallel(
            "V3B01HoldAndClear", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        hold_and_clear.add_child(DriveDistance(
            self.ego_vehicles[0], self._clear_distance_m, name="V3B01EgoClears"))
        hold_and_clear.add_child(ScenarioTimeout(
            self._max_exposure_s, self.__class__.__name__, name="V3B01ExposureTimeout"))
        hold_and_clear.add_child(HoldStationary(self._intruder))
        scenario_sequence.add_child(hold_and_clear)
        scenario_sequence.add_child(EventMarker("cleared", self.ego_vehicles[0], [self._intruder],
                                      "ego cleared the exposure window"))
        scenario_sequence.add_child(ActorDestroy(self._intruder, name="V3B01DestroyIntruder"))

        # Root as Parallel: main scenario sequence + ambient clearing behavior
        root = py_trees.composites.Parallel(
            "V3B01RootWithAmbientClearing", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(scenario_sequence)
        root.add_child(SustainedClearAmbient(self._ambient_clear_bubble_m))
        return root

    def _create_test_criteria(self):
        return [
            CollisionTest(
                self.ego_vehicles[0], other_actor=self._intruder,
                terminate_on_failure=True, name="V3B01EgoIntruderCollision"),
            ActorBlockedTest(
                self.ego_vehicles[0], min_speed=5.0, max_time=35.0,
                terminate_on_failure=True, name="V3B01Blocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
