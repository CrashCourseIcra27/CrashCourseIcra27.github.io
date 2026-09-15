"""Lead vehicle brakes to full stop (Town13).

A lead vehicle decelerates and comes to a full stop in the ego's lane.
"""
from __future__ import print_function

import json
import math
import os

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy, AtomicBehavior, ScenarioTimeout, WaypointFollower)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest, CollisionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance)
from srunner.scenarios.basic_scenario import BasicScenario


def _value(config, name, default, low, high):
    """Parse and validate a config parameter."""
    value = float(config.other_parameters.get(name, {}).get("value", default))
    if value < low or value > high:
        raise ValueError("{} outside [{}, {}]".format(name, low, high))
    return value


def _ahead(waypoint, distance_m):
    """Walk forward along the route from a waypoint."""
    candidates = [item for item in waypoint.next(distance_m)
                  if item.road_id == waypoint.road_id
                  and item.lane_id == waypoint.lane_id
                  and item.lane_type == carla.LaneType.Driving
                  and not item.is_junction]
    if not candidates:
        raise ValueError("cannot advance {} m on road 40 lane -2".format(distance_m))
    return candidates[0]


def _speed_mps(actor):
    """Get actor speed in m/s."""
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _progress_m(actor, origin, forward):
    """Get actor progress along forward axis from origin."""
    location = actor.get_location()
    return (location.x - origin.x) * forward.x + (location.y - origin.y) * forward.y


def _emit_event(name, ego, actors, reason):
    """Emit a lifecycle event."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "LeadVehicleBrakesToFullStopTown13", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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
    """One-shot event marker."""
    def __init__(self, event, ego, actors, reason):
        super(EventMarker, self).__init__("V3B06aEventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


class SustainedClearAmbient(AtomicBehavior):
    """Reissue LeaveSpaceInFront every tick to keep ambient traffic cleared."""
    def __init__(self, space, name="SustainedClearAmbient"):
        super(SustainedClearAmbient, self).__init__(name)
        self._space = space

    def update(self):
        py_trees.blackboard.Blackboard().set("BA_LeaveSpaceInFront", self._space, overwrite=True)
        return py_trees.common.Status.RUNNING


class CruiseControl(AtomicBehavior):
    """Drive lead actor toward cruise speed via proportional throttle control."""
    def __init__(self, actor, cruise_speed_mps, name="CruiseControl"):
        super(CruiseControl, self).__init__(name, actor)
        self._cruise = cruise_speed_mps

    def update(self):
        speed = _speed_mps(self._actor)
        error = self._cruise - speed
        throttle = max(0.0, min(0.6, 0.15 + 0.10 * error))
        self._actor.apply_control(carla.VehicleControl(throttle=throttle, brake=0.0, steer=0.0))
        return py_trees.common.Status.RUNNING


class GapTrigger(AtomicBehavior):
    """SUCCESS when bumper-to-bumper gap <= fixed trigger distance."""
    def __init__(self, ego, lead, origin, forward, gap_distance_m,
                 name="GapTrigger"):
        super(GapTrigger, self).__init__(name, ego)
        self._ego = ego
        self._lead = lead
        self._origin = origin
        self._forward = forward
        self._gap_distance_m = gap_distance_m

    def update(self):
        ego_prog = _progress_m(self._ego, self._origin, self._forward)
        lead_prog = _progress_m(self._lead, self._origin, self._forward)
        ego_half = self._ego.bounding_box.extent.x
        lead_half = self._lead.bounding_box.extent.x
        gap = (lead_prog - lead_half) - (ego_prog + ego_half)
        if gap <= self._gap_distance_m:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class BrakeToStop(AtomicBehavior):
    """Brake at constant deceleration until at rest."""
    def __init__(self, actor, decel_mps2, name="BrakeToStop"):
        super(BrakeToStop, self).__init__(name, actor)
        self._decel = decel_mps2
        self._start_s = None
        self._start_speed = None
        self._converged = False

    def initialise(self):
        if self._start_s is None:
            self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
            self._start_speed = _speed_mps(self._actor)

    def update(self):
        if self._converged:
            return py_trees.common.Status.SUCCESS
        now_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        elapsed_s = now_s - self._start_s
        target_speed = max(0.0, self._start_speed - self._decel * elapsed_s)
        speed = _speed_mps(self._actor)
        error = target_speed - speed
        brake = max(0.1, min(1.0, 0.25 - 0.20 * error))
        self._actor.apply_control(carla.VehicleControl(throttle=0.0, brake=brake, steer=0.0))
        # Converge when at rest or past kinematic deadline
        if speed <= 0.15 or elapsed_s >= (self._start_speed / max(self._decel, 0.1)) + 1.0:
            self._converged = True
            self._actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class LeadVehicleBrakesToFullStopTown13(BasicScenario):
    timeout = 40

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=40):
        self._map = CarlaDataProvider.get_map()
        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._trigger_wp is None or self._trigger_wp.is_junction
                or self._trigger_wp.road_id != 1349):
            raise ValueError("requires verified Town04 road 40")

        # Parse scenario parameters
        self._lead_spawn_distance_m = _value(config, "lead_spawn_distance_m", 30.0, 25.0, 40.0)
        self._cruise_speed_mps = _value(config, "cruise_speed_mps", 6.0, 5.0, 7.0)
        self._gap_trigger_distance_m = _value(config, "gap_trigger_distance_m", 20.0, 15.0, 30.0)
        self._brake_decel_mps2 = _value(config, "brake_decel_mps2", 7.0, 6.0, 8.0)
        self._max_exposure_s = _value(config, "max_exposure_s", 12.0, 8.0, 18.0)
        self._ego_clear_distance_m = _value(config, "ego_clear_distance_m", 80.0, 60.0, 100.0)

        self._lead = None
        self._route_origin = self._trigger_wp.transform.location
        forward = self._trigger_wp.transform.get_forward_vector()
        forward_len = math.hypot(forward.x, forward.y)
        self._route_forward = carla.Vector3D(
            x=forward.x / forward_len, y=forward.y / forward_len, z=0.0)

        super(LeadVehicleBrakesToFullStopTown13, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and lead actor spawned")

    def _initialize_actors(self, config):
        """Spawn lead vehicle 30 m ahead, physics-ON, at rest."""
        lead_wp = _ahead(self._trigger_wp, self._lead_spawn_distance_m)
        self._lead = CarlaDataProvider.request_new_actor(
            "vehicle.lincoln.mkz_2020", lead_wp.transform,
            rolename="scenario.lead", color="225,225,225")
        if self._lead is None:
            raise RuntimeError("lead spawn failed")
        self.other_actors.append(self._lead)

    def _create_behavior(self):
        """Build behavior tree."""
        # Main scenario events in sequence
        event = py_trees.composites.Sequence("V3B06aEventSequence")

        # Cruise phase: lead drives at 6 m/s while ego approaches
        approach = py_trees.composites.Parallel(
            "V3B06aApproachAndTrigger", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        approach.add_child(CruiseControl(self._lead, self._cruise_speed_mps))
        approach.add_child(GapTrigger(
            self.ego_vehicles[0], self._lead, self._route_origin, self._route_forward,
            self._gap_trigger_distance_m))
        # Fallback: commit by ego progress to avoid infinite approach
        approach.add_child(DriveDistance(
            self.ego_vehicles[0], self._lead_spawn_distance_m + 10.0, name="V3B06aProgressFallback"))
        event.add_child(approach)

        # Brake phase: lead decelerates to stop
        event.add_child(EventMarker(
            "armed", self.ego_vehicles[0], [self._lead], "gap trigger fired, lead commits to braking"))
        event.add_child(BrakeToStop(self._lead, self._brake_decel_mps2))
        event.add_child(EventMarker(
            "committed", self.ego_vehicles[0], [self._lead], "lead at rest in ego lane"))

        # Hold phase: lead stays stopped while ego reacts
        event.add_child(EventMarker(
            "conflict_entered", self.ego_vehicles[0], [self._lead],
            "lead at rest occupying ego's lane"))
        exposure = py_trees.composites.Parallel(
            "V3B06aExposure", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        exposure.add_child(WaypointFollower(self._lead, 0.0, name="V3B06aLeadHold"))
        exposure.add_child(DriveDistance(self.ego_vehicles[0], self._ego_clear_distance_m, name="V3B06aEgoClear"))
        exposure.add_child(ScenarioTimeout(
            self._max_exposure_s, self.__class__.__name__, name="V3B06aExposureTimeout"))
        event.add_child(exposure)

        event.add_child(EventMarker(
            "cleared", self.ego_vehicles[0], [self._lead], "ego cleared or exposure timeout"))
        event.add_child(ActorDestroy(self._lead, name="V3B06aDestroyLead"))

        # Wrap main event sequence with ambient clearing running in parallel
        # The parallel will complete when the main event sequence completes
        root = py_trees.composites.Parallel(
            "V3B06aRoot", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(SustainedClearAmbient(
            self._ego_clear_distance_m + 40.0, name="V3B06aClearAmbient"))
        root.add_child(event)

        return root

    def _create_test_criteria(self):
        """Define pass/fail criteria."""
        return [
            CollisionTest(
                self.ego_vehicles[0], other_actor=self._lead,
                terminate_on_failure=True, name="V3B06aEgoLeadCollision"),
            ActorBlockedTest(
                self.ego_vehicles[0], min_speed=5.0, max_time=35.0,
                terminate_on_failure=True, name="V3B06aEgoBlocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
