"""Low grip vehicle pair collision blocks lane (Town12).

Two reduced-grip vehicles slide together, collide, and block the ego's lane.
"""
from __future__ import print_function

import json
import math
import operator
import os

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    AtomicBehavior, ScenarioTimeout)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest, CollisionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance, InTimeToArrivalToLocation)
from srunner.scenarios.basic_scenario import BasicScenario

ROAD_ID = 837
ICY_TIRE_FRICTION = 1.0  # .md section 1 -- CARLA 0.9.15 default is ~3.5


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


def _speed(actor):
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


class _CollisionLatch(object):
    """One-shot collision latch (the acceptance criterion)."""

    def __init__(self):
        self.hit = False


def _set_icy_friction(actor):
    """real grip change, tire_friction=1.0 on both hazard actors."""
    physics_control = actor.get_physics_control()
    wheels = physics_control.wheels
    for wheel in wheels:
        wheel.tire_friction = ICY_TIRE_FRICTION
    physics_control.wheels = wheels
    actor.apply_physics_control(physics_control)


def _emit_event(name, ego, actors, reason):
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "LowGripVehiclePairCollisionBlocksLaneTown12", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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


def _emit_track(name, actor, extra):
    """Additive telemetry emitter (the acceptance criterion convention)."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "LowGripVehiclePairCollisionBlocksLaneTown12", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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
        super(EventMarker, self).__init__("V3A09EventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


class SustainedClearAmbient(AtomicBehavior):
    """Reissue LeaveSpaceInFront every tick to keep ambient traffic clear."""

    def __init__(self, space, name="SustainedClearAmbient"):
        super(SustainedClearAmbient, self).__init__(name)
        self._space = space

    def update(self):
        py_trees.blackboard.Blackboard().set("BA_LeaveSpaceInFront", self._space, overwrite=True)
        return py_trees.common.Status.RUNNING


class HoldStationary(AtomicBehavior):
    """Full brake (capped at <=0.4, never handbrake) applied every tick."""

    def __init__(self, actor):
        super(HoldStationary, self).__init__("HoldStationary", actor)

    def update(self):
        if self._actor is not None and self._actor.is_alive:
            self._actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.4))
        return py_trees.common.Status.RUNNING


class EgoApproachTrack(AtomicBehavior):
    """Pure observer: emit ego_approach_track every tick plus ego_reached_conflict_zone on arrival."""

    ARRIVAL_RADIUS_M = 5.0

    def __init__(self, ego, conflict_location, icy_left, icy_right):
        super(EgoApproachTrack, self).__init__("EgoApproachTrack", ego)
        self._ego = ego
        self._conflict_location = conflict_location
        self._icy_left = icy_left
        self._icy_right = icy_right
        self._arrival_emitted = False

    def initialise(self):
        self._arrival_emitted = False

    def update(self):
        loc = self._ego.get_location()
        distance_conflict = loc.distance(self._conflict_location)
        distance_left = loc.distance(self._icy_left.get_location()) if self._icy_left.is_alive else None
        distance_right = loc.distance(self._icy_right.get_location()) if self._icy_right.is_alive else None
        _emit_track("ego_approach_track", self._ego, {
            "ego_x": loc.x, "ego_y": loc.y, "ego_speed_mps": _speed(self._ego),
            "distance_to_conflict_m": distance_conflict,
            "distance_to_icy_left_m": distance_left, "distance_to_icy_right_m": distance_right})
        if not self._arrival_emitted and distance_conflict <= self.ARRIVAL_RADIUS_M:
            self._arrival_emitted = True
            _emit_event("ego_reached_conflict_zone", self._ego, [self._icy_left, self._icy_right],
                        "ego within ARRIVAL_RADIUS_M={} m of the conflict point".format(self.ARRIVAL_RADIUS_M))
        return py_trees.common.Status.RUNNING


class IcyTwinSlide(AtomicBehavior):
    """improved control law: increased throttle cap (0.7 vs 0.5) to ensure
    convergence at the tighter target geometry. Owns both actors from arm through settle.
    """

    HANDS_OFF_S = 1.5
    BRAKE_MAX = 0.4
    REST_SPEED_MPS = 0.3
    MIN_SETTLE_S = 1.0

    def __init__(self, icy_left, icy_right, left_target, right_target,
                 slide_timeout_s, cruise_speed_mps, outer_timeout_s, collision_latch):
        super(IcyTwinSlide, self).__init__("IcyTwinSlide", icy_left)
        self._icy_left = icy_left
        self._icy_right = icy_right
        self._left_target = left_target
        self._right_target = right_target
        self._slide_timeout_s = slide_timeout_s
        self._cruise_speed_mps = cruise_speed_mps
        self._outer_timeout_s = outer_timeout_s
        self._collision_latch = collision_latch
        self._start_s = None
        self._left_steer = 0.0
        self._right_steer = 0.0
        self._impact_s = None
        self._rest_emitted = False

    def initialise(self):
        self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        self._left_steer = 0.0
        self._right_steer = 0.0
        self._impact_s = None
        self._rest_emitted = False
        for actor in (self._icy_left, self._icy_right):
            try:
                actor.set_light_state(carla.VehicleLightState(
                    carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker))
            except AttributeError:
                pass
        _emit_event("committed", None, [self._icy_left, self._icy_right],
                    "twin slide committed; both actors point-seeking into lane -4")

    @staticmethod
    def _icy_slide_control(actor, target, elapsed, timeout_s, steer_state, cruise_speed_mps):
        """Scenario helper."""
        transform = actor.get_transform()
        dx = target.x - transform.location.x
        dy = target.y - transform.location.y
        remaining = math.hypot(dx, dy)
        wanted_yaw = math.atan2(dy, dx)
        yaw = math.radians(transform.rotation.yaw)
        yaw_error = (wanted_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
        spin_bias = 0.30 if elapsed < 0.9 else (-0.22 if elapsed < 2.4 else 0.0)
        wanted_steer = max(-0.6, min(0.6, 0.85 * yaw_error + spin_bias))
        new_steer = max(steer_state - 0.04, min(steer_state + 0.04, wanted_steer))
        velocity = actor.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        near_target = remaining < 2.6
        near_timeout = elapsed > (timeout_s - 1.0)
        brake = 0.25 if (near_target or near_timeout) else 0.0
        throttle = (max(0.0, min(0.7, 0.15 + 0.05 * (cruise_speed_mps - speed)))
                    if not brake else 0.0)
        actor.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=new_steer))
        return new_steer, remaining

    def update(self):
        now_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        elapsed = now_s - self._start_s

        if self._impact_s is None:
            self._left_steer, remaining_left = self._icy_slide_control(
                self._icy_left, self._left_target, elapsed, self._slide_timeout_s,
                self._left_steer, self._cruise_speed_mps)
            self._right_steer, remaining_right = self._icy_slide_control(
                self._icy_right, self._right_target, elapsed, self._slide_timeout_s,
                self._right_steer, self._cruise_speed_mps)
            _emit_track("slide_track", self._icy_left, {
                "elapsed_s": elapsed, "remaining_left_m": remaining_left, "remaining_right_m": remaining_right,
                "left_speed_mps": _speed(self._icy_left), "right_speed_mps": _speed(self._icy_right),
                "left_x": self._icy_left.get_location().x, "left_y": self._icy_left.get_location().y,
                "right_x": self._icy_right.get_location().x, "right_y": self._icy_right.get_location().y})

            if self._collision_latch.hit:
                self._impact_s = now_s
                self._icy_left.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
                self._icy_right.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
                _emit_event("conflict_entered", None, [self._icy_left, self._icy_right],
                            "collision sensor fired: icy_left contacted icy_right")
            elif elapsed >= self._slide_timeout_s:
                # No contact within timeout = clean MISS (non-exposure)
                self._icy_left.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5))
                self._icy_right.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5))
                return py_trees.common.Status.SUCCESS
        else:
            since_impact = now_s - self._impact_s
            if since_impact < self.HANDS_OFF_S:
                # Hands-off: physics resolves impact
                self._icy_left.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
                self._icy_right.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
                _emit_track("impact_track", self._icy_left, {
                    "since_impact_s": since_impact,
                    "left_speed_mps": _speed(self._icy_left), "right_speed_mps": _speed(self._icy_right)})
            else:
                # Constant brake, capped at BRAKE_MAX
                left_speed = _speed(self._icy_left)
                right_speed = _speed(self._icy_right)
                self._icy_left.apply_control(carla.VehicleControl(throttle=0.0, brake=self.BRAKE_MAX, steer=0.0))
                self._icy_right.apply_control(carla.VehicleControl(throttle=0.0, brake=self.BRAKE_MAX, steer=0.0))
                both_at_rest = (since_impact > self.MIN_SETTLE_S
                                 and left_speed < self.REST_SPEED_MPS
                                 and right_speed < self.REST_SPEED_MPS)
                _emit_track("settle_track", self._icy_left, {
                    "since_impact_s": since_impact, "left_speed_mps": left_speed, "right_speed_mps": right_speed,
                    "both_at_rest": bool(both_at_rest),
                    "left_x": self._icy_left.get_location().x, "left_y": self._icy_left.get_location().y,
                    "left_yaw": self._icy_left.get_transform().rotation.yaw,
                    "right_x": self._icy_right.get_location().x, "right_y": self._icy_right.get_location().y,
                    "right_yaw": self._icy_right.get_transform().rotation.yaw})
                if both_at_rest and not self._rest_emitted:
                    self._rest_emitted = True
                    _emit_event("both_at_rest", None, [self._icy_left, self._icy_right],
                                "both actors braked below REST_SPEED_MPS, MIN_SETTLE_S elapsed since impact")

        if self._rest_emitted or elapsed >= self._outer_timeout_s:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        for actor in (self._icy_left, self._icy_right):
            if actor is not None and actor.is_alive:
                actor.apply_control(carla.VehicleControl(throttle=0.0, brake=self.BRAKE_MAX))
        super(IcyTwinSlide, self).terminate(new_status)


class LowGripVehiclePairCollisionBlocksLaneTown12(BasicScenario):
    timeout = 45

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=45):
        self._map = CarlaDataProvider.get_map()
        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._trigger_wp is None or self._trigger_wp.is_junction
                or self._trigger_wp.road_id != ROAD_ID):
            raise ValueError("requires verified Town06 road {}".format(ROAD_ID))
        self._left_lane_wp = _left_driving(self._trigger_wp, hops=1)   # lane -3
        self._right_lane_wp = _right_driving(self._trigger_wp, hops=1)  # lane -5

        self._icy_left_spawn_m = _value(config, "icy_left_spawn_distance_m", 155.0, 100.0, 155.0)
        self._icy_right_spawn_m = _value(config, "icy_right_spawn_distance_m", 153.0, 100.0, 153.0)
        self._icy_left_target_m = _value(config, "icy_left_target_distance_m", 164.5, 150.0, 178.0)
        self._icy_right_target_m = _value(config, "icy_right_target_distance_m", 162.5, 148.0, 176.0)
        self._cruise_speed_mps = _value(config, "icy_cruise_speed_mps", 9.0, 8.0, 9.0)
        self._icy_slide_timeout_s = _value(config, "icy_slide_timeout_s", 6.5, 3.5, 6.5)
        self._arm_ttc_s = _value(config, "arm_ttc_s", 11.0, 9.0, 11.0)
        # Conflict point moved closer to reflect tighter target geometry
        self._conflict_distance_m = _value(config, "conflict_distance_m", 163.5, 150.0, 178.0)
        self._arm_progress_fallback_m = _value(config, "arm_progress_fallback_m", 155.0, 40.0, 178.0)
        self._max_exposure_s = _value(config, "max_exposure_s", 20.0, 12.0, 30.0)
        self._ego_clear_distance_m = _value(config, "ego_clear_distance_m", 90.0, 40.0, 110.0)

        self._icy_left = None
        self._icy_right = None
        self._collision_sensor = None
        self._collision_latch = _CollisionLatch()
        super(LowGripVehiclePairCollisionBlocksLaneTown12, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        left_wp = _advance(self._left_lane_wp, self._icy_left_spawn_m)
        self._icy_left = CarlaDataProvider.request_new_actor(
            "vehicle.lincoln.mkz_2017", left_wp.transform,
            rolename="scenario.icy_left", color="15,15,20")
        if self._icy_left is None:
            raise RuntimeError("icy_left spawn failed")
        _set_icy_friction(self._icy_left)
        self.other_actors.append(self._icy_left)

        right_wp = _advance(self._right_lane_wp, self._icy_right_spawn_m)
        self._icy_right = CarlaDataProvider.request_new_actor(
            "vehicle.nissan.patrol_2021", right_wp.transform,
            rolename="scenario.icy_right", color="120,20,20")
        if self._icy_right is None:
            raise RuntimeError("icy_right spawn failed")
        _set_icy_friction(self._icy_right)
        self.other_actors.append(self._icy_right)

        # Collision sensor on icy_left listening for icy_right
        world_ref = CarlaDataProvider.get_world()
        blueprint = world_ref.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = world_ref.spawn_actor(
            blueprint, carla.Transform(), attach_to=self._icy_left)
        icy_right_id = self._icy_right.id
        latch = self._collision_latch

        def _on_collision(event):
            if latch.hit:
                return
            other = event.other_actor
            if other is not None and other.id == icy_right_id:
                latch.hit = True

        self._collision_sensor.listen(_on_collision)

    def _create_behavior(self):
        left_target = _advance(self._trigger_wp, self._icy_left_target_m).transform.location
        right_target = _advance(self._trigger_wp, self._icy_right_target_m).transform.location
        conflict = _advance(self._trigger_wp, self._conflict_distance_m).transform.location

        arm = py_trees.composites.Parallel(
            "V3A09ApproachAndArm", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        arm.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._arm_ttc_s, conflict,
            comparison_operator=operator.lt, name="V3A09LiveTTCArm"))
        arm.add_child(DriveDistance(
            self.ego_vehicles[0], self._arm_progress_fallback_m, name="V3A09ProgressFallback"))

        event = py_trees.composites.Sequence("V3A09EventPhases")
        event.add_child(arm)
        event.add_child(EventMarker("armed", self.ego_vehicles[0], [self._icy_left, self._icy_right],
                                         "ego live-TTC arm condition satisfied"))
        event.add_child(IcyTwinSlide(
            self._icy_left, self._icy_right, left_target, right_target,
            self._icy_slide_timeout_s, self._cruise_speed_mps,
            outer_timeout_s=self._icy_slide_timeout_s + 7.0,
            collision_latch=self._collision_latch))

        clear = py_trees.composites.Parallel(
            "V3A09ClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(self.ego_vehicles[0], self._ego_clear_distance_m, name="V3A09EgoClears"))
        clear.add_child(ScenarioTimeout(self._max_exposure_s, self.__class__.__name__,
                                        name="V3A09ExposureTimeout"))
        clear.add_child(HoldStationary(self._icy_left))
        clear.add_child(HoldStationary(self._icy_right))
        event.add_child(clear)
        event.add_child(EventMarker("cleared", self.ego_vehicles[0], [self._icy_left, self._icy_right],
                                         "ego cleared the bounded exposure window"))

        ego_location = self.ego_vehicles[0].get_location()
        ambient_clearance_m = ego_location.distance(conflict) + self._ego_clear_distance_m + 20.0

        root = py_trees.composites.Parallel(
            "V3A09RootWithTelemetry", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(event)
        root.add_child(SustainedClearAmbient(ambient_clearance_m))
        root.add_child(EgoApproachTrack(
            self.ego_vehicles[0], conflict, self._icy_left, self._icy_right))
        return root

    def _create_test_criteria(self):
        return [
            CollisionTest(self.ego_vehicles[0], other_actor=self._icy_left,
                          terminate_on_failure=True, name="V3A09EgoIcyLeftCollision"),
            CollisionTest(self.ego_vehicles[0], other_actor=self._icy_right,
                          terminate_on_failure=True, name="V3A09EgoIcyRightCollision"),
            ActorBlockedTest(self.ego_vehicles[0], min_speed=3.0, max_time=35.0,
                             terminate_on_failure=True, name="V3A09Blocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        if getattr(self, "_collision_sensor", None) is not None:
            try:
                self._collision_sensor.stop()
                self._collision_sensor.destroy()
            except Exception:
                pass
            self._collision_sensor = None
        self.remove_all_actors()
