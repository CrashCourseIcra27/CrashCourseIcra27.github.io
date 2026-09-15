"""Two overtaking vehicles collide ahead (Town06).

Two overtaking vehicles converge and collide ahead, leaving a wreck in the ego's path.
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
    AtomicBehavior, ScenarioTimeout, WaypointFollower)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorBlockedTest, CollisionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    AtomicCondition, DriveDistance, TriggerVelocity)
from srunner.scenarios.basic_scenario import BasicScenario

ROAD_ID = 40  # Town06 -- see module docstring "Road geometry" for why this
              # diverges from .md's own (unworkable) road-38 assumption.


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
        raise ValueError("cannot advance {} m".format(distance_m))
    return candidates[0]


def _behind(waypoint, distance_m, step_m=2.0):
    """Scenario helper."""
    cur = waypoint
    travelled = 0.0
    while travelled < distance_m:
        step_now = min(step_m, distance_m - travelled)
        prevs = cur.previous(step_now)
        if not prevs:
            raise ValueError("behind-spawn walk: dead end after {:.1f} m".format(travelled))
        cur = prevs[0]
        if (cur.lane_type != waypoint.lane_type or cur.road_id != waypoint.road_id
                or cur.lane_id != waypoint.lane_id or cur.is_junction):
            raise ValueError("behind-spawn walk: left the lane after {:.1f} m".format(travelled))
        travelled += step_now
    return cur


def _inboard_driving(waypoint):
    """Scenario helper."""
    candidate = waypoint.get_left_lane()
    if (candidate is None or candidate.lane_type != carla.LaneType.Driving
            or candidate.road_id != waypoint.road_id
            or candidate.lane_id * waypoint.lane_id <= 0):
        raise ValueError("requires an inboard (left) driving lane for cutter_left")
    return candidate


def _outboard_driving(waypoint):
    """Scenario helper."""
    candidate = waypoint.get_right_lane()
    if (candidate is None or candidate.lane_type != carla.LaneType.Driving
            or candidate.road_id != waypoint.road_id
            or candidate.lane_id * waypoint.lane_id <= 0):
        raise ValueError("requires an outboard (right) driving lane for cutter_right")
    return candidate


def _verify_junction_free_chain(waypoint, distance_m, label):
    """Design-time (not just design-doc) proof of .md section 1's own
    junction-free-chain requirement -- raises loud rather than silently
    trusting the map."""
    cur = waypoint
    travelled = 0.0
    step = 20.0
    while travelled < distance_m:
        step_now = min(step, distance_m - travelled)
        nxts = cur.next(step_now)
        if (not nxts or nxts[0].road_id != waypoint.road_id
                or nxts[0].lane_id != waypoint.lane_id or nxts[0].is_junction):
            raise ValueError("{} chain only {:.1f} m junction-free, "
                              "needed {:.1f} m".format(label, travelled, distance_m))
        cur = nxts[0]
        travelled += step_now


def _speed(actor):
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _rate_limit(previous, desired, max_step):
    return max(previous - max_step, min(previous + max_step, desired))


def _wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _dp_distance_m(v_ego):
    """.md section 3's own formula: clamp(v_ego*(v_ego+6)*3.5/6, 50, 230).
    Cross-checked against the instruction's own worked examples (v=8 -> 65 m,
    v=16 -> 205 m) during design -- both match to within rounding."""
    return max(50.0, min(230.0, v_ego * (v_ego + 6.0) * 3.5 / 6.0))


def _lane_point_ahead(actor, distance_m):
    """Ego-conditioned interaction clock: walk `distance_m` forward along."""
    carla_map = CarlaDataProvider.get_map()
    anchor = carla_map.get_waypoint(actor.get_location(), project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if anchor is None:
        raise RuntimeError("cannot resolve a lane waypoint under the ego")
    remaining = distance_m
    cur = anchor
    step = 10.0
    while remaining > 1e-6:
        step_now = min(step, remaining)
        candidates = [item for item in cur.next(step_now)
                      if item.road_id == anchor.road_id and item.lane_id == anchor.lane_id
                      and item.lane_type == carla.LaneType.Driving and not item.is_junction]
        if not candidates:
            raise RuntimeError("cannot project P {} m ahead of the ego".format(distance_m))
        cur = candidates[0]
        remaining -= step_now
    fwd = cur.transform.get_forward_vector()
    return cur.transform.location, (fwd.x, fwd.y)


def _rear_right_quarter(actor, long_m, lat_m):
    """`actor`'s live rear-right-quarter corner, world frame -- map-relative,."""
    tf = actor.get_transform()
    yaw = math.radians(tf.rotation.yaw)
    world_x = tf.location.x + long_m * math.cos(yaw) - lat_m * math.sin(yaw)
    world_y = tf.location.y + long_m * math.sin(yaw) + lat_m * math.cos(yaw)
    return carla.Location(x=world_x, y=world_y, z=tf.location.z)


def _emit_event(name, ego, actors, reason, extra=None):
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "TwoOvertakingVehiclesCollideAheadTown06", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
                "frame": snapshot.frame, "simulation_time": snapshot.timestamp.elapsed_seconds,
                "route_progress_m": float(getattr(ego, "_route_progress_m", 0.0)) if ego is not None else 0.0,
                "reason": reason,
                "actors": [{"id": actor.id, "role_name": actor.attributes.get("role_name", ""),
                            "alive": actor.is_alive} for actor in actors if actor is not None]}
        if extra:
            item.update(extra)
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    except Exception:
        pass


def _emit_rig_track(name, actor, extra):
    """Additive telemetry emitter (the acceptance criterion): write-only, never controls a."""
    path = os.environ.get("_EVENT_LOG")
    if not path:
        return
    try:
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        item = {"event": name, "scenario_id": "TwoOvertakingVehiclesCollideAheadTown06", "revision": os.environ.get("SCENARIO_REVISION", "unknown"),
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
        super(EventMarker, self).__init__("V2B02EventMarker_" + event, actors[0] if actors else ego)
        self._event, self._ego, self._actors, self._reason = event, ego, actors, reason

    def update(self):
        _emit_event(self._event, self._ego, self._actors, self._reason)
        return py_trees.common.Status.SUCCESS


class BothCuttersLeadEgo(AtomicCondition):
    """SUCCESS once BOTH cutters lead the ego by >= lead_m, measured by
    projecting each cutter's position onto the ego's own forward axis --
    deliberately NOT Euclidean (both cutters are still one lane over
    during the overtake). Stateless/idempotent -- safe under any re-tick."""

    def __init__(self, ego, cutter_left, cutter_right, lead_m, name="BothCuttersLeadEgo"):
        super(BothCuttersLeadEgo, self).__init__(name)
        self._ego = ego
        self._left = cutter_left
        self._right = cutter_right
        self._lead_m = lead_m

    def _leads(self, forward, ego_loc, actor):
        delta = actor.get_location() - ego_loc
        return (delta.x * forward.x + delta.y * forward.y) >= self._lead_m

    def update(self):
        ego_transform = self._ego.get_transform()
        forward = ego_transform.get_forward_vector()
        ego_loc = ego_transform.location
        if self._leads(forward, ego_loc, self._left) and self._leads(forward, ego_loc, self._right):
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class SampleOvertakeSpeed(AtomicBehavior):
    """One-shot: freezes overtake_speed = clamp(v_ego_at_gate + 6, min, max)."""

    def __init__(self, ego, holder, speed_min, speed_max):
        super(SampleOvertakeSpeed, self).__init__("SampleOvertakeSpeed", ego)
        self._ego = ego
        self._holder = holder
        self._min = speed_min
        self._max = speed_max

    def update(self):
        v_gate = _speed(self._ego)
        self._holder[0] = max(self._min, min(self._max, v_gate + 6.0))
        return py_trees.common.Status.SUCCESS


class CruiseAt(AtomicBehavior):
    """Lane-centred cruise controller at a fixed target speed read once."""

    STEER_RATE_MAX = 0.05
    LOOKAHEAD_M = 6.0

    def __init__(self, actor, speed_holder, name):
        super(CruiseAt, self).__init__(name, actor)
        self._holder = speed_holder
        self._target_speed = None
        self._last_steer = 0.0

    def initialise(self):
        self._target_speed = self._holder[0]
        self._last_steer = 0.0

    def update(self):
        carla_map = CarlaDataProvider.get_map()
        tf = self._actor.get_transform()
        waypoint = carla_map.get_waypoint(tf.location, project_to_road=True,
                                           lane_type=carla.LaneType.Driving)
        lookahead = waypoint.next(self.LOOKAHEAD_M) if waypoint is not None else []
        if lookahead:
            target = lookahead[0].transform.location
            dx, dy = target.x - tf.location.x, target.y - tf.location.y
            desired_yaw = math.atan2(dy, dx)
            current_yaw = math.radians(tf.rotation.yaw)
            desired_steer = max(-0.4, min(0.4, 0.9 * _wrap_angle(desired_yaw - current_yaw)))
        else:
            desired_steer = 0.0
        steer = _rate_limit(self._last_steer, desired_steer, self.STEER_RATE_MAX)
        self._last_steer = steer
        speed = _speed(self._actor)
        speed_error = self._target_speed - speed
        throttle = max(0.0, min(0.95, 0.16 + 0.14 * speed_error))
        brake = max(0.0, min(0.3, -0.08 * speed_error))
        if brake > 0.02:
            throttle = 0.0
        self._actor.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        return py_trees.common.Status.RUNNING


class ConvergeAndCrash(AtomicBehavior):
    """Owns both cutters' controls from commit through settle. See module."""

    STEER_RATE_MAX = 0.05

    def __init__(self, ego, cutter_left, cutter_right, overtake_speed_holder,
                 homing_radius_m, homing_timeout_s, straighten_radius_m,
                 left_final_speed_mps, striker_preseek_offset_m,
                 rear_quarter_long_m, rear_quarter_lat_m, homing_speed_bonus_mps,
                 hands_off_s, brake_ramp_s, min_settle_s, rest_speed_mps,
                 safety_timeout_s):
        super(ConvergeAndCrash, self).__init__("ConvergeAndCrash", cutter_right)
        self._ego = ego
        self._left = cutter_left
        self._right = cutter_right
        self._overtake_holder = overtake_speed_holder
        self._homing_radius_m = homing_radius_m
        self._homing_timeout_s = homing_timeout_s
        self._straighten_radius_m = straighten_radius_m
        self._left_final_speed_mps = left_final_speed_mps
        self._striker_preseek_offset_m = striker_preseek_offset_m
        self._rear_quarter_long_m = rear_quarter_long_m
        self._rear_quarter_lat_m = rear_quarter_lat_m
        self._homing_speed_bonus_mps = homing_speed_bonus_mps
        self._hands_off_s = hands_off_s
        self._brake_ramp_s = brake_ramp_s
        self._min_settle_s = min_settle_s
        self._rest_speed_mps = rest_speed_mps
        self._safety_timeout_s = safety_timeout_s

        self._start_s = None
        self._overtake_speed = None
        self._p_location = None
        self._dp_m = None
        self._striker_seek_target = None
        self._left_straightened = False
        self._homing = False
        self._homing_start_s = None
        self._impact_s = None
        self._impact_is_fallback = False
        self._collision_hit = False
        self._collision_sensor = None
        self._rest_emitted = False
        self._left_last_steer = 0.0
        self._right_last_steer = 0.0

    def initialise(self):
        self._start_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        self._overtake_speed = self._overtake_holder[0]
        v_ego = _speed(self._ego)
        self._dp_m = _dp_distance_m(v_ego)
        self._p_location, forward = _lane_point_ahead(self._ego, self._dp_m)
        self._striker_seek_target = carla.Location(
            x=self._p_location.x - forward[0] * self._striker_preseek_offset_m,
            y=self._p_location.y - forward[1] * self._striker_preseek_offset_m,
            z=self._p_location.z)
        self._left_straightened = False
        self._homing = False
        self._homing_start_s = None
        self._impact_s = None
        self._impact_is_fallback = False
        self._collision_hit = False
        self._rest_emitted = False
        self._left_last_steer = 0.0
        self._right_last_steer = 0.0
        world = CarlaDataProvider.get_world()
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = world.spawn_actor(
            blueprint, carla.Transform(), attach_to=self._right)
        self._collision_sensor.listen(lambda event: self._on_collision(event))
        _emit_event("committed", self._ego, [self._left, self._right],
                    "phase1 overtake complete; P frozen on ego lane centreline, "
                    "live-v_ego-derived, homing choreography begins",
                    extra={"dp_m": self._dp_m, "p_x": self._p_location.x, "p_y": self._p_location.y,
                           "overtake_speed_mps": self._overtake_speed, "v_ego_at_commit": v_ego})

    def _on_collision(self, event):
        if self._collision_hit:
            return
        other = event.other_actor
        if other is None or other.id != self._left.id:
            return
        self._collision_hit = True

    def _steer_toward(self, actor, target, last_steer):
        tf = actor.get_transform()
        dx, dy = target.x - tf.location.x, target.y - tf.location.y
        distance_m = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        current_yaw = math.radians(tf.rotation.yaw)
        desired_steer = max(-0.55, min(0.55, 0.85 * _wrap_angle(desired_yaw - current_yaw)))
        steer = _rate_limit(last_steer, desired_steer, self.STEER_RATE_MAX)
        return steer, distance_m

    def _drive_to_speed(self, actor, target_speed, steer):
        speed = _speed(actor)
        speed_error = target_speed - speed
        throttle = max(0.0, min(0.7, 0.18 + 0.06 * speed_error))
        brake = max(0.0, min(0.35, -0.08 * speed_error))
        if brake > 0.02:
            throttle = 0.0
        actor.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

    def update(self):
        now_s = CarlaDataProvider.get_world().get_snapshot().timestamp.elapsed_seconds
        elapsed = now_s - self._start_s

        if self._impact_s is None:
            # --- left: point-seek P, straighten + slow once close ---
            left_steer, dist_left_to_p = self._steer_toward(self._left, self._p_location, self._left_last_steer)
            if dist_left_to_p <= self._straighten_radius_m:
                self._left_straightened = True
            if self._left_straightened:
                left_steer = _rate_limit(self._left_last_steer, 0.0, self.STEER_RATE_MAX)
                left_target_speed = self._left_final_speed_mps
            else:
                left_target_speed = self._overtake_speed
            self._left_last_steer = left_steer
            self._drive_to_speed(self._left, left_target_speed, left_steer)

            # --- right: pre-seek, then per-tick homing once close ---
            left_loc = self._left.get_transform().location
            right_loc = self._right.get_transform().location
            dist_to_left = right_loc.distance(left_loc)
            if not self._homing and dist_to_left <= self._homing_radius_m:
                self._homing = True
                self._homing_start_s = now_s
            if not self._homing:
                right_target = self._striker_seek_target
                right_target_speed = self._overtake_speed
            else:
                right_target = _rear_right_quarter(self._left, self._rear_quarter_long_m,
                                                    self._rear_quarter_lat_m)
                right_target_speed = _speed(self._left) + self._homing_speed_bonus_mps
            right_steer, dist_right_to_target = self._steer_toward(
                self._right, right_target, self._right_last_steer)
            self._right_last_steer = right_steer
            self._drive_to_speed(self._right, right_target_speed, right_steer)

            _emit_rig_track("converge_track", self._right, {
                "elapsed_s": elapsed, "homing": bool(self._homing),
                "dist_to_left_m": dist_to_left, "dist_left_to_p_m": dist_left_to_p,
                "left_speed_mps": _speed(self._left), "right_speed_mps": _speed(self._right),
                "left_x": left_loc.x, "left_y": left_loc.y,
                "right_x": right_loc.x, "right_y": right_loc.y,
                "left_yaw_deg": self._left.get_transform().rotation.yaw,
                "right_yaw_deg": self._right.get_transform().rotation.yaw})

            if self._collision_hit:
                self._impact_s = now_s
                self._left.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
                self._right.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
                _emit_event("conflict_entered", self._ego, [self._left, self._right],
                            "collision sensor fired: cutter_right struck cutter_left's rear-right quarter")
            elif self._homing and (now_s - self._homing_start_s) >= self._homing_timeout_s:
                # .md section 3: 3.0 s no-contact fallback -- brake to
                # rest wherever they are (still blocks the lane), emit
                # contact_missing. conflict_entered is NEVER emitted on
                # this path (wired only to the real collision event), so
                # this path is structurally distinguishable downstream.
                self._impact_s = now_s
                self._impact_is_fallback = True
                _emit_event("contact_missing", self._ego, [self._left, self._right],
                            "homing_timeout_s elapsed with no collision-sensor contact -- "
                            "rep invalid, never a silent pass")
        else:
            since_impact = now_s - self._impact_s
            hands_off_window = 0.0 if self._impact_is_fallback else self._hands_off_s
            if since_impact < hands_off_window:
                self._left.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
                self._right.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
            else:
                ramp = min(1.0, (since_impact - hands_off_window) / self._brake_ramp_s)
                self._left.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=max(0.3, ramp), hand_brake=ramp >= 1.0))
                self._right.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=max(0.3, ramp), hand_brake=ramp >= 1.0))
            left_speed = _speed(self._left)
            right_speed = _speed(self._right)
            both_at_rest = (since_impact > self._min_settle_s
                             and left_speed < self._rest_speed_mps
                             and right_speed < self._rest_speed_mps)
            left_loc = self._left.get_transform().location
            right_loc = self._right.get_transform().location
            _emit_rig_track("settle_track", self._right, {
                "since_impact_s": since_impact, "fallback": bool(self._impact_is_fallback),
                "left_speed_mps": left_speed, "right_speed_mps": right_speed,
                "both_at_rest": bool(both_at_rest),
                "left_x": left_loc.x, "left_y": left_loc.y,
                "right_x": right_loc.x, "right_y": right_loc.y,
                "left_yaw_deg": self._left.get_transform().rotation.yaw,
                "right_yaw_deg": self._right.get_transform().rotation.yaw})
            if both_at_rest and not self._rest_emitted:
                self._rest_emitted = True
                _emit_event("both_at_rest", self._ego, [self._left, self._right],
                            "both cutters braked below rest_speed_mps, min_settle_s elapsed")

        if self._rest_emitted or elapsed >= self._safety_timeout_s:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._collision_sensor is not None:
            try:
                self._collision_sensor.stop()
                self._collision_sensor.destroy()
            except Exception:
                pass
            self._collision_sensor = None
        for actor in (self._left, self._right):
            if actor is not None and actor.is_alive:
                actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        super(ConvergeAndCrash, self).terminate(new_status)


class HoldStationary(AtomicBehavior):
    """Full brake + handbrake, re-applied every tick, for the rest of the
    exposure window -- the persistent-blockage pattern this suite already
    uses in ////."""

    def __init__(self, actor):
        super(HoldStationary, self).__init__("HoldStationary", actor)

    def initialise(self):
        try:
            self._actor.set_light_state(carla.VehicleLightState.Hazard)
        except AttributeError:
            pass

    def update(self):
        if self._actor is not None and self._actor.is_alive:
            self._actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        return py_trees.common.Status.RUNNING


class EgoApproachTrack(AtomicBehavior):
    """Pure observer, never touches ego's controls. Ports /'s."""

    ARRIVAL_RADIUS_M = 5.0

    def __init__(self, ego, cutter_left, cutter_right):
        super(EgoApproachTrack, self).__init__("EgoApproachTrack", ego)
        self._ego = ego
        self._left = cutter_left
        self._right = cutter_right
        self._arrival_emitted = False
        self._conflict_location = None

    def initialise(self):
        self._arrival_emitted = False
        self._conflict_location = None

    def update(self):
        loc = self._ego.get_location()
        left_loc = self._left.get_location() if self._left.is_alive else None
        right_loc = self._right.get_location() if self._right.is_alive else None
        # Track distance to whichever hazard actor is currently nearer --
        # P itself is only known inside ConvergeAndCrash, but the rest
        # position of either cutter is a fine live proxy for arrival.
        distances = [loc.distance(p) for p in (left_loc, right_loc) if p is not None]
        nearest = min(distances) if distances else None
        _emit_rig_track("ego_approach_track", self._ego, {
            "ego_x": loc.x, "ego_y": loc.y,
            "distance_to_left_m": loc.distance(left_loc) if left_loc else None,
            "distance_to_right_m": loc.distance(right_loc) if right_loc else None})
        if not self._arrival_emitted and nearest is not None and nearest <= self.ARRIVAL_RADIUS_M:
            self._arrival_emitted = True
            _emit_event("ego_reached_conflict_zone", self._ego, [self._left, self._right],
                        "ego within ARRIVAL_RADIUS_M={} m of a hazard actor".format(self.ARRIVAL_RADIUS_M))
        return py_trees.common.Status.RUNNING


class TwoOvertakingVehiclesCollideAheadTown06(BasicScenario):
    timeout = 55

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=55):
        self._map = CarlaDataProvider.get_map()
        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._trigger_wp is None or self._trigger_wp.is_junction
                or self._trigger_wp.road_id != ROAD_ID):
            raise ValueError("requires verified Town06 road {}".format(ROAD_ID))
        self._left_lane_wp = _inboard_driving(self._trigger_wp)
        self._right_lane_wp = _outboard_driving(self._trigger_wp)
        # Design-time proof of .md section 1's own chain requirement
        # (260 m ego lane, 80 m flanks) -- fails closed, not just declared.
        _verify_junction_free_chain(self._trigger_wp, 260.0, "ego lane -4")
        _verify_junction_free_chain(self._left_lane_wp, 80.0, "flank lane -3")
        _verify_junction_free_chain(self._right_lane_wp, 80.0, "flank lane -5")

        self._cutter_left_spawn_m = _value(config, "cutter_left_spawn_distance_m", 26.0, 26.0, 32.0)
        self._cutter_right_spawn_m = _value(config, "cutter_right_spawn_distance_m", 36.0, 26.0, 36.0)
        self._approach_speed_mps = _value(config, "approach_speed_mps", 8.0, 6.0, 10.0)
        self._gate_speed_mps = _value(config, "gate_speed_mps", 7.5, 5.0, 9.0)
        self._overtake_lead_gap_m = _value(config, "overtake_lead_gap_m", 8.0, 6.0, 10.0)
        self._overtake_timeout_s = _value(config, "overtake_timeout_s", 10.0, 6.0, 10.0)
        self._overtake_speed_min_mps = _value(config, "overtake_speed_min_mps", 13.0, 10.0, 15.0)
        self._overtake_speed_max_mps = _value(config, "overtake_speed_max_mps", 22.0, 18.0, 26.0)
        self._homing_radius_m = _value(config, "homing_radius_m", 12.0, 10.0, 14.0)
        self._homing_timeout_s = _value(config, "homing_timeout_s", 3.0, 2.5, 4.0)
        self._straighten_radius_m = _value(config, "straighten_radius_m", 1.5, 1.0, 2.0)
        self._left_final_speed_mps = _value(config, "left_final_speed_mps", 5.0, 4.0, 6.0)
        self._striker_preseek_offset_m = _value(config, "striker_preseek_offset_m", 6.0, 4.0, 8.0)
        self._rear_quarter_long_m = _value(config, "rear_quarter_long_m", -2.2, -3.0, -1.5)
        self._rear_quarter_lat_m = _value(config, "rear_quarter_lat_m", 0.6, 0.3, 1.0)
        self._homing_speed_bonus_mps = _value(config, "homing_speed_bonus_mps", 5.0, 3.0, 7.0)
        self._hands_off_s = _value(config, "hands_off_s", 0.3, 0.2, 0.5)
        self._brake_ramp_s = _value(config, "brake_ramp_s", 1.0, 0.5, 1.5)
        self._min_settle_s = _value(config, "min_settle_s", 1.0, 0.5, 1.5)
        self._rest_speed_mps = _value(config, "rest_speed_mps", 0.1, 0.05, 0.2)
        self._crash_safety_timeout_s = _value(config, "crash_safety_timeout_s", 16.0, 12.0, 20.0)
        self._clear_drive_distance_m = _value(config, "clear_drive_distance_m", 20.0, 15.0, 30.0)
        self._max_exposure_s = _value(config, "max_exposure_s", 25.0, 18.0, 35.0)

        self._cutter_left = None
        self._cutter_right = None
        super(TwoOvertakingVehiclesCollideAheadTown06, self).__init__(
            self.__class__.__name__, ego_vehicles, config, world, debug_mode,
            criteria_enable=criteria_enable)
        _emit_event("initialized", self.ego_vehicles[0], self.other_actors,
                    "scenario initialized and actors spawned")

    def _initialize_actors(self, config):
        left_wp = _behind(self._left_lane_wp, self._cutter_left_spawn_m)
        self._cutter_left = CarlaDataProvider.request_new_actor(
            "vehicle.dodge.charger_2020", left_wp.transform,
            rolename="scenario.cutter_left", color="15,15,18")
        if self._cutter_left is None:
            raise RuntimeError("cutter_left spawn failed")
        self.other_actors.append(self._cutter_left)

        right_wp = _behind(self._right_lane_wp, self._cutter_right_spawn_m)
        self._cutter_right = CarlaDataProvider.request_new_actor(
            "vehicle.nissan.patrol_2021", right_wp.transform,
            rolename="scenario.cutter_right", color="225,225,215")
        if self._cutter_right is None:
            raise RuntimeError("cutter_right spawn failed")
        self.other_actors.append(self._cutter_right)

    def _create_behavior(self):
        ego = self.ego_vehicles[0]

        phase0_gate = py_trees.composites.Parallel(
            "V2B02EgoSpeedGate", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        phase0_gate.add_child(TriggerVelocity(
            ego, self._gate_speed_mps, comparison_operator=operator.ge, name="V2B02EgoSpeedGate"))
        phase0_gate.add_child(WaypointFollower(
            self._cutter_left, self._approach_speed_mps, name="V2B02LeftApproach"))
        phase0_gate.add_child(WaypointFollower(
            self._cutter_right, self._approach_speed_mps, name="V2B02RightApproach"))
        phase0_gate.add_child(DriveDistance(ego, 45.0, name="V2B02GateProgressFallback"))

        overtake_speed_holder = [self._overtake_speed_min_mps]

        # Phase 1: sample overtake_speed once, then cruise both cutters at
        # that fixed speed until both lead the ego by overtake_lead_gap_m,
        # or overtake_timeout_s elapses.
        phase1_overtake = py_trees.composites.Parallel(
            "V2B02Overtake", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        phase1_overtake.add_child(BothCuttersLeadEgo(
            ego, self._cutter_left, self._cutter_right, self._overtake_lead_gap_m))
        phase1_overtake.add_child(ScenarioTimeout(
            self._overtake_timeout_s, self.__class__.__name__, name="V2B02OvertakeTimeout"))
        phase1_overtake.add_child(CruiseAt(
            self._cutter_left, overtake_speed_holder, name="V2B02LeftOvertakeCruise"))
        phase1_overtake.add_child(CruiseAt(
            self._cutter_right, overtake_speed_holder, name="V2B02RightOvertakeCruise"))

        event = py_trees.composites.Sequence("V2B02EventPhases")
        event.add_child(phase0_gate)
        event.add_child(EventMarker("armed", ego, [self._cutter_left, self._cutter_right],
                                         "ego-relative arm condition satisfied"))
        event.add_child(SampleOvertakeSpeed(
            ego, overtake_speed_holder, self._overtake_speed_min_mps, self._overtake_speed_max_mps))
        event.add_child(phase1_overtake)
        # "committed" and "conflict_entered" are emitted from inside
        # ConvergeAndCrash at the actual P-freeze and collision-sensor
        # moments (both timing-critical, internal state -- see that class).
        event.add_child(ConvergeAndCrash(
            ego, self._cutter_left, self._cutter_right, overtake_speed_holder,
            self._homing_radius_m, self._homing_timeout_s, self._straighten_radius_m,
            self._left_final_speed_mps, self._striker_preseek_offset_m,
            self._rear_quarter_long_m, self._rear_quarter_lat_m, self._homing_speed_bonus_mps,
            self._hands_off_s, self._brake_ramp_s, self._min_settle_s, self._rest_speed_mps,
            self._crash_safety_timeout_s))

        clear = py_trees.composites.Parallel(
            "V2B02ClearOrTimeout", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        clear.add_child(DriveDistance(ego, self._clear_drive_distance_m, name="V2B02EgoClears"))
        clear.add_child(ScenarioTimeout(self._max_exposure_s, self.__class__.__name__,
                                        name="V2B02ExposureTimeout"))
        clear.add_child(HoldStationary(self._cutter_left))
        clear.add_child(HoldStationary(self._cutter_right))
        event.add_child(clear)
        event.add_child(EventMarker("cleared", ego, [self._cutter_left, self._cutter_right],
                                         "ego cleared the bounded exposure window"))

        root = py_trees.composites.Parallel(
            "V2B02RootWithApproachTelemetry", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(event)
        root.add_child(EgoApproachTrack(ego, self._cutter_left, self._cutter_right))
        return root

    def _create_test_criteria(self):
        return [
            CollisionTest(self.ego_vehicles[0], other_actor=self._cutter_left,
                          terminate_on_failure=True, name="V2B02EgoLeftCollision"),
            CollisionTest(self.ego_vehicles[0], other_actor=self._cutter_right,
                          terminate_on_failure=True, name="V2B02EgoRightCollision"),
            ActorBlockedTest(self.ego_vehicles[0], min_speed=3.0, max_time=35.0,
                             terminate_on_failure=True, name="V2B02Blocked"),
        ]

    def __del__(self):
        try:
            if self.ego_vehicles:
                _emit_event("cleanup", self.ego_vehicles[0], self.other_actors,
                            "scenario actor cleanup")
        except Exception:
            pass
        self.remove_all_actors()
