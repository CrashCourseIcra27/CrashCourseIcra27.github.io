"""Local custom atomic behaviors for the custom scenario set."""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _vector_payload(vector):
    if vector is None:
        return None
    return {"x": float(vector.x), "y": float(vector.y), "z": float(vector.z)}


def _rotation_payload(rotation):
    if rotation is None:
        return None
    return {
        "pitch": float(rotation.pitch),
        "yaw": float(rotation.yaw),
        "roll": float(rotation.roll),
    }


def _control_payload(actor):
    try:
        control = actor.get_control()
    except (AttributeError, RuntimeError):
        return None
    payload = {}
    for name in ("throttle", "brake", "steer", "hand_brake", "reverse", "manual_gear_shift", "gear"):
        if hasattr(control, name):
            value = getattr(control, name)
            if isinstance(value, bool):
                payload[name] = value
            elif isinstance(value, int):
                payload[name] = int(value)
            else:
                payload[name] = float(value)
    return payload


class CarlaRolloutLogger(py_trees.behaviour.Behaviour):
    """Export strict CARLA actor kinematics for the BeamNG intermediate layer."""

    def __init__(
        self,
        actors: Iterable[Tuple[str, Optional[carla.Actor]]],
        scenario_id: str,
        route_id: Optional[str] = None,
        name: str = "CarlaRolloutLogger",
    ):
        super().__init__(name)
        self._actors = list(actors)
        self._scenario_id = scenario_id
        self._route_id = route_id
        self._enabled = False
        self._out_dir = None
        self._rollout_file = None
        self._collision_file = None
        self._sensors = []
        self._last_frame = None
        self._started = False

    def initialise(self):
        if self._started:
            return
        self._started = True
        self._enabled = _env_enabled("CUSTOM_SCENARIO_LOG_ROLLOUT")
        if not self._enabled:
            return

        out_dir = os.environ.get("CUSTOM_SCENARIO_ROLLOUT_DIR", "").strip()
        if not out_dir:
            out_dir = os.path.join(os.getcwd(), "carla_rollout")
        self._out_dir = out_dir
        os.makedirs(self._out_dir, exist_ok=True)

        self._write_json("manifest.json", self._manifest_payload(status="running"))
        self._write_json("actors.json", self._actors_payload())
        self._rollout_file = open(os.path.join(self._out_dir, "rollout.jsonl"), "a", encoding="utf-8")
        self._collision_file = open(os.path.join(self._out_dir, "collisions.jsonl"), "a", encoding="utf-8")
        self._attach_collision_sensors()

    def update(self):
        if not self._enabled:
            return py_trees.common.Status.RUNNING

        world = CarlaDataProvider.get_world()
        snapshot = world.get_snapshot() if world is not None else None
        frame = int(snapshot.frame) if snapshot is not None else None
        if frame is not None and frame == self._last_frame:
            return py_trees.common.Status.RUNNING
        self._last_frame = frame

        timestamp = snapshot.timestamp if snapshot is not None else None
        payload = {
            "schema_version": 1,
            "scenario_id": self._scenario_id,
            "route_id": self._route_id,
            "frame": frame,
            "elapsed_seconds": float(timestamp.elapsed_seconds) if timestamp is not None else float(GameTime.get_time()),
            "delta_seconds": float(timestamp.delta_seconds) if timestamp is not None else None,
            "platform": "carla",
            "states": [self._actor_state(label, actor) for label, actor in self._actors],
        }
        self._write_line(self._rollout_file, payload)
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._enabled:
            self._write_json("manifest.json", self._manifest_payload(status=str(new_status)))
            self._destroy_sensors()
            if self._rollout_file is not None:
                self._rollout_file.close()
                self._rollout_file = None
            if self._collision_file is not None:
                self._collision_file.close()
                self._collision_file = None
        super().terminate(new_status)

    def _manifest_payload(self, status: str) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "carla_rollout_export",
            "scenario_id": self._scenario_id,
            "route_id": self._route_id,
            "status": status,
            "rollout": "rollout.jsonl",
            "collisions": "collisions.jsonl",
            "actors": "actors.json",
            "contract": "CARLA supplies route, scenario logic, triggers, and actor kinematics; BeamNG supplies map fit, road shading, collision physics, and post-impact dynamics.",
        }

    def _actors_payload(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "scenario_id": self._scenario_id,
            "actors": [self._actor_metadata(label, actor) for label, actor in self._actors],
        }

    def _actor_metadata(self, label: str, actor: Optional[carla.Actor]) -> Dict[str, object]:
        if actor is None:
            return {"label": label, "id": None, "type_id": None, "role_name": None, "alive": False}
        role_name = None
        try:
            role_name = actor.attributes.get("role_name")
        except (AttributeError, RuntimeError):
            pass
        return {
            "label": label,
            "id": int(actor.id),
            "type_id": actor.type_id,
            "role_name": role_name,
            "alive": bool(actor.is_alive),
        }

    def _actor_state(self, label: str, actor: Optional[carla.Actor]) -> Dict[str, object]:
        base = self._actor_metadata(label, actor)
        if actor is None or not actor.is_alive:
            return base
        try:
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            acceleration = actor.get_acceleration()
            angular_velocity = actor.get_angular_velocity()
        except RuntimeError:
            base["alive"] = False
            return base
        base.update({
            "location": _vector_payload(transform.location),
            "rotation": _rotation_payload(transform.rotation),
            "velocity": _vector_payload(velocity),
            "acceleration": _vector_payload(acceleration),
            "angular_velocity": _vector_payload(angular_velocity),
            "control": _control_payload(actor),
        })
        return base

    def _attach_collision_sensors(self):
        world = CarlaDataProvider.get_world()
        if world is None:
            return
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        attached_ids = set()
        for label, actor in self._actors:
            if actor is None or not actor.is_alive or actor.id in attached_ids:
                continue
            attached_ids.add(actor.id)
            try:
                sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=actor)
                sensor.listen(lambda event, actor_label=label: self._on_collision(actor_label, event))
                self._sensors.append(sensor)
            except RuntimeError:
                pass

    def _on_collision(self, actor_label: str, event):
        if self._collision_file is None:
            return
        other = getattr(event, "other_actor", None)
        impulse = getattr(event, "normal_impulse", None)
        transform = getattr(event, "transform", None)
        payload = {
            "schema_version": 1,
            "scenario_id": self._scenario_id,
            "route_id": self._route_id,
            "frame": int(getattr(event, "frame", -1)),
            "timestamp": float(getattr(event, "timestamp", GameTime.get_time())),
            "actor_label": actor_label,
            "other_actor_id": int(other.id) if other is not None else None,
            "other_actor_type_id": other.type_id if other is not None else None,
            "normal_impulse": _vector_payload(impulse),
            "location": _vector_payload(transform.location) if transform is not None else None,
            "rotation": _rotation_payload(transform.rotation) if transform is not None else None,
        }
        if impulse is not None:
            payload["impulse_magnitude"] = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._write_line(self._collision_file, payload)

    def _destroy_sensors(self):
        for sensor in self._sensors:
            if sensor is not None and sensor.is_alive:
                try:
                    sensor.stop()
                    sensor.destroy()
                except RuntimeError:
                    pass
        self._sensors = []

    def _write_json(self, filename: str, payload: Dict[str, object]):
        if self._out_dir is None:
            return
        path = os.path.join(self._out_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @staticmethod
    def _write_line(handle, payload: Dict[str, object]):
        if handle is None:
            return
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()


class StopActorsOnEgoCollision(py_trees.behaviour.Behaviour):
    """
    Monitor an ego vehicle collision sensor. On first impact, apply full brake
    plus hand brake to every actor in ``actors_to_stop`` and return SUCCESS.
    """

    def __init__(
        self,
        ego_vehicle: carla.Vehicle,
        actors_to_stop: List[Optional[carla.Vehicle]],
        name: str = "StopActorsOnEgoCollision",
    ):
        super().__init__(name)
        self._ego = ego_vehicle
        self._actors = actors_to_stop
        self._collision_sensor = None
        self._impact_detected = False

    def initialise(self):
        if self._collision_sensor is not None:
            return

        world = CarlaDataProvider.get_world()
        if world is None:
            return

        bp = world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._ego,
        )
        self._collision_sensor.listen(self._on_collision)

    def _on_collision(self, event):
        del event
        self._impact_detected = True

    def update(self):
        if not self._impact_detected:
            return py_trees.common.Status.RUNNING

        stop_control = carla.VehicleControl()
        stop_control.throttle = 0.0
        stop_control.brake = 1.0
        stop_control.steer = 0.0
        stop_control.hand_brake = True

        for actor in self._actors:
            if actor is not None and actor.is_alive:
                try:
                    actor.apply_control(stop_control)
                except RuntimeError:
                    pass

        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        if self._collision_sensor is not None and self._collision_sensor.is_alive:
            try:
                self._collision_sensor.stop()
                self._collision_sensor.destroy()
            except RuntimeError:
                pass
            self._collision_sensor = None
        super().terminate(new_status)


class HoldUntilEgoMoves(py_trees.behaviour.Behaviour):
    """Hold an actor stationary until the ego produces non-zero velocity."""

    def __init__(self, actor, ego, speed_threshold=0.5, name="HoldUntilEgoMoves"):
        super().__init__(name)
        self._actor = actor
        self._ego = ego
        self._speed_threshold = speed_threshold

    def update(self):
        if self._actor is not None and self._actor.is_alive:
            ctrl = carla.VehicleControl()
            ctrl.brake = 1.0
            ctrl.hand_brake = True
            self._actor.apply_control(ctrl)

        if self._ego is not None and self._ego.is_alive:
            velocity = self._ego.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            if speed > self._speed_threshold:
                return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class WaitUntilAheadOfEgo(py_trees.behaviour.Behaviour):
    """Return SUCCESS when an actor is sufficiently ahead of the ego."""

    def __init__(self, actor, ego, ahead_distance=5.0, name="WaitUntilAheadOfEgo"):
        super().__init__(name)
        self._actor = actor
        self._ego = ego
        self._ahead_distance = ahead_distance

    def update(self):
        if self._ego is None or self._actor is None:
            return py_trees.common.Status.FAILURE
        if not self._ego.is_alive or not self._actor.is_alive:
            return py_trees.common.Status.FAILURE

        ego_transform = self._ego.get_transform()
        forward = ego_transform.rotation.get_forward_vector()
        delta = self._actor.get_location() - ego_transform.location
        projection = delta.x * forward.x + delta.y * forward.y + delta.z * forward.z

        if projection >= self._ahead_distance:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class WaitForActorCollision(py_trees.behaviour.Behaviour):
    """Wait until a collision sensor attached to an actor fires."""

    def __init__(self, actor, name="WaitForActorCollision"):
        super().__init__(name)
        self._actor = actor
        self._sensor = None
        self._collided = False

    def initialise(self):
        world = CarlaDataProvider.get_world()
        if world is None:
            return
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self._sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._actor)
        self._sensor.listen(lambda _: setattr(self, "_collided", True))

    def update(self):
        if self._collided:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._sensor is not None and self._sensor.is_alive:
            try:
                self._sensor.stop()
                self._sensor.destroy()
            except RuntimeError:
                pass
        super().terminate(new_status)


class InSameDrivingLaneAsActor(py_trees.behaviour.Behaviour):
    """Return SUCCESS when actor is in the same driving lane as reference_actor."""

    def __init__(self, actor, reference_actor, name="InSameDrivingLaneAsActor"):
        super().__init__(name)
        self._actor = actor
        self._ref = reference_actor

    def update(self):
        if self._actor is None or self._ref is None:
            return py_trees.common.Status.FAILURE
        world_map = CarlaDataProvider.get_map()
        waypoint_a = world_map.get_waypoint(self._actor.get_location())
        waypoint_b = world_map.get_waypoint(self._ref.get_location())
        if waypoint_a is not None and waypoint_b is not None:
            if waypoint_a.road_id == waypoint_b.road_id and waypoint_a.lane_id == waypoint_b.lane_id:
                return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class ApplyControlContinuous(py_trees.behaviour.Behaviour):
    """Apply a fixed VehicleControl every tick and always keep RUNNING."""

    def __init__(self, actor, throttle=0.0, brake=0.0, steer=0.0,
                 hand_brake=False, name="ApplyControlContinuous"):
        super().__init__(name)
        self._actor = actor
        self._ctrl = carla.VehicleControl()
        self._ctrl.throttle = float(throttle)
        self._ctrl.brake = float(brake)
        self._ctrl.steer = max(-1.0, min(1.0, float(steer)))
        self._ctrl.hand_brake = bool(hand_brake)

    def update(self):
        if self._actor is not None and self._actor.is_alive:
            self._actor.apply_control(self._ctrl)
        return py_trees.common.Status.RUNNING


class ApplyControlForDuration(py_trees.behaviour.Behaviour):
    """Apply a fixed VehicleControl for a fixed duration, then return SUCCESS."""

    def __init__(self, actor, throttle=0.0, brake=0.0, steer=0.0,
                 hand_brake=False, duration=1.0, name="ApplyControlForDuration"):
        super().__init__(name)
        self._actor = actor
        self._ctrl = carla.VehicleControl()
        self._ctrl.throttle = float(throttle)
        self._ctrl.brake = float(brake)
        self._ctrl.steer = max(-1.0, min(1.0, float(steer)))
        self._ctrl.hand_brake = bool(hand_brake)
        self._duration = float(duration)
        self._start_time = None

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        if self._actor is not None and self._actor.is_alive:
            self._actor.apply_control(self._ctrl)
        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0
        if elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class PhysicsLossOfControlBreach(py_trees.behaviour.Behaviour):
    """Placeholder physics loss-of-control behavior used by adversarial breach."""

    def __init__(self, actor, duration=3.0, name="PhysicsLossOfControlBreach"):
        super().__init__(name)
        self._actor = actor
        self._duration = duration
        self._start_time = None

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        if self._actor is not None and not self._actor.is_alive:
            return py_trees.common.Status.SUCCESS
        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0
        if elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class AdversarialSpeedChaser(py_trees.behaviour.Behaviour):
    """Accelerate an actor toward a high target speed before a breach maneuver."""

    def __init__(self, actor, target_speed=30.0, name="AdversarialSpeedChaser"):
        super().__init__(name)
        self._actor = actor
        self._target_speed = float(target_speed)

    def update(self):
        if self._actor is None or not self._actor.is_alive:
            return py_trees.common.Status.FAILURE

        velocity = self._actor.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        ctrl = carla.VehicleControl()
        if speed < self._target_speed:
            ctrl.throttle = 0.8
            ctrl.brake = 0.0
        else:
            ctrl.throttle = 0.0
            ctrl.brake = 0.3
        self._actor.apply_control(ctrl)
        return py_trees.common.Status.RUNNING


class PeriodicStatusLogger(py_trees.behaviour.Behaviour):
    """Print periodic scenario status lines and always keep RUNNING."""

    def __init__(self, ego, adv=None, npc=None, interval=2.0, name="PeriodicStatusLogger"):
        super().__init__(name)
        self._ego = ego
        self._adv = adv
        self._npc = npc
        self._interval = interval
        self._last_log_time = None

    def update(self):
        now = GameTime.get_time()
        if self._last_log_time is None or (now - self._last_log_time) >= self._interval:
            self._last_log_time = now
            try:
                if self._ego and self._ego.is_alive:
                    velocity = self._ego.get_velocity()
                    speed_kmh = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                    parts = [f"[StatusLog] t={now:.1f}s  ego_speed={speed_kmh:.1f} km/h"]
                    if self._adv and self._adv.is_alive:
                        adv_velocity = self._adv.get_velocity()
                        adv_speed = 3.6 * math.sqrt(
                            adv_velocity.x ** 2 + adv_velocity.y ** 2 + adv_velocity.z ** 2
                        )
                        parts.append(f"adv_speed={adv_speed:.1f}")
                    print("  ".join(parts), flush=True)
            except Exception:
                pass
        return py_trees.common.Status.RUNNING


class PanicDriverControl(py_trees.behaviour.Behaviour):
    """Delayed overcorrection loop used by the adversarial breach scenario."""

    def __init__(self, actor, duration=3.0, reaction_delay=0.3, gain=1.5,
                 yaw_rate_scale=90.0, name="PanicDriverControl"):
        super().__init__(name)
        self._actor = actor
        self._duration = float(duration)
        self._reaction_delay = float(reaction_delay)
        self._gain = float(gain)
        self._yaw_rate_scale = float(yaw_rate_scale)
        self._start_time = None
        self._last_yaw = None

    def initialise(self):
        self._start_time = GameTime.get_time()
        if self._actor and self._actor.is_alive:
            self._last_yaw = self._actor.get_transform().rotation.yaw

    def update(self):
        if self._actor is None or not self._actor.is_alive:
            return py_trees.common.Status.SUCCESS

        now = GameTime.get_time()
        elapsed = now - self._start_time if self._start_time else 0.0
        if elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS

        current_yaw = self._actor.get_transform().rotation.yaw
        yaw_rate = 0.0
        if self._last_yaw is not None:
            delta = (current_yaw - self._last_yaw + 180.0) % 360.0 - 180.0
            yaw_rate = delta * 20.0
        self._last_yaw = current_yaw

        steer = 0.0
        if elapsed > self._reaction_delay:
            normalized = yaw_rate / self._yaw_rate_scale
            steer = -self._gain * normalized
            steer = max(-1.0, min(1.0, steer))

        ctrl = carla.VehicleControl()
        ctrl.throttle = 0.3
        ctrl.steer = steer
        self._actor.apply_control(ctrl)

        return py_trees.common.Status.RUNNING


class SpinOutControl(py_trees.behaviour.Behaviour):
    """Multi-phase spin-out maneuver used by the adversarial breach scenario."""

    def __init__(self, actor, steer_direction=1.0,
                 cut_in_steer=0.85, cut_in_throttle=0.25, cut_in_duration=0.5,
                 drift_steer=0.25, drift_steer_sign=-1.0,
                 drift_throttle=0.0, drift_brake=0.4, drift_duration=2.0,
                 spin_throttle=0.4, spin_duration=1.5,
                 block_duration=5.0,
                 name="SpinOutControl"):
        super().__init__(name)
        self._actor = actor
        self._dir = float(steer_direction)
        self._ci_steer = float(cut_in_steer)
        self._ci_throttle = float(cut_in_throttle)
        self._ci_dur = float(cut_in_duration)
        self._dr_steer = float(drift_steer)
        self._dr_sign = float(drift_steer_sign)
        self._dr_throttle = float(drift_throttle)
        self._dr_brake = float(drift_brake)
        self._dr_dur = float(drift_duration)
        self._sp_throttle = float(spin_throttle)
        self._sp_dur = float(spin_duration)
        self._bl_dur = float(block_duration)
        self._start_time = None

    def initialise(self):
        self._start_time = GameTime.get_time()

    @property
    def _total_duration(self):
        return self._ci_dur + self._dr_dur + self._sp_dur + self._bl_dur

    def update(self):
        if self._actor is None or not self._actor.is_alive:
            return py_trees.common.Status.SUCCESS

        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0
        if elapsed >= self._total_duration:
            return py_trees.common.Status.SUCCESS

        ctrl = carla.VehicleControl()

        if elapsed < self._ci_dur:
            ctrl.steer = max(-1.0, min(1.0, self._dir * self._ci_steer))
            ctrl.throttle = self._ci_throttle
        elif elapsed < self._ci_dur + self._dr_dur:
            ctrl.steer = max(-1.0, min(1.0, self._dir * self._dr_sign * self._dr_steer))
            ctrl.throttle = self._dr_throttle
            ctrl.brake = self._dr_brake
        elif elapsed < self._ci_dur + self._dr_dur + self._sp_dur:
            ctrl.throttle = self._sp_throttle
        else:
            ctrl.hand_brake = True

        self._actor.apply_control(ctrl)
        return py_trees.common.Status.RUNNING


class LaneCleaner(py_trees.behaviour.Behaviour):
    """Destroy ambient vehicles in protected lanes around the scenario area."""

    def __init__(self, protected_actors, lane_waypoints, center_location,
                 radius=500.0, interval=1.0, name="LaneCleaner"):
        super().__init__(name)
        self._protected = set(actor.id for actor in protected_actors if actor is not None)
        self._lane_wps = lane_waypoints
        self._center = center_location
        self._radius = float(radius)
        self._interval = float(interval)
        self._last_clean = None

    def update(self):
        now = GameTime.get_time()
        if self._last_clean is not None and (now - self._last_clean) < self._interval:
            return py_trees.common.Status.RUNNING
        self._last_clean = now

        world = CarlaDataProvider.get_world()
        if world is None:
            return py_trees.common.Status.RUNNING

        world_map = CarlaDataProvider.get_map()
        to_destroy = []

        for actor in world.get_actors().filter("vehicle.*"):
            if actor.id in self._protected:
                continue
            if actor.attributes.get("role_name") == "bg_ambient":
                continue  # keep deterministic ambient traffic (EXTRA_TRAFFIC)
            location = actor.get_location()
            distance = math.sqrt(
                (location.x - self._center.x) ** 2 +
                (location.y - self._center.y) ** 2
            )
            if distance > self._radius:
                continue

            waypoint = world_map.get_waypoint(location)
            if waypoint is None:
                continue
            for lane_wp in self._lane_wps:
                if waypoint.road_id == lane_wp.road_id and waypoint.lane_id == lane_wp.lane_id:
                    to_destroy.append(actor)
                    break

        for actor in to_destroy:
            try:
                actor.destroy()
            except RuntimeError:
                pass

        return py_trees.common.Status.RUNNING


def actor_speed(actor):
    """Planar speed (m/s) of an actor, 0.0 if missing/dead."""
    if actor is None or not actor.is_alive:
        return 0.0
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def planar_distance(loc_a, loc_b):
    return math.hypot(loc_a.x - loc_b.x, loc_a.y - loc_b.y)


def lane_follow_control(world_map, actor, target_speed_mps,
                        steer_gain=0.9, lookahead_min=4.0, max_throttle=0.85):
    """Natural physics lane-keeping: pure-pursuit steering toward a waypoint ahead
    in the actor's current lane + P-control on throttle/brake toward target speed.

    Returns a carla.VehicleControl. Never teleports — motion comes from the
    vehicle's own engine/brakes/tires, so it stays glued to the road surface.
    """
    ctrl = carla.VehicleControl()
    if actor is None or not actor.is_alive:
        return ctrl
    transform = actor.get_transform()
    speed = actor_speed(actor)
    waypoint = world_map.get_waypoint(
        transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    target = None
    if waypoint is not None:
        nxt = waypoint.next(max(lookahead_min, speed * 0.8))
        target = (nxt[0] if nxt else waypoint).transform.location
    if target is not None:
        yaw = math.radians(transform.rotation.yaw)
        desired = math.atan2(target.y - transform.location.y,
                             target.x - transform.location.x)
        err = (desired - yaw + math.pi) % (2.0 * math.pi) - math.pi
        ctrl.steer = max(-1.0, min(1.0, steer_gain * err))
    dv = target_speed_mps - speed
    if dv > 0.3:
        ctrl.throttle = max(0.0, min(max_throttle, 0.35 + 0.25 * dv))
    elif dv < -1.0:
        ctrl.brake = max(0.0, min(0.8, -0.30 * dv))
    return ctrl


def chase_control(actor, target_location, throttle=1.0, steer_gain=1.2):
    """Aggressive physics pursuit of a world location (used for a deliberate ram).
    Pure steering toward the target at the given throttle — real dynamics only.
    """
    ctrl = carla.VehicleControl()
    if actor is None or not actor.is_alive:
        return ctrl
    transform = actor.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    desired = math.atan2(target_location.y - transform.location.y,
                         target_location.x - transform.location.x)
    err = (desired - yaw + math.pi) % (2.0 * math.pi) - math.pi
    ctrl.steer = max(-1.0, min(1.0, steer_gain * err))
    ctrl.throttle = float(throttle)
    return ctrl


class DriveOffAndVanish(py_trees.behaviour.Behaviour):
    """Have an actor drive away naturally (physics lane-follow) and despawn only
    once it is far from the ego (out of sight), so it never vanishes on camera.
    SUCCESS when the actor is destroyed (or already gone).
    """

    def __init__(self, actor, ego, speed_mps=13.0, vanish_distance=70.0,
                 max_duration=20.0, name="DriveOffAndVanish"):
        super().__init__(name)
        self._actor = actor
        self._ego = ego
        self._speed = float(speed_mps)
        self._vanish_distance = float(vanish_distance)
        self._max_duration = float(max_duration)
        self._start_time = None

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        if self._actor is None or not self._actor.is_alive:
            return py_trees.common.Status.SUCCESS
        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0
        world_map = CarlaDataProvider.get_map()
        self._actor.apply_control(
            lane_follow_control(world_map, self._actor, self._speed))
        far = (self._ego is None or not self._ego.is_alive or
               planar_distance(self._actor.get_location(), self._ego.get_location())
               > self._vanish_distance)
        if far or elapsed > self._max_duration:
            try:
                self._actor.destroy()
            except RuntimeError:
                pass
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class EgoSpeedGovernor(py_trees.behaviour.Behaviour):
    """Apply gentle braking when ego exceeds the configured speed cap.

    If max_brake is set (> 0), also clamp the ego's brake input to that
    value at all times — preventing the autopilot from emergency-braking
    even when below the speed cap.
    """

    def __init__(self, ego, speed_cap_kmh=33.0, brake_force=0.2,
                 max_brake=0.0, name="EgoSpeedGovernor"):
        super().__init__(name)
        self._ego = ego
        self._cap_mps = float(speed_cap_kmh) / 3.6
        self._brake = float(brake_force)
        self._max_brake = float(max_brake)

    def update(self):
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.RUNNING

        velocity = self._ego.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        if speed > self._cap_mps:
            ctrl = self._ego.get_control()
            ctrl.brake = self._brake
            ctrl.throttle = 0.0
            self._ego.apply_control(ctrl)
        elif self._max_brake > 0:
            ctrl = self._ego.get_control()
            if ctrl.brake > self._max_brake:
                ctrl.brake = self._max_brake
                self._ego.apply_control(ctrl)

        return py_trees.common.Status.RUNNING
