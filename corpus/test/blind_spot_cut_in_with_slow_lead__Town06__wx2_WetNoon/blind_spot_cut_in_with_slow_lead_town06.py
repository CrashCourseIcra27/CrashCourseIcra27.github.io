"""Blind spot cut in with slow lead (Town06).

A vehicle cuts in from the ego's blind spot while a slow lead vehicle limits the escape.
"""

import os
import subprocess
import math
import threading
import queue
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.scenariomanager.scenarioatomics.custom_atomics_blind_spot_cut_in_with_slow_lead_town06 import (
    HoldUntilEgoMoves,
    ApplyControlForDuration,
    PeriodicStatusLogger,
    LaneCleaner,
    EgoSpeedGovernor,
    StopActorsOnEgoCollision,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny helper: fire-and-forget one-shot behavior
# ─────────────────────────────────────────────────────────────────────────────

class _OneShot(py_trees.behaviour.Behaviour):
    """Execute a callback once on first tick, then return SUCCESS forever."""

    def __init__(self, callback, name="OneShot"):
        super().__init__(name)
        self._callback = callback
        self._fired = False

    def update(self):
        if not self._fired:
            self._callback()
            self._fired = True
        return py_trees.common.Status.SUCCESS


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    """Override autopilot braking -- force ego to maintain minimum speed."""

    def __init__(self, ego, min_speed_kmh=55.0, throttle=0.8,
                 name="EgoMinSpeedForcer"):
        super().__init__(name)
        self._ego = ego
        self._min_mps = float(min_speed_kmh) / 3.6
        self._throttle = float(throttle)

    def update(self):
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.RUNNING
        v = self._ego.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        if speed > 3.0 and speed < self._min_mps:
            ctrl = self._ego.get_control()
            ctrl.brake = 0.0
            ctrl.throttle = self._throttle
            self._ego.apply_control(ctrl)
        return py_trees.common.Status.RUNNING


class _ForcedLaneChange(py_trees.behaviour.Behaviour):
    """Move the ADV laterally while preserving a side-impact longitudinal gap."""

    def __init__(self, actor, target_actor, forward_speed_ms, lateral_speed_ms,
                 duration, lateral_sign, name="ForcedLaneChange"):
        super().__init__(name)
        self._actor = actor
        self._target = target_actor
        self._max_forward_speed = float(forward_speed_ms)
        self._max_lateral_speed = float(lateral_speed_ms)
        self._duration = float(duration)
        self._lateral_sign = float(lateral_sign)
        self._start_time = None
        self._last_log = -999.0
        self._desired_longitudinal_gap = 2.5

    @staticmethod
    def _clamp(value, lower, upper):
        return max(lower, min(upper, value))

    def initialise(self):
        self._start_time = GameTime.get_time()
        self._last_log = -999.0
        if self._actor is None or not self._actor.is_alive:
            return
        if self._target is not None and self._target.is_alive:
            transform = self._actor.get_transform()
            forward = transform.get_forward_vector()
            delta = self._target.get_location() - self._actor.get_location()
            initial_longitudinal = delta.x * forward.x + delta.y * forward.y
            self._desired_longitudinal_gap = self._clamp(
                initial_longitudinal * 0.25, 1.5, 3.0,
            )
        print(
            f"[BlindSpot] Cut-in begins: cap={self._max_forward_speed * 3.6:.0f} "
            f"km/h lateral={self._max_lateral_speed:.1f}m/s "
            f"sign={self._lateral_sign:+.0f} "
            f"desired_long={self._desired_longitudinal_gap:.1f}m",
            flush=True,
        )

    def update(self):
        if self._actor is None or not self._actor.is_alive:
            return py_trees.common.Status.SUCCESS
        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0
        if elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS

        transform = self._actor.get_transform()
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()
        forward = carla.Vector3D(forward.x, forward.y, 0.0)
        right = carla.Vector3D(right.x, right.y, 0.0)
        target_longitudinal = None
        target_lateral = None
        if self._target is not None and self._target.is_alive:
            delta = self._target.get_location() - self._actor.get_location()
            target_longitudinal = delta.x * forward.x + delta.y * forward.y
            target_lateral = delta.x * right.x + delta.y * right.y
            target_velocity = self._target.get_velocity()
            target_forward_speed = max(
                0.0,
                target_velocity.x * forward.x + target_velocity.y * forward.y,
            )
            longitudinal_error = target_longitudinal - self._desired_longitudinal_gap
            relative_speed = self._clamp(longitudinal_error / 0.35, -2.0, 9.0)
            forward_speed = self._clamp(
                target_forward_speed + relative_speed,
                0.0,
                self._max_forward_speed,
            )
            lateral_start_gap = self._desired_longitudinal_gap + 1.5
            if target_longitudinal > lateral_start_gap:
                lateral_speed = 0.0
                steer = 0.0
            else:
                lateral_speed = self._clamp(
                    target_lateral / 0.35,
                    -self._max_lateral_speed,
                    self._max_lateral_speed,
                )
                steer = self._clamp(target_lateral / 3.5, -1.0, 1.0) * 0.30
        else:
            forward_speed = self._max_forward_speed
            lateral_speed = self._lateral_sign * self._max_lateral_speed
            steer = 0.0

        velocity = carla.Vector3D(
            forward.x * forward_speed + right.x * lateral_speed,
            forward.y * forward_speed + right.y * lateral_speed,
            0.0,
        )
        self._actor.set_target_velocity(velocity)
        self._actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self._actor.apply_control(carla.VehicleControl(
            throttle=0.25,
            steer=steer,
            brake=0.0,
        ))

        now = GameTime.get_time()
        if now - self._last_log >= 0.5:
            self._last_log = now
            message = (
                f"[BlindSpot] Cut-in t={elapsed:.2f}s "
                f"cmd_speed={forward_speed * 3.6:.1f} "
                f"cmd_lat={lateral_speed:.1f}"
            )
            if target_longitudinal is not None:
                message += (
                    f" target_long={target_longitudinal:.1f}m"
                    f" target_lat={target_lateral:.1f}m"
                    f" desired_long={self._desired_longitudinal_gap:.1f}m"
                )
            print(message, flush=True)
        return py_trees.common.Status.RUNNING


class _StopAfterImpact(py_trees.behaviour.Behaviour):
    """Brake scenario actors shortly after the first ego contact."""

    def __init__(self, ego, actors, hold_after_s=1.5, impact_distance_m=8.0, name="StopAfterImpact"):
        super().__init__(name)
        self._ego = ego
        self._actors = actors
        self._hold_after_s = float(hold_after_s)
        self._impact_distance = float(impact_distance_m)
        self._impact_time = None

    def update(self):
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.SUCCESS
        snapshot = self._ego.get_world().get_snapshot()
        now = snapshot.timestamp.elapsed_seconds if snapshot else 0.0
        if self._impact_time is None:
            if any(
                actor is not None and actor.is_alive and
                self._ego.get_location().distance(actor.get_location()) < self._impact_distance
                for actor in self._actors
            ):
                self._impact_time = now
            else:
                return py_trees.common.Status.RUNNING

        stop = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=True)
        for actor in [self._ego] + self._actors:
            if actor is not None and actor.is_alive:
                actor.set_autopilot(False)
                actor.apply_control(stop)
        if now - self._impact_time >= self._hold_after_s:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


def _request_four_wheel_vehicle(model, transform, rolename):
    actor = CarlaDataProvider.request_new_actor(model, transform, rolename=rolename)
    if actor is not None and actor.type_id.startswith("vehicle.") and ".kawasaki." not in actor.type_id and ".yamaha." not in actor.type_id:
        return actor
    if actor is not None and actor.is_alive:
        actor.destroy()
    return None


def _spawn_exact_vehicle(world, model, transform, rolename):
    matches = list(world.get_blueprint_library().filter(model))
    if not matches:
        return None
    blueprint = matches[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", rolename)
    actor = world.try_spawn_actor(blueprint, transform)
    if actor is not None:
        CarlaDataProvider._carla_actor_pool[actor.id] = actor
        CarlaDataProvider.register_actor(actor, transform)
    return actor


# ─────────────────────────────────────────────────────────────────────────────
# Scenario
# ─────────────────────────────────────────────────────────────────────────────

class BlindSpotCutInWithSlowLeadTown06(BasicScenario):
    """
    Blind-spot lane-change collision on a multi-lane highway.

    Setup
    ─────
    • Ego drives in the right lane (lane 4) behind a slow truck.
    • A fast car (ADV) drives in the left / overtaking lane (lane 3),
      positioned in ego's blind spot (just behind and alongside).

    Collision mechanism
    ───────────────────
    After a configurable cruise phase the behavior tree forces the ego to
    steer left (simulating a driver who fails to check mirrors).  The ADV,
    travelling at a similar or slightly higher speed in lane 3, cannot avoid
    the side / rear-quarter collision.

    XML parameters (direct children of <scenario>)
    ────────────────────────────────────────────────
      activation_distance_m    — trigger proximity (m)
      truck_ahead_distance_m   — truck spawn offset ahead of ego (m)
      adv_spawn_behind_ego     — ADV spawn offset behind ego in lane 3 (m)
      truck_speed_kmh          — truck TM autopilot speed
      adv_speed_kmh            — ADV TM autopilot speed
      ego_speed_cap_kmh        — EgoSpeedGovernor cap
      ego_cap_brake            — brake force when above cap
      cruise_duration_s        — seconds before forcing lane change
      lane_change_steer        — steer value (negative = left)
      lane_change_throttle     — throttle during lane change
      lane_change_duration     — seconds of forced steering
      aftermath_duration_s     — hold time after lane change
    """

    timeout = 180

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_param(op, name, default, cast=float):
        if not op or name not in op:
            return default
        raw = op[name]
        try:
            if isinstance(raw, dict):
                return cast(raw.get("value", default))
            return cast(raw)
        except (TypeError, ValueError):
            return default

    # ── init ──────────────────────────────────────────────────────────────

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=180):

        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._ego = ego_vehicles[0]

        op = config.other_parameters if hasattr(config, "other_parameters") else {}
        self._activation_distance   = self._get_param(op, "activation_distance_m", 80.0)
        self._truck_ahead_distance  = self._get_param(op, "truck_ahead_distance_m", 43.0)
        self._adv_spawn_behind      = self._get_param(op, "adv_spawn_behind_ego", -3.0)
        self._truck_speed_kmh       = self._get_param(op, "truck_speed_kmh", 60.0)
        self._adv_speed_kmh         = self._get_param(op, "adv_speed_kmh", 65.0)
        self._ego_speed_cap_kmh     = self._get_param(op, "ego_speed_cap_kmh", 70.0)
        self._ego_cap_brake         = self._get_param(op, "ego_cap_brake", 0.20)
        self._cruise_duration       = self._get_param(op, "cruise_duration_s", 2.4)
        self._lane_change_steer     = self._get_param(op, "lane_change_steer", -0.35)
        self._lane_change_throttle  = self._get_param(op, "lane_change_throttle", 0.45)
        self._lane_change_duration  = self._get_param(op, "lane_change_duration", 0.55)
        self._aftermath_duration    = self._get_param(op, "aftermath_duration_s", 5.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)

        self._truck = None
        self._adv = None
        self._adv_spawn_wp = None

        self._camera_sensor = None
        self._frame_dir = os.path.join(
            os.path.expanduser("~"),
            "ziang/corl2026/custom_scenario/run_output/blind_spot/frames",
        )
        self._frame_count = [0]

        super().__init__(
            name="BlindSpotCutInWithSlowLeadTown06",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            terminate_on_failure=True,
            criteria_enable=criteria_enable,
        )

    # ── actors ────────────────────────────────────────────────────────────

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(self._ego.get_location())

        tm_port = CarlaDataProvider.get_traffic_manager_port()
        tm = CarlaDataProvider.get_client().get_trafficmanager(tm_port)

        # ── ADV: left lane (lane 3), BEHIND ego ──────────────────────────
        ego_left = ego_wp.get_left_lane()
        if ego_left is None or ego_left.lane_type != carla.LaneType.Driving:
            raise RuntimeError(
                f"[BlindSpot] No drivable left lane at ego. "
                f"road={ego_wp.road_id} lane={ego_wp.lane_id}"
            )
        adv_prevs = [
            wp for wp in ego_left.previous(abs(self._adv_spawn_behind))
            if wp.road_id == ego_wp.road_id and wp.lane_id == ego_left.lane_id
        ]
        self._adv_spawn_wp = adv_prevs[0] if adv_prevs else ego_left

        adv_t = carla.Transform(
            carla.Location(
                x=self._adv_spawn_wp.transform.location.x,
                y=self._adv_spawn_wp.transform.location.y,
                z=self._adv_spawn_wp.transform.location.z + 0.5,
            ),
            self._adv_spawn_wp.transform.rotation,
        )
        self._adv = None
        for adv_model in ("vehicle.tesla.model3", "vehicle.lincoln.mkz_2020", "vehicle.audi.tt", "vehicle.dodge.charger_2020"):
            self._adv = _spawn_exact_vehicle(self._world, adv_model, adv_t, "adversarial")
            if self._adv is not None:
                break
        if self._adv is None:
            raise RuntimeError("[BlindSpot] ADV spawn failed.")
        self._adv.set_target_velocity(carla.Vector3D(0, 0, 0))
        self._adv.set_simulate_physics(True)
        # Keep ADV stationary during the blind-spot setup; the cut-in controller
        # takes over after ego has opened the longitudinal gap.
        self._adv.set_autopilot(False)
        tm.set_desired_speed(self._adv, self._adv_speed_kmh)
        tm.auto_lane_change(self._adv, False)
        tm.ignore_vehicles_percentage(self._adv, 100)
        tm.ignore_walkers_percentage(self._adv, 100)
        tm.ignore_lights_percentage(self._adv, 100)
        tm.ignore_signs_percentage(self._adv, 100)

        self.other_actors.append(self._adv)
        adv_right = self._adv_spawn_wp.transform.get_right_vector()
        ego_delta = self._ego.get_location() - self._adv.get_location()
        self._adv_lateral_sign = 1.0 if (
            ego_delta.x * adv_right.x + ego_delta.y * adv_right.y >= 0.0
        ) else -1.0

        truck_nexts = ego_wp.next(max(1.0, self._truck_ahead_distance))
        truck_wp = truck_nexts[0] if truck_nexts else ego_wp
        truck_t = carla.Transform(
            carla.Location(
                x=truck_wp.transform.location.x,
                y=truck_wp.transform.location.y,
                z=truck_wp.transform.location.z + 0.5,
            ),
            truck_wp.transform.rotation,
        )
        for truck_model in (
            "vehicle.carlamotors.carlacola",
            "vehicle.mercedes.sprinter",
            "vehicle.volkswagen.t2",
        ):
            self._truck = _spawn_exact_vehicle(
                self._world, truck_model, truck_t, "blind_spot_lead",
            )
            if self._truck is not None:
                break
        if self._truck is None:
            raise RuntimeError("[BlindSpot] lead truck spawn failed.")
        self._truck.set_simulate_physics(True)
        self._truck.set_target_velocity(carla.Vector3D(0, 0, 0))
        self._truck.set_autopilot(True, tm_port)
        tm.set_desired_speed(self._truck, self._truck_speed_kmh)
        tm.auto_lane_change(self._truck, False)
        tm.ignore_vehicles_percentage(self._truck, 100)
        tm.ignore_walkers_percentage(self._truck, 100)
        tm.ignore_lights_percentage(self._truck, 100)
        tm.ignore_signs_percentage(self._truck, 100)
        self.other_actors.append(self._truck)

        print(
            f"\n[BlindSpot] ── Spawn ──────────────────────────────────────────\n"
            f"  Ego:   road={ego_wp.road_id}  lane={ego_wp.lane_id}\n"
            f"  ADV:   road={self._adv_spawn_wp.road_id}  lane={self._adv_spawn_wp.lane_id}  "
            f"{self._adv_spawn_behind:.0f}m behind  {self._adv_speed_kmh:.0f} km/h  "
            f"type={self._adv.type_id}\n"
            f"  ego cap={self._ego_speed_cap_kmh:.0f} km/h  "
            f"cruise={self._cruise_duration:.1f}s  "
            f"steer={self._lane_change_steer:.2f}  dur={self._lane_change_duration:.1f}s\n"
            f"────────────────────────────────────────────────────────────────\n",
            flush=True,
        )

    # ── trigger ───────────────────────────────────────────────────────────

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateBlindSpot",
        )

    # ── behavior tree ─────────────────────────────────────────────────────

    def _create_behavior(self):
        """
        outer = Parallel(SUCCESS_ON_ONE)
          root = Sequence
            Phase0  HoldADV braked | Timeout(8) — wait for ego to start
            Phase1  TimeOut(cruise_duration) — ADV overtakes in the left lane
            Phase2  ApplyControlForDuration(ADV, steer right) | Timeout
            Phase3  aftermath
            Cleanup ActorDestroy
          EgoSpeedGovernor        (background, always RUNNING)
          PeriodicStatusLogger    (background, always RUNNING)
          StopActorsOnEgoCollision (SUCCESS on collision → terminates outer)
          LaneCleaner             (background)
        """
        P = py_trees.common.ParallelPolicy

        root = py_trees.composites.Sequence("BlindSpot_Root")

        # ── Phase 0: hold ADV until ego starts ───────────────────────────
        phase0 = py_trees.composites.Parallel(
            "Phase0_Sync", policy=P.SUCCESS_ON_ONE,
        )
        phase0.add_child(TimeOut(8.0, name="Phase0_Timeout"))
        phase0.add_child(
            HoldUntilEgoMoves(
                self._adv, self._ego, speed_threshold=10.0, name="HoldADV",
            ),
        )
        root.add_child(phase0)

        # ── Phase 1: cruise — ADV overtakes in the left lane ─────────────
        root.add_child(TimeOut(self._cruise_duration, name="Phase1_Cruise"))

        # ── Phase 2: ADV cuts right from the blind spot into ego's lane ───
        root.add_child(_OneShot(
            lambda: self._adv.set_autopilot(False),
            name="ReleaseADVAutopilot",
        ))
        phase2 = py_trees.composites.Parallel(
            "Phase2_ADVOvertakeCutIn", policy=P.SUCCESS_ON_ONE,
        )
        phase2.add_child(_ForcedLaneChange(
            self._adv,
            target_actor=self._ego,
            forward_speed_ms=self._adv_speed_kmh / 3.6,
            lateral_speed_ms=max(4.0, abs(self._lane_change_steer) * 8.0),
            duration=self._lane_change_duration,
            lateral_sign=self._adv_lateral_sign,
            name="ForceADVCutIn",
        ))
        phase2.add_child(TimeOut(self._lane_change_duration + 0.3, name="Phase2_Timeout"))
        root.add_child(phase2)

        # ── Phase 4: aftermath hold ──────────────────────────────────────
        root.add_child(TimeOut(self._aftermath_duration, name="Phase4_Aftermath"))

        # ── Cleanup ──────────────────────────────────────────────────────
        cleanup = py_trees.composites.Sequence("Cleanup")
        cleanup.add_child(ActorDestroy(self._adv, name="DestroyADV"))
        cleanup.add_child(ActorDestroy(self._truck, name="DestroyLeadTruck"))
        root.add_child(cleanup)

        # ── Outer: background helpers ────────────────────────────────────
        ego_lane_wp = self._trigger_wp
        left_lane_wp = self._trigger_wp.get_left_lane()
        lane_wps = [
            wp for wp in [ego_lane_wp, left_lane_wp]
            if wp and wp.lane_type == carla.LaneType.Driving
        ]
        protected = [self._ego, self._adv]

        outer = py_trees.composites.Parallel(
            "BlindSpot_Outer", policy=P.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=protected,
            lane_waypoints=lane_wps,
            center_location=self._trigger_wp.transform.location,
            radius=900.0,
            interval=0.2,
            name="ScenarioLaneCleaner",
        ))
        outer.add_child(PeriodicStatusLogger(
            self._ego, self._adv, self._truck,
            interval=1.0, name="StatusLogger",
        ))
        outer.add_child(StopActorsOnEgoCollision(
            self._ego, [self._adv, self._truck],
            name="StopOnEgoCollision",
            freeze_actors_on_collision=True,
        ))
        outer.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            name="EgoSpeedGovernor",
        ))
        outer.add_child(_EgoMinSpeedForcer(
            self._ego, min_speed_kmh=55.0, throttle=0.8,
            name="EgoMinSpeedForcer",
        ))
        return outer

    # ── embedded camera ──────────────────────────────────────────────────

    def _attach_camera_sensor(self):
        try:
            os.makedirs(self._frame_dir, exist_ok=True)
            bp = self._world.get_blueprint_library().find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", "1280")
            bp.set_attribute("image_size_y", "720")
            bp.set_attribute("fov", "110")
            transform = carla.Transform(
                carla.Location(x=-6.0, z=15.0),
                carla.Rotation(pitch=-30.0),
            )
            self._camera_sensor = self._world.spawn_actor(
                bp, transform, attach_to=self._ego,
            )
            frame_dir = self._frame_dir
            frame_count = self._frame_count

            # Async writer: queue frames and write in background thread
            self._frame_queue = queue.Queue(maxsize=2000)
            self._writer_stop = threading.Event()

            def _writer():
                while not self._writer_stop.is_set():
                    try:
                        idx, raw, w, h = self._frame_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    import numpy as np
                    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
                    # BGRA -> BGR for cv2
                    bgr = arr[:, :, :3][:, :, ::-1].copy()
                    import cv2
                    cv2.imwrite(os.path.join(frame_dir, f"{idx:06d}.png"), bgr)

            self._writer_thread = threading.Thread(target=_writer, daemon=True)
            self._writer_thread.start()

            def _save(image):
                idx = frame_count[0]
                frame_count[0] += 1
                try:
                    self._frame_queue.put_nowait(
                        (idx, bytes(image.raw_data), image.width, image.height)
                    )
                except queue.Full:
                    pass  # drop frame if queue full
                if idx % 50 == 0 and idx > 0:
                    print(
                        f"[BlindSpotCam] frame {idx}",
                        flush=True,
                    )

            self._camera_sensor.listen(_save)
            print(
                f"[BlindSpotCam] camera attached (async writer), saving to {frame_dir}",
                flush=True,
            )
        except Exception as e:
            print(f"[BlindSpotCam] WARNING: {e}")
            self._camera_sensor = None

    # ── criteria & cleanup ───────────────────────────────────────────────

    def _create_test_criteria(self):
        return [CollisionTest(self._ego)]

    def __del__(self):
        try:
            if self._camera_sensor is not None:
                try:
                    self._camera_sensor.stop()
                    self._camera_sensor.destroy()
                except Exception:
                    pass
                self._camera_sensor = None

            # Wait for writer thread to drain remaining frames
            if hasattr(self, '_writer_stop') and self._writer_stop is not None:
                import time as _time
                _time.sleep(0.5)  # let last callbacks arrive
                self._writer_stop.set()
                if hasattr(self, '_writer_thread'):
                    self._writer_thread.join(timeout=30)
                # Drain any remaining items
                while not self._frame_queue.empty():
                    try:
                        idx, raw, w, h = self._frame_queue.get_nowait()
                        import numpy as np
                        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
                        bgr = arr[:, :, :3][:, :, ::-1].copy()
                        import cv2
                        cv2.imwrite(
                            os.path.join(self._frame_dir, f"{idx:06d}.png"), bgr
                        )
                    except Exception:
                        break
            _time.sleep(0.5)

            frames = self._frame_count[0] if self._frame_count else 0
            if frames > 5 and os.path.isdir(self._frame_dir):
                out_mp4 = os.path.join(
                    os.path.dirname(self._frame_dir), "recording.mp4",
                )
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-framerate", "20",
                        "-i", os.path.join(self._frame_dir, "%06d.png"),
                        "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
                        "-pix_fmt", "yuv420p", out_mp4,
                    ],
                    check=False,
                    capture_output=True,
                )
                print(f"[BlindSpotCam] Saved {frames} frames -> {out_mp4}")
        except Exception as e:
            print(f"[BlindSpotCam] WARNING encode: {e}")
        self.remove_all_actors()
