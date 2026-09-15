"""Overtaking vehicle cuts in and blocks lane (Town04).

An overtaking vehicle cuts in ahead of the ego and blocks the lane.
"""

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy,
    WaypointFollower,
)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
    TriggerVelocity,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import RemoveRoadLane

from srunner.scenariomanager.scenarioatomics.custom_atomics_overtaking_vehicle_cuts_in_and_blocks_lane_town04 import (
    PhysicsLossOfControlBreach,
    AdversarialSpeedChaser,
    WaitUntilAheadOfEgo,
    HoldUntilEgoMoves,
    ApplyControlContinuous,
    ApplyControlForDuration,
    InSameDrivingLaneAsActor,
    WaitForActorCollision,
    PeriodicStatusLogger,
    StopActorsOnEgoCollision,
    PanicDriverControl,
    SpinOutControl,
    LaneCleaner,
    EgoSpeedGovernor,
)


class DirectSideBreachControl(py_trees.behaviour.Behaviour):
    def __init__(self, actor, target_actor, forward_speed_kmh=74.0,
                 lateral_speed_ms=7.5, cut_duration=2.4, block_duration=5.0,
                 name="DirectSideBreachControl"):
        super().__init__(name)
        self._actor = actor
        self._target_actor = target_actor
        self._forward_speed = float(forward_speed_kmh) / 3.6
        self._lateral_speed = float(lateral_speed_ms)
        self._cut_duration = float(cut_duration)
        self._block_duration = float(block_duration)
        self._start_time = None
        self._forward = None
        self._right = None
        self._side_sign = 1.0
        self._last_log = -999.0

    def initialise(self):
        self._start_time = GameTime.get_time()
        self._last_log = -999.0
        if self._actor is None or not self._actor.is_alive:
            return
        transform = self._actor.get_transform()
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()
        self._forward = carla.Vector3D(forward.x, forward.y, 0.0)
        self._right = carla.Vector3D(right.x, right.y, 0.0)
        if self._target_actor is not None and self._target_actor.is_alive:
            delta = self._target_actor.get_location() - transform.location
            self._side_sign = 1.0 if (delta.x * self._right.x + delta.y * self._right.y) >= 0.0 else -1.0
        print(
            f"[HPB] Panic breach begins: forward={self._forward_speed * 3.6:.0f} km/h "
            f"lateral_peak={self._lateral_speed:.1f} m/s sign={self._side_sign:+.0f}",
            flush=True,
        )

    @property
    def _total_duration(self):
        return self._cut_duration + self._block_duration

    @staticmethod
    def _smoothstep(value):
        value = max(0.0, min(1.0, value))
        return value * value * (3.0 - 2.0 * value)

    def update(self):
        if self._actor is None or not self._actor.is_alive:
            return py_trees.common.Status.SUCCESS

        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0
        if elapsed >= self._total_duration:
            return py_trees.common.Status.SUCCESS

        if elapsed < self._cut_duration and self._forward is not None and self._right is not None:
            progress = max(0.0, min(1.0, elapsed / max(0.2, self._cut_duration)))
            lateral_gain = self._smoothstep(min(1.0, progress / 0.55))
            release_gain = 1.0 - 0.35 * self._smoothstep(max(0.0, (progress - 0.70) / 0.30))
            lateral_speed = self._lateral_speed * lateral_gain * release_gain
            forward_speed = self._forward_speed * (1.0 - 0.10 * self._smoothstep(progress))
            velocity = carla.Vector3D(
                self._forward.x * forward_speed + self._right.x * self._side_sign * lateral_speed,
                self._forward.y * forward_speed + self._right.y * self._side_sign * lateral_speed,
                0.0,
            )
            self._actor.set_target_velocity(velocity)
            self._actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.45 * self._side_sign * lateral_gain))
            self._actor.apply_control(carla.VehicleControl(throttle=0.22, steer=0.22 * self._side_sign * lateral_gain, brake=0.0))
            if elapsed - self._last_log >= 0.8:
                self._last_log = elapsed
                print(
                    f"[HPB] panic breach progress={progress:.2f} lateral={lateral_speed:.1f}m/s",
                    flush=True,
                )
        else:
            self._actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self._actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self._actor.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        return py_trees.common.Status.RUNNING


class BrakeLeadForAdversary(py_trees.behaviour.Behaviour):
    def __init__(self, lead_actor, adversary, cruise_speed_kmh=58.0,
                 brake_gap_m=45.0, min_delay_s=2.0, name="BrakeLeadForAdversary"):
        super().__init__(name)
        self._lead_actor = lead_actor
        self._adversary = adversary
        self._cruise_speed = float(cruise_speed_kmh) / 3.6
        self._brake_gap = float(brake_gap_m)
        self._min_delay = float(min_delay_s)
        self._start_time = None
        self._braking = False
        self._last_log = -999.0

    def initialise(self):
        if self._start_time is None:
            self._start_time = GameTime.get_time()

    def update(self):
        if self._lead_actor is None or self._adversary is None:
            return py_trees.common.Status.RUNNING
        if not self._lead_actor.is_alive or not self._adversary.is_alive:
            return py_trees.common.Status.RUNNING

        now = GameTime.get_time()
        elapsed = now - self._start_time if self._start_time is not None else 0.0
        lead_tf = self._lead_actor.get_transform()
        lead_loc = lead_tf.location
        adv_loc = self._adversary.get_location()
        fwd = lead_tf.get_forward_vector()
        gap = ((lead_loc.x - adv_loc.x) * fwd.x +
               (lead_loc.y - adv_loc.y) * fwd.y +
               (lead_loc.z - adv_loc.z) * fwd.z)

        if not self._braking and elapsed >= self._min_delay and 0.0 < gap <= self._brake_gap:
            self._braking = True
            print(f"[HPB] Lead vehicle hard-brakes ahead of ADV gap={gap:.1f}m", flush=True)

        if self._braking:
            self._lead_actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self._lead_actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self._lead_actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=True))
        else:
            self._lead_actor.set_target_velocity(carla.Vector3D(
                fwd.x * self._cruise_speed,
                fwd.y * self._cruise_speed,
                0.0,
            ))
            self._lead_actor.apply_control(carla.VehicleControl(throttle=0.35, brake=0.0, steer=0.0))

        if now - self._last_log >= 1.0:
            self._last_log = now
            mode = "BRAKE" if self._braking else "cruise"
            print(f"[HPB] lead={mode} gap_to_adv={gap:.1f}m", flush=True)
        return py_trees.common.Status.RUNNING


class WaitUntilCloseBehindLead(py_trees.behaviour.Behaviour):
    def __init__(self, adversary, lead_actor, gap_m=45.0, name="WaitUntilCloseBehindLead"):
        super().__init__(name)
        self._adversary = adversary
        self._lead_actor = lead_actor
        self._gap_m = float(gap_m)
        self._lat_tolerance = 2.6

    def update(self):
        if self._adversary is None or self._lead_actor is None:
            return py_trees.common.Status.RUNNING
        if not self._adversary.is_alive or not self._lead_actor.is_alive:
            return py_trees.common.Status.RUNNING

        lead_tf = self._lead_actor.get_transform()
        lead_loc = lead_tf.location
        adv_loc = self._adversary.get_location()
        fwd = lead_tf.get_forward_vector()
        right = lead_tf.get_right_vector()
        dx = lead_loc.x - adv_loc.x
        dy = lead_loc.y - adv_loc.y
        gap = dx * fwd.x + dy * fwd.y
        lateral = dx * right.x + dy * right.y
        if 0.0 < gap <= self._gap_m and abs(lateral) <= self._lat_tolerance:
            print(f"[HPB] ADV forced to breach by braking lead gap={gap:.1f}m", flush=True)
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

class OvertakingVehicleCutsInAndBlocksLaneTown04(BasicScenario):
    """
    Near-miss evasive maneuver scenario — highway sudden loss of control.

    ADV is in the left lane behind ego. It approaches at high speed, and once it is
    ahead of ego by `ahead_trigger_distance_m` metres, it suddenly executes a violent
    right-steer breach into ego's lane with zero warning. It then immediately enters
    `PanicDriverControl` for `panic_duration` seconds — a physics-driven delayed
    overcorrection loop that amplifies yaw instability each cycle.

    Ego has the panic_duration window to decide: brake, evade right, or hold.
    After panic control releases, the car coasts for 3 s and physics determines the
    terminal state (spin-out, lane block, barrier hit) — no artificial random modes.

    NPC runs in the right lane throughout as a parallel obstacle, limiting ego's
    rightward evasion options.

    XML other_parameters:
      activation_distance_m       — activate this far before trigger (m)
      adv_spawn_behind_ego        — ADV spawn offset behind ego (m)
      npc_spawn_behind_ego        — NPC spawn offset behind ego (m)
      approach_speed_kmh          — ADV left-lane approach speed (km/h)
      npc_speed_kmh               — NPC right-lane cruise speed (km/h)
      ahead_trigger_distance_m    — breach fires when ADV is this far ahead of ego (m)
      breach_steer                — steer for initial breach (+= right, toward ego lane)
      breach_duration             — seconds of hard steer during breach
      panic_duration              — PanicDriverControl duration (ego's decision window, s)
      panic_reaction_delay        — human reaction lag (s)
      panic_gain                  — overshoot factor; >1.0 guarantees instability
      yaw_rate_scale              — normalisation deg/s → steer=1.0 (tune for vehicle)
    """

    timeout = 180

    @staticmethod
    def _get_param(op, name, default, cast=float):
        """Read an XML other_parameter robustly."""
        if not op or name not in op:
            return default
        raw = op[name]
        try:
            if isinstance(raw, dict):
                return cast(raw.get("value", default))
            return cast(raw)
        except (TypeError, ValueError):
            return default

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False, criteria_enable=True, timeout=180):

        self._world = world
        self._map   = CarlaDataProvider.get_map()
        self._ego   = ego_vehicles[0]

        op = dict(config.other_parameters) if hasattr(config, "other_parameters") else {}
        if hasattr(config, "scenario") and hasattr(config.scenario, "other_parameters"):
            op.update(config.scenario.other_parameters)
        self._activation_distance    = self._get_param(op, "activation_distance_m", 140.0)
        self._adv_spawn_behind       = self._get_param(op, "adv_spawn_behind_ego", 55.0)
        self._npc_ahead_distance     = self._get_param(op, "npc_ahead_distance_m", 180.0)
        self._approach_speed         = self._get_param(op, "approach_speed_kmh", 118.0) / 3.6
        self._npc_speed              = self._get_param(op, "npc_speed_kmh", 58.0) / 3.6
        self._lead_brake_gap         = self._get_param(op, "lead_brake_gap_m", 45.0)
        self._lead_brake_delay       = self._get_param(op, "lead_brake_delay_s", 2.0)
        self._lead_breach_gap        = self._get_param(op, "lead_breach_gap_m", 45.0)
        self._ahead_trigger_dist  = self._get_param(op, "ahead_trigger_distance_m", 1.0)
        self._cut_in_steer        = self._get_param(op, "cut_in_steer", 0.72)
        self._cut_in_throttle     = self._get_param(op, "cut_in_throttle", 0.65)
        self._cut_in_duration     = self._get_param(op, "cut_in_duration", 1.60)
        self._drift_steer         = self._get_param(op, "drift_steer", 0.15)
        self._drift_steer_sign    = self._get_param(op, "drift_steer_sign", -1.0)
        self._drift_throttle      = self._get_param(op, "drift_throttle", 0.15)
        self._drift_brake         = self._get_param(op, "drift_brake", 0.25)
        self._drift_duration      = self._get_param(op, "drift_duration", 0.70)
        self._spin_throttle       = self._get_param(op, "spin_throttle", 0.00)
        self._spin_duration       = self._get_param(op, "spin_duration", 0.50)
        self._block_duration      = self._get_param(op, "block_duration", 6.00)
        self._direct_breach_forward_kmh = self._get_param(op, "direct_breach_forward_kmh", 74.0)
        self._direct_breach_lateral_ms = self._get_param(op, "direct_breach_lateral_ms", 7.5)
        self._ego_speed_cap_kmh   = self._get_param(op, "ego_speed_cap_kmh", 72.0)
        self._ego_cap_brake       = self._get_param(op, "ego_cap_brake", 0.12)
        self._road_friction_scale = self._get_param(op, "road_friction_scale", 1.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)

        self._adversarial  = None
        self._npc_parallel = None
        self._left_lane_wps = []
        self._ego_collision_sensor = None
        self._ego_hit_adversary = False
        self._post_impact_static = False
        self._post_impact_transforms = {}

        super().__init__(
            name="OvertakingVehicleCutsInAndBlocksLaneTown04",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            terminate_on_failure=True,
            criteria_enable=criteria_enable,
        )

    def _initialize_environment(self, world):
        """Apply weather and optional global road-friction trigger."""
        super()._initialize_environment(world)

        friction_scale = max(0.1, float(self._road_friction_scale))
        if abs(friction_scale - 1.0) < 1e-6:
            return

        friction_bp = world.get_blueprint_library().find("static.trigger.friction")
        extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
        friction_bp.set_attribute("friction", str(friction_scale))
        friction_bp.set_attribute("extent_x", str(extent.x))
        friction_bp.set_attribute("extent_y", str(extent.y))
        friction_bp.set_attribute("extent_z", str(extent.z))

        transform = carla.Transform()
        transform.location = carla.Location(-10000.0, -10000.0, 0.0)
        try:
            world.spawn_actor(friction_bp, transform)
        except RuntimeError:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # ACTOR SPAWNING
    # ─────────────────────────────────────────────────────────────────────

    def _initialize_actors(self, config):
        """Spawn ADV (left lane) and NPC (right lane) behind ego."""
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        # ── ADV: same-direction adjacent lane, adv_spawn_behind m behind ego ──
        ego_forward = ego_wp.transform.get_forward_vector()
        adjacent_candidates = (
            (ego_wp.get_right_lane(), True),
            (ego_wp.get_left_lane(), False),
        )
        ego_adjacent_wp = None
        breach_from_right = False
        for candidate, from_right in adjacent_candidates:
            if candidate is None or candidate.lane_type != carla.LaneType.Driving:
                continue
            if candidate.road_id != ego_wp.road_id:
                continue
            candidate_forward = candidate.transform.get_forward_vector()
            alignment = (ego_forward.x * candidate_forward.x +
                         ego_forward.y * candidate_forward.y)
            if alignment > 0.5:
                ego_adjacent_wp = candidate
                breach_from_right = from_right
                break
        if ego_adjacent_wp is None:
            raise RuntimeError(
                f"[HPB] No same-direction drivable adjacent lane at trigger. "
                f"road_id={self._trigger_wp.road_id}, lane_id={self._trigger_wp.lane_id}"
            )
        adv_prevs = [
            wp for wp in ego_adjacent_wp.previous(self._adv_spawn_behind)
            if wp.road_id == ego_wp.road_id
            and wp.lane_id == ego_adjacent_wp.lane_id
        ]
        adv_spawn_wp = adv_prevs[0] if adv_prevs else ego_adjacent_wp

        # ── Lead/brake car: same adjacent lane, ahead of ADV.
        # It cruises first, then hard-brakes when ADV closes in, creating the panic breach trigger.
        npc_spawn_wp = adv_spawn_wp
        traversed_m = 0.0
        while traversed_m < self._npc_ahead_distance:
            candidates = [
                wp for wp in npc_spawn_wp.next(2.0)
                if wp.lane_id == adv_spawn_wp.lane_id
                and wp.lane_type == carla.LaneType.Driving
            ]
            if not candidates:
                break
            npc_spawn_wp = candidates[0]
            traversed_m += 2.0

        npc_spawn_candidates = [npc_spawn_wp]
        for offset_m in (12.0, 24.0, 36.0):
            for candidate in (
                npc_spawn_wp.next(offset_m) + npc_spawn_wp.previous(offset_m)
            ):
                if (
                    candidate.lane_id == adv_spawn_wp.lane_id
                    and candidate.lane_type == carla.LaneType.Driving
                ):
                    npc_spawn_candidates.append(candidate)

        # ── Pre-compute left-lane waypoints from ADV spawn (800 m) ─────────
        # 800 m ensures plan never runs out during a 25-s phase at 120 km/h (= 833 m max).
        # Do NOT apply lane-id correction inside the loop — it silently appends wrong-lane
        # waypoints when get_left_lane() returns None, causing WaypointFollower to brake.
        self._left_lane_wps = []
        wp = adv_spawn_wp
        dist = 0.0
        while dist < 800.0:
            nexts = wp.next(2.0)
            if not nexts:
                break
            wp = nexts[0]
            dist += 2.0
            self._left_lane_wps.append(wp)

        # NPC uses TM autopilot — no waypoint plan needed.

        print(
            f"\n[HPB] ── Spawn ──────────────────────────────────────────────────\n"
            f"  Ego: road_id={ego_wp.road_id}, lane_id={ego_wp.lane_id}\n"
            f"  ADV: road_id={adv_spawn_wp.road_id}, lane_id={adv_spawn_wp.lane_id} "
            f"side={'right' if breach_from_right else 'left'} "
            f"({self._adv_spawn_behind:.0f}m behind)  plan={len(self._left_lane_wps)} wps "
            f"(~{len(self._left_lane_wps)*2:.0f}m)\n"
            f"  LeadBrake: road_id={npc_spawn_wp.road_id}, lane_id={npc_spawn_wp.lane_id} "
            f"({self._npc_ahead_distance:.0f}m AHEAD of ADV)  cruise={self._npc_speed*3.6:.0f} km/h "
            f"brake_gap={self._lead_brake_gap:.0f}m breach_gap={self._lead_breach_gap:.0f}m\n"
            f"  approach={self._approach_speed * 3.6:.0f} km/h  "
            f"ahead_trigger={self._ahead_trigger_dist:.1f}m\n"
            f"  spinout: cut_in={self._cut_in_steer:.2f}steer/{self._cut_in_duration:.2f}s  "
            f"drift={self._drift_steer:.2f}steer/{self._drift_brake:.2f}brk/{self._drift_duration:.1f}s  "
            f"spin={self._spin_throttle:.2f}thr/{self._spin_duration:.1f}s  "
            f"block={self._block_duration:.1f}s\n"
            f"────────────────────────────────────────────────────────────────────\n",
            flush=True,
        )

        # ── Spawn ADV ───────────────────────────────────────────────────────
        adv_t = carla.Transform(
            carla.Location(
                x=adv_spawn_wp.transform.location.x,
                y=adv_spawn_wp.transform.location.y,
                z=adv_spawn_wp.transform.location.z + 0.5,
            ),
            adv_spawn_wp.transform.rotation,
        )
        self._adversarial = CarlaDataProvider.request_new_actor(
            "vehicle.tesla.model3", adv_t, rolename="adversarial"
        )
        if self._adversarial is None:
            raise RuntimeError("[HPB] ADV spawn failed — location occupied.")
        self._adversarial.set_target_velocity(carla.Vector3D(0, 0, 0))
        self._adversarial.set_simulate_physics(True)
        self.other_actors.append(self._adversarial)

        # ── Spawn lead/brake car ─────────────────────────────────────────────
        self._npc_parallel = None
        for candidate in npc_spawn_candidates:
            npc_t = carla.Transform(
                carla.Location(
                    x=candidate.transform.location.x,
                    y=candidate.transform.location.y,
                    z=candidate.transform.location.z + 0.5,
                ),
                candidate.transform.rotation,
            )
            self._npc_parallel = CarlaDataProvider.request_new_actor(
                "vehicle.tesla.model3", npc_t, rolename="adv_lane_brake_lead"
            )
            if self._npc_parallel is not None:
                npc_spawn_wp = candidate
                break
        if self._npc_parallel is None:
            raise RuntimeError("[HPB] Lead brake vehicle spawn failed — location occupied.")
        self._npc_parallel.set_simulate_physics(True)
        self._npc_parallel.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))

        self.other_actors.append(self._npc_parallel)

        try:
            adv_id = self._adversarial.id
            cbp = self._world.get_blueprint_library().find("sensor.other.collision")
            self._ego_collision_sensor = self._world.spawn_actor(
                cbp, carla.Transform(), attach_to=self._ego)

            def _on_ego_collision(event, adv_id=adv_id):
                other = event.other_actor
                if other is not None and other.id == adv_id:
                    self._ego_hit_adversary = True
                    print(f"[HPB] EGO COLLISION with {other.type_id} (ADVERSARY)", flush=True)
            self._ego_collision_sensor.listen(_on_ego_collision)
        except RuntimeError:
            self._ego_collision_sensor = None

        self._post_impact_static = False
        self._post_impact_transforms = {}

        print(
            f"[HPB] ADV id={self._adversarial.id} at {self._adversarial.get_location()}\n"
            f"[HPB] LeadBrake id={self._npc_parallel.id} at {self._npc_parallel.get_location()}\n",
            flush=True,
        )

    @staticmethod
    def _stop_actor_motion(actor):
        if actor is None or not actor.is_alive:
            return
        zero = carla.Vector3D(0.0, 0.0, 0.0)
        try:
            actor.set_target_velocity(zero)
            actor.set_target_angular_velocity(zero)
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=True))
        except RuntimeError:
            pass

    def _hold_post_collision_static(self):
        actors = [actor for actor in (self._adversarial, self._npc_parallel)
                  if actor is not None and actor.is_alive]
        if not self._post_impact_static:
            self._post_impact_static = True
            self._post_impact_transforms = {actor.id: actor.get_transform() for actor in actors}
            for actor in actors:
                try:
                    actor.set_simulate_physics(False)
                except RuntimeError:
                    pass
            print("[HPB] post-impact static hold engaged", flush=True)

        for actor in actors:
            transform = self._post_impact_transforms.get(actor.id)
            if transform is not None:
                try:
                    actor.set_transform(transform)
                except RuntimeError:
                    pass
            self._stop_actor_motion(actor)

    class _PostImpactStaticHold(py_trees.behaviour.Behaviour):
        def __init__(self, scenario, name="PostImpactStaticHold"):
            super().__init__(name)
            self._scenario = scenario
            self._last_log = -999.0
            self._hold_start = None

        def update(self):
            scenario = self._scenario
            if not getattr(scenario, "_ego_hit_adversary", False):
                return py_trees.common.Status.RUNNING
            now = GameTime.get_time()
            if self._hold_start is None:
                self._hold_start = now
            scenario._hold_post_collision_static()
            if now - self._last_log >= 1.0:
                self._last_log = now
                print(f"[HPB] t={now:.1f}s post-impact static hold", flush=True)
            if now - self._hold_start >= 3.0:
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING

    # ─────────────────────────────────────────────────────────────────────
    # SCENARIO TRIGGER
    # ─────────────────────────────────────────────────────────────────────

    def _setup_scenario_trigger(self, config):
        """
        Use Euclidean distance trigger (not InTriggerDistanceToLocationAlongRoute).
        In leaderboard_2, config.route is (Transform, RoadOption) tuples and the
        along-route helper fails with 'Transform has no attribute x'.
        With activation_distance_m=140 and ego start ~124 m from trigger,
        the condition fires immediately at ego spawn — same pattern as AdversarialLaneBreach.
        """
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateHighwayPanicBreach",
        )

    # ─────────────────────────────────────────────────────────────────────
    # BEHAVIOR TREE
    # ─────────────────────────────────────────────────────────────────────

    def _create_behavior(self):
        """
        outer = Parallel(SUCCESS_ON_ONE)
          root = Sequence
            Phase0_Sync          Parallel(SUCCESS_ON_ALL): HoldUntilEgoMoves x2
            Phase1_Approach      Parallel(SUCCESS_ON_ONE): timeout + WaitUntilAheadOfEgo
                                   + WaypointFollower(ADV) + WaypointFollower(NPC)
            Phase2_Breach        Parallel(SUCCESS_ON_ONE): ApplyControlForDuration(ADV)
                                   + WaypointFollower(NPC) + timeout
            Phase3_PanicDriver   Parallel(SUCCESS_ON_ONE): PanicDriverControl(ADV)
                                   + WaypointFollower(NPC) + timeout
            Phase4_PhysicsCoast  TimeOut(3.0) — no control, physics resolves freely
            Cleanup              Sequence: ActorDestroy x2
          PeriodicStatusLogger   (background, always RUNNING)
          StopActorsOnEgoCollision (SUCCESS on collision → terminates outer)
        """
        from agents.navigation.local_planner import RoadOption
        P = py_trees.common.ParallelPolicy

        root = py_trees.composites.Sequence("HighwayPanicBreach_Root")

        # ── Phase 0: hold ADV until TF++ produces first control (or 8 s timeout) ──
        # NPC is AHEAD of ego — no need to hold it; it will start moving in Phase 1.
        # SUCCESS_ON_ONE: exits as soon as ego moves OR after 8 s (standalone/debug).
        phase0 = py_trees.composites.Parallel("Phase0_Sync", policy=P.SUCCESS_ON_ONE)
        phase0.add_child(TimeOut(8.0, name="Phase0_Timeout"))
        phase0.add_child(HoldUntilEgoMoves(
            self._adversarial, self._ego, name="HoldADV"
        ))
        root.add_child(phase0)

        def left_plan():
            return [(wp, RoadOption.LANEFOLLOW) for wp in self._left_lane_wps]

        # ── Phase 1: ADV approaches in left lane until ahead_trigger_dist ahead of ego ──
        # NPC is on TM autopilot — no WaypointFollower needed.
        phase1 = py_trees.composites.Parallel("Phase1_Approach", policy=P.SUCCESS_ON_ONE)
        phase1.add_child(TimeOut(25.0, name="Phase1_Timeout"))
        phase1.add_child(WaitUntilAheadOfEgo(
            self._adversarial, self._ego,
            ahead_distance=self._ahead_trigger_dist,
            name="WaitADVAhead",
        ))
        phase1.add_child(WaitUntilCloseBehindLead(
            self._adversarial, self._npc_parallel,
            gap_m=self._lead_breach_gap,
            name="WaitADVCloseToBrakingLead",
        ))
        phase1.add_child(WaypointFollower(
            self._adversarial,
            target_speed=self._approach_speed,
            plan=left_plan(),
            name="ADV_Approach",
        ))
        root.add_child(phase1)

        phase2_timeout = self._cut_in_duration + self._block_duration + 2.0
        phase2 = py_trees.composites.Parallel("Phase2_SpinOut", policy=P.SUCCESS_ON_ONE)
        phase2.add_child(DirectSideBreachControl(
            self._adversarial,
            self._ego,
            forward_speed_kmh=self._direct_breach_forward_kmh,
            lateral_speed_ms=self._direct_breach_lateral_ms,
            cut_duration=self._cut_in_duration,
            block_duration=self._block_duration,
            name="ADV_DirectSideBreach",
        ))
        phase2.add_child(TimeOut(phase2_timeout, name="Phase2_Timeout"))
        root.add_child(phase2)

        # ── Phase 3: scene hold — ADV blocked, ego resolves ──────────────────
        root.add_child(TimeOut(3.0, name="Phase3_Hold"))

        # ── Cleanup ───────────────────────────────────────────────────────────
        cleanup = py_trees.composites.Sequence("Cleanup")
        cleanup.add_child(ActorDestroy(self._adversarial, name="DestroyADV"))
        cleanup.add_child(ActorDestroy(self._npc_parallel, name="DestroyNPC"))
        root.add_child(cleanup)

        # ── Outer wrapper: background logger + collision stopper + lane cleaner ─
        # LaneCleaner runs every second and destroys any ambient vehicle that spawns
        # in the ego lane or ADV's left lane within 500 m of the trigger point.
        # Works without BackgroundActivity — direct actor scan + destroy.
        ego_lane_wp  = self._trigger_wp
        left_lane_wp = self._trigger_wp.get_left_lane()
        lane_wps     = [wp for wp in [ego_lane_wp, left_lane_wp]
                        if wp and wp.lane_type == carla.LaneType.Driving]
        protected    = [self._ego, self._adversarial, self._npc_parallel]

        outer = py_trees.composites.Parallel(
            "HighwayPanicBreach_Logged", policy=P.SUCCESS_ON_ONE
        )
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=protected,
            lane_waypoints=lane_wps,
            center_location=self._trigger_wp.transform.location,
            radius=500.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        outer.add_child(BrakeLeadForAdversary(
            self._npc_parallel, self._adversarial,
            cruise_speed_kmh=self._npc_speed * 3.6,
            brake_gap_m=self._lead_brake_gap,
            min_delay_s=self._lead_brake_delay,
            name="LeadHardBrake",
        ))
        outer.add_child(PeriodicStatusLogger(
            self._ego, self._adversarial, self._npc_parallel,
            interval=1.0, name="StatusLogger",
        ))
        outer.add_child(self._PostImpactStaticHold(self))
        outer.add_child(StopActorsOnEgoCollision(
            self._ego, [self._adversarial, self._npc_parallel],
            name="StopOnEgoCollision",
        ))
        outer.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            name="EgoSpeedGovernor",
        ))
        return outer

    # ─────────────────────────────────────────────────────────────────────
    # CRITERIA & CLEANUP
    # ─────────────────────────────────────────────────────────────────────

    def _create_test_criteria(self):
        return [CollisionTest(self._ego)]

    def remove_all_actors(self):
        if self._ego_collision_sensor is not None and self._ego_collision_sensor.is_alive:
            self._ego_collision_sensor.stop()
            self._ego_collision_sensor.destroy()
            self._ego_collision_sensor = None
        super().remove_all_actors()

    def __del__(self):
        self.remove_all_actors()
