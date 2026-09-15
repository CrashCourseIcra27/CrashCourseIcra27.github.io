"""Oncoming motorcycle during left turn (Town03).

An oncoming motorcycle approaches at speed while the ego is committed to a left turn.
"""

import math
import os

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy,
    TrafficLightFreezer,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
)
from srunner.scenariomanager.scenarioatomics.custom_atomics_oncoming_motorcycle_during_left_turn_town03 import (
    CarlaRolloutLogger,
    EgoSpeedGovernor,
    HoldUntilEgoMoves,
    LaneCleaner,
)
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenarios.basic_scenario import BasicScenario


def _speed_mps(actor):
    if actor is None or not actor.is_alive:
        return 0.0
    v = actor.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def _planar_distance(loc_a, loc_b):
    return math.hypot(loc_a.x - loc_b.x, loc_a.y - loc_b.y)


def _yaw_delta_deg(a_deg, b_deg):
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def _signed_yaw_delta_deg(a_deg, b_deg):
    return (a_deg - b_deg + 180.0) % 360.0 - 180.0


class _RunMotorcycleAttack(py_trees.behaviour.Behaviour):
    """
    Phase 1 controller:
      - Motorcycle waits stationary until the ego commits into its turn.
      - Then it accelerates from the opposing lane straight at the ego's
        front/front-right at full speed, with physics ENABLED so the real
        collision response fires the ego collision sensor.
      - On impact (scenario._impact_detected, set by the ego collision
        sensor) it applies an upward/forward fling so the bike is visibly
        launched, then returns SUCCESS. Physics stays ON so the bike tumbles
        and falls naturally afterwards.
    """

    def __init__(self, scenario, name="RunMotorcycleAttack"):
        super().__init__(name)
        self._scenario = scenario
        self._start_time = None
        self._attack_started = False
        self._last_log = -999.0
        self._fling_applied = False

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        scenario = self._scenario

        # --- Post-impact: fling the bike, keep physics ON, finish phase ---
        if scenario._impact_detected:
            if not self._fling_applied:
                self._fling_applied = True
                moto = scenario._motorcycle
                if moto and moto.is_alive:
                    fwd = moto.get_transform().get_forward_vector()
                    # Throw the bike up and forward off the ego and give it a
                    # strong sideways tumble (roll) so it flips over rather
                    # than landing back on its wheels and riding away. The
                    # _PostImpactHold phase then lays it on its side on the
                    # road and freezes it.
                    moto.set_simulate_physics(True)
                    moto.set_target_velocity(carla.Vector3D(
                        fwd.x * 6.0, fwd.y * 6.0, 6.5))
                    moto.set_target_angular_velocity(
                        carla.Vector3D(11.0, 2.0, 4.0))
                print("[MotoHang] *** COLLISION - motorcycle launched ***",
                      flush=True)
            return py_trees.common.Status.SUCCESS

        elapsed = (GameTime.get_time() - self._start_time
                   if self._start_time else 0.0)
        ego_loc = scenario._ego.get_location()
        ego_tf = scenario._ego.get_transform()
        dist_to_conflict = _planar_distance(ego_loc, scenario._impact_point)
        turn_delta = _yaw_delta_deg(
            ego_tf.rotation.yaw, scenario._initial_ego_yaw)
        turn_committed = turn_delta >= scenario._turn_commit_yaw_deg
        if (scenario._straight_attack_without_turn and
                abs(scenario._forced_turn_steer) < 1e-3 and
                scenario._forced_turn_start_distance <= 0.0):
            turn_committed = True

        if (not self._attack_started and turn_committed
                and dist_to_conflict <= scenario._trigger_distance):
            self._attack_started = True
            print(
                f"[MotoHang] Ego committed to turn "
                f"(yaw_delta={turn_delta:.1f}deg dist={dist_to_conflict:.1f}m) "
                f"- motorcycle GO!",
                flush=True,
            )

        moto = scenario._motorcycle
        if moto is not None and moto.is_alive:
            if self._attack_started:
                # Aim at the ego front / front-right so the bike strikes the
                # turning Tesla head-on and is thrown upward.
                ego_fwd = ego_tf.get_forward_vector()
                ego_right = ego_tf.get_right_vector()
                target = carla.Location(
                    x=ego_loc.x + ego_fwd.x * 1.6 + ego_right.x * 0.9,
                    y=ego_loc.y + ego_fwd.y * 1.6 + ego_right.y * 0.9,
                    z=ego_loc.z,
                )
                moto_loc = moto.get_location()
                dx = target.x - moto_loc.x
                dy = target.y - moto_loc.y
                mag = math.hypot(dx, dy)
                speed = scenario._moto_speed
                # Drive at full speed until almost in contact; within 3.5 m
                # stop overriding velocity so momentum + the real physics
                # collision response carry it into the ego (firing the ego
                # collision sensor) instead of being fought by
                # set_target_velocity.
                if mag > 3.5:
                    moto.set_target_velocity(carla.Vector3D(
                        speed * dx / mag, speed * dy / mag, 0.0))
                    moto.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                moto.apply_control(carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=0.0))
            else:
                moto.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                moto.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=1.0, steer=0.0))

        if elapsed - self._last_log >= 1.0:
            self._last_log = elapsed
            ego_spd = _speed_mps(scenario._ego) * 3.6
            moto_spd = _speed_mps(moto) * 3.6 if moto else 0
            print(
                f"[MotoHang] t={elapsed:.1f}s  ego={ego_spd:.0f}km/h  "
                f"moto={moto_spd:.0f}km/h  dist={dist_to_conflict:.1f}m  "
                f"yaw_delta={turn_delta:.1f}  "
                f"attack={self._attack_started}  "
                f"impact={scenario._impact_detected}",
                flush=True,
            )

        if elapsed >= scenario._max_run_time:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING

class _PostImpactHold(py_trees.behaviour.Behaviour):
    """Let the bike fly off and tumble under real physics after the hit.

    The previous version teleported the bike along a scripted arc, which
    looked like it "flew away and came back". Now we simply keep physics
    enabled and wait, so the launched motorcycle follows a natural ballistic
    tumble and settles on the road.
    """

    def __init__(self, scenario, duration, freeze_delay=0.0,
                 flight_duration=0.65, arc_height=1.1,
                 name="PostImpactHold"):
        super().__init__(name)
        self._scenario = scenario
        self._duration = float(duration)
        self._start_time = None
        self._tumble_time = 0.7
        self._laid_down = False

    def initialise(self):
        self._start_time = GameTime.get_time()
        self._laid_down = False

    def update(self):
        now = GameTime.get_time()
        elapsed = now - self._start_time if self._start_time else 0.0
        scenario = self._scenario
        moto = scenario._motorcycle

        if moto is not None and moto.is_alive:
            if elapsed < self._tumble_time:
                try:
                    moto.set_simulate_physics(True)
                except RuntimeError:
                    pass
            elif not self._laid_down:
                self._laid_down = True
                loc = moto.get_location()
                wp = scenario._map.get_waypoint(
                    loc, project_to_road=True,
                    lane_type=carla.LaneType.Any)
                ground_z = wp.transform.location.z if wp else loc.z
                yaw = moto.get_transform().rotation.yaw
                downed = carla.Transform(
                    carla.Location(x=loc.x, y=loc.y, z=ground_z + 0.35),
                    carla.Rotation(pitch=0.0, yaw=yaw, roll=88.0),
                )
                try:
                    moto.set_target_velocity(carla.Vector3D(0, 0, 0))
                    moto.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                    moto.set_simulate_physics(False)
                    moto.set_transform(downed)
                except RuntimeError:
                    pass
                print(
                    f"[MotoHang] Motorcycle down on its side at "
                    f"({loc.x:.1f}, {loc.y:.1f}, {ground_z + 0.35:.1f})",
                    flush=True,
                )
            else:
                # Keep it pinned on its side (physics already off).
                pass

        # Ego: come to a FULL stop in its committed left-turn pose. Hard
        # brake + handbrake and actively kill velocity so the agent cannot
        # straighten out and drive forward after the hit.
        ego = scenario._ego
        if ego is not None and ego.is_alive:
            ego.apply_control(carla.VehicleControl(
                throttle=0.0,
                steer=scenario._forced_turn_steer,
                brake=1.0,
                hand_brake=True,
            ))
            try:
                ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                ego.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            except RuntimeError:
                pass
        if elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class _SideWindForce(py_trees.behaviour.Behaviour):
    """Apply a continuous lateral force to the ego and motorcycle."""

    def __init__(self, scenario, name="SideWindForce"):
        super().__init__(name)
        self._scenario = scenario
        self._last_log = -999.0

    def update(self):
        scenario = self._scenario
        if scenario._side_wind_force <= 0.0:
            return py_trees.common.Status.RUNNING

        yaw = math.radians(scenario._side_wind_yaw_deg)
        force = carla.Vector3D(
            math.cos(yaw) * scenario._side_wind_force,
            math.sin(yaw) * scenario._side_wind_force,
            0.0,
        )
        for actor in (scenario._ego, scenario._motorcycle):
            if actor is not None and actor.is_alive and hasattr(actor, "add_force"):
                actor.add_force(force)

        now = GameTime.get_time()
        if now - self._last_log >= 2.0:
            self._last_log = now
            print(
                f"[MotoHang] side_wind force={scenario._side_wind_force:.0f}N "
                f"yaw={scenario._side_wind_yaw_deg:.0f}deg",
                flush=True,
            )
        return py_trees.common.Status.RUNNING


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    """Override autopilot braking — force ego to maintain minimum speed."""

    def __init__(self, ego, min_speed_kmh=25.0, throttle=1.0,
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
        if speed < self._min_mps:
            ctrl = self._ego.get_control()
            ctrl.brake = 0.0
            ctrl.throttle = self._throttle
            self._ego.apply_control(ctrl)
        return py_trees.common.Status.RUNNING


class _EgoIntersectionLeftTurnForcer(py_trees.behaviour.Behaviour):
    """Force the ego to visibly turn left once it reaches the junction."""

    def __init__(self, scenario, name="EgoIntersectionLeftTurnForcer"):
        super().__init__(name)
        self._scenario = scenario
        self._active_since = None
        self._last_log = -999.0

    def update(self):
        scenario = self._scenario
        ego = scenario._ego
        if ego is None or not ego.is_alive or scenario._impact_detected:
            return py_trees.common.Status.RUNNING

        now = GameTime.get_time()
        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        dist_to_conflict = _planar_distance(ego_loc, scenario._impact_point)
        signed_turn = _signed_yaw_delta_deg(
            ego_tf.rotation.yaw, scenario._initial_ego_yaw)
        turn_delta = abs(signed_turn)

        should_start = dist_to_conflict <= scenario._forced_turn_start_distance
        still_turning = (
            self._active_since is not None and
            now - self._active_since <= scenario._forced_turn_duration and
            turn_delta < scenario._forced_turn_end_yaw_deg
        )

        if should_start and self._active_since is None:
            self._active_since = now
            print(
                f"[MotoHang] Ego forced LEFT turn start "
                f"(dist={dist_to_conflict:.1f}m yaw_signed={signed_turn:.1f})",
                flush=True,
            )

        if (self._active_since is None and
                dist_to_conflict <= scenario._straight_hold_start_distance):
            ego.apply_control(carla.VehicleControl(
                throttle=scenario._straight_hold_throttle,
                steer=0.0,
                brake=0.0,
            ))
            if now - self._last_log >= 0.5:
                self._last_log = now
                print(
                    f"[MotoHang] holding straight before left turn: "
                    f"dist={dist_to_conflict:.1f}m yaw_signed={signed_turn:.1f}",
                    flush=True,
                )

        if self._active_since is not None and still_turning:
            ego.apply_control(carla.VehicleControl(
                throttle=scenario._forced_turn_throttle,
                steer=scenario._forced_turn_steer,
                brake=0.0,
            ))
            if now - self._last_log >= 0.5:
                self._last_log = now
                print(
                    f"[MotoHang] forcing left: dist={dist_to_conflict:.1f}m "
                    f"yaw_signed={signed_turn:.1f} steer={scenario._forced_turn_steer:.2f}",
                    flush=True,
                )

        return py_trees.common.Status.RUNNING


class _WaitForMotorcycleImpact(py_trees.behaviour.Behaviour):
    def __init__(self, scenario, name="WaitForMotorcycleImpact"):
        super().__init__(name)
        self._scenario = scenario

    def update(self):
        return (py_trees.common.Status.SUCCESS
                if self._scenario._impact_detected
                else py_trees.common.Status.RUNNING)


class _WaitForMotorcyclePostImpact(py_trees.behaviour.Behaviour):
    def __init__(self, scenario, name="WaitForMotorcyclePostImpact"):
        super().__init__(name)
        self._scenario = scenario

    def update(self):
        if not self._scenario._impact_detected:
            return py_trees.common.Status.RUNNING
        impact_time = self._scenario._impact_time
        if impact_time is None:
            return py_trees.common.Status.RUNNING
        elapsed = self._scenario._world.get_snapshot().timestamp.elapsed_seconds - impact_time
        hold_time = self._scenario._post_impact_hold + self._scenario._route_completion_hold
        if elapsed >= hold_time:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class _IntersectionTrafficCleaner(py_trees.behaviour.Behaviour):
    """Remove unrelated traffic from the intersection conflict box."""

    def __init__(self, scenario, name="IntersectionTrafficCleaner"):
        super().__init__(name)
        self._scenario = scenario
        self._last_clean = -999.0
        self._removed = 0

    def update(self):
        if os.environ.get("CUSTOM_DISABLE_BACKGROUND_NPCS", "").lower() in ("1", "true", "yes", "on"):
            return py_trees.common.Status.RUNNING

        now = GameTime.get_time()
        if now - self._last_clean < self._scenario._traffic_clean_interval:
            return py_trees.common.Status.RUNNING
        self._last_clean = now

        world = CarlaDataProvider.get_world()
        if world is None:
            return py_trees.common.Status.RUNNING

        protected_ids = {self._scenario._ego.id}
        if self._scenario._motorcycle is not None:
            protected_ids.add(self._scenario._motorcycle.id)
        if self._scenario._display_motorcycle is not None:
            protected_ids.add(self._scenario._display_motorcycle.id)

        center = self._scenario._intersection_center
        half_x = self._scenario._traffic_clean_half_x
        half_y = self._scenario._traffic_clean_half_y
        to_destroy = []
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.id in protected_ids:
                continue
            loc = actor.get_location()
            if (center.x - half_x <= loc.x <= center.x + half_x and
                    center.y - half_y <= loc.y <= center.y + half_y):
                to_destroy.append(actor)

        for actor in to_destroy:
            try:
                actor_type = actor.type_id
                loc = actor.get_location()
                actor.destroy()
                self._removed += 1
                print(
                    f"[MotoHang] Removed ambient vehicle from intersection: "
                    f"{actor_type} at ({loc.x:.1f}, {loc.y:.1f})",
                    flush=True,
                )
            except RuntimeError:
                pass

        return py_trees.common.Status.RUNNING


class OncomingMotorcycleDuringLeftTurnTown03(BasicScenario):
    """
    A slow-turning ego Tesla is struck by a fast motorcycle arriving from
    the opposite direction. Collision occurs at the front/front-right during
    the turn, then the motorcycle is launched toward the overhead signal bar.
    """

    timeout = 120

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

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=120):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._ego = ego_vehicles[0]
        op = dict(config.other_parameters) if hasattr(config, "other_parameters") else {}
        if hasattr(config, "scenario") and hasattr(config.scenario, "other_parameters"):
            op.update(config.scenario.other_parameters)
        self._activation_distance = self._get_param(
            op, "activation_distance_m", 70.0)

        # Bench2Drive is dropping other_parameters here, so these defaults
        # are tuned to the route geometry directly.
        int_x = self._get_param(op, "intersection_center_x", 6.5)
        int_y = self._get_param(op, "intersection_center_y", -130.0)
        int_z = self._get_param(op, "intersection_center_z", 0.0)
        self._intersection_center = carla.Location(x=int_x, y=int_y, z=int_z)
        self._impact_point = carla.Location(x=int_x + 2.2, y=int_y + 0.6, z=int_z)

        self._moto_speed = self._get_param(
            op, "motorcycle_speed_kmh", 110.0) / 3.6
        self._trigger_distance = self._get_param(
            op, "trigger_distance_m", 10.0)
        self._distance_impact_threshold = self._get_param(
            op, "distance_impact_threshold_m", 5.4)
        self._turn_commit_yaw_deg = self._get_param(
            op, "turn_commit_yaw_deg", 1.0)
        self._launch_z_mps = self._get_param(
            op, "launch_z_mps", 16.0)

        self._forced_turn_start_distance = self._get_param(
            op, "forced_turn_start_distance_m", 6.5)
        self._straight_hold_start_distance = self._get_param(
            op, "straight_hold_start_distance_m", 13.0)
        self._straight_hold_throttle = self._get_param(
            op, "straight_hold_throttle", 0.32)
        self._forced_turn_steer = self._get_param(
            op, "forced_turn_steer", -0.85)
        self._straight_attack_without_turn = bool(self._get_param(
            op, "straight_attack_without_turn", 0, cast=int))
        self._forced_turn_throttle = self._get_param(
            op, "forced_turn_throttle", 0.36)
        self._forced_turn_duration = self._get_param(
            op, "forced_turn_duration_s", 4.0)
        self._forced_turn_end_yaw_deg = self._get_param(
            op, "forced_turn_end_yaw_deg", 65.0)

        self._traffic_clean_half_x = self._get_param(
            op, "traffic_clean_half_x", 35.0)
        self._traffic_clean_half_y = self._get_param(
            op, "traffic_clean_half_y", 24.0)
        self._traffic_clean_interval = self._get_param(
            op, "traffic_clean_interval", 0.25)

        self._ego_speed_cap_kmh = self._get_param(
            op, "ego_speed_cap_kmh", 14.0)
        self._ego_min_speed_kmh = self._get_param(
            op, "ego_min_speed_kmh", 9.0)
        self._ego_cap_brake = self._get_param(
            op, "ego_cap_brake", 0.08)
        self._road_friction_scale = self._get_param(
            op, "road_friction_scale", 1.0)
        self._side_wind_force = self._get_param(
            op, "side_wind_force_n", 0.0)
        self._side_wind_yaw_deg = self._get_param(
            op, "side_wind_yaw_deg", 0.0)

        # Timing
        self._max_run_time = self._get_param(op, "max_run_time", 30.0)
        self._post_impact_hold = self._get_param(
            op, "post_impact_hold", 6.0)
        self._route_completion_hold = self._get_param(
            op, "route_completion_hold_s", 0.0)
        self._post_impact_capture_delay = self._get_param(
            op, "post_impact_capture_delay_s", 0.0)
        self._post_impact_flight_duration = self._get_param(
            op, "post_impact_flight_duration_s", 0.65)
        self._post_impact_arc_height = self._get_param(
            op, "post_impact_arc_height_m", 1.1)
        self._post_impact_bump_distance = self._get_param(
            op, "post_impact_bump_distance_m", 9.0)

        self._trigger_wp = self._map.get_waypoint(
            config.trigger_points[0].location)
        self._motorcycle = None
        self._display_motorcycle = None
        self._hang_display_motorcycle = None
        self._motorcycle_model = "vehicle.kawasaki.ninja"
        self._collision_sensor = None
        self._tl_dict = {}
        self._impact_detected = False
        self._impact_time = None
        self._post_impact_launch_transform = None
        self._initial_ego_yaw = None
        self._hang_transform = carla.Transform(
            carla.Location(
                x=self._get_param(op, "hang_target_x", self._intersection_center.x + 0.5),
                y=self._get_param(op, "hang_target_y", self._intersection_center.y - 2.0),
                z=self._get_param(op, "hang_target_z", 5.8),
            ),
            carla.Rotation(
                pitch=self._get_param(op, "hang_target_pitch", 12.0),
                yaw=self._get_param(op, "hang_target_yaw", 90.0),
                roll=self._get_param(op, "hang_target_roll", 85.0),
            ),
        )

        super().__init__(
            name="OncomingMotorcycleDuringLeftTurnTown03",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            terminate_on_failure=True,
            criteria_enable=criteria_enable,
        )

    def _initialize_environment(self, world):
        super()._initialize_environment(world)
        friction = max(0.1, float(self._road_friction_scale))
        if abs(friction - 1.0) < 1e-6:
            return
        bp = world.get_blueprint_library().find("static.trigger.friction")
        bp.set_attribute("friction", str(friction))
        bp.set_attribute("extent_x", "1000000.0")
        bp.set_attribute("extent_y", "1000000.0")
        bp.set_attribute("extent_z", "1000000.0")
        try:
            world.spawn_actor(bp, carla.Transform(
                carla.Location(-10000.0, -10000.0, 0.0)))
        except RuntimeError:
            pass

    def _initialize_actors(self, config):
        self._initial_ego_yaw = self._ego.get_transform().rotation.yaw

        # Spawn motorcycle from config or computed position
        if config.other_actors and len(config.other_actors) > 0:
            actor_config = config.other_actors[0]
            moto_transform = carla.Transform(
                carla.Location(
                    x=actor_config.transform.location.x,
                    y=actor_config.transform.location.y,
                    z=actor_config.transform.location.z + 0.3,
                ),
                actor_config.transform.rotation,
            )
        else:
            # Default: south of intersection, facing toward it
            moto_loc = carla.Location(
                x=self._intersection_center.x + 10.0,
                y=self._intersection_center.y - 35.0,
                z=0.5,
            )
            yaw = math.degrees(math.atan2(
                self._intersection_center.y - moto_loc.y,
                self._intersection_center.x - moto_loc.x,
            ))
            moto_transform = carla.Transform(
                moto_loc, carla.Rotation(yaw=yaw))

        # Try motorcycle models
        moto_models = [
            "vehicle.kawasaki.ninja",
            "vehicle.yamaha.yzf",
            "vehicle.harley-davidson.low_rider",
        ]
        self._motorcycle = None
        for model in moto_models:
            self._motorcycle = CarlaDataProvider.request_new_actor(
                model, moto_transform, rolename="hanging_motorcycle",
            )
            if self._motorcycle is not None:
                break
        if self._motorcycle is None:
            raise RuntimeError("[MotoHang] Motorcycle spawn failed")

        self._motorcycle.set_simulate_physics(True)
        self._motorcycle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self._motorcycle_model = self._motorcycle.type_id
        self.other_actors.append(self._motorcycle)

        # Collision sensor on ego
        collision_bp = self._world.get_blueprint_library().find(
            "sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self._ego)
        self._collision_sensor.listen(lambda event: self._on_collision(event))

        ego_loc = self._ego.get_location()
        for traffic_light in self._world.get_actors().filter("traffic.traffic_light"):
            loc = traffic_light.get_location()
            if math.hypot(loc.x - ego_loc.x, loc.y - ego_loc.y) <= 80.0:
                self._tl_dict[traffic_light] = carla.TrafficLightState.Green

        print(
            f"\n[MotoHang] Spawn\n"
            f"  Motorcycle at ({moto_transform.location.x:.1f}, "
            f"{moto_transform.location.y:.1f}) "
            f"model={self._motorcycle.type_id}\n"
            f"  Intersection center=({self._intersection_center.x:.1f}, "
            f"{self._intersection_center.y:.1f})  impact=({self._impact_point.x:.1f}, "
            f"{self._impact_point.y:.1f})\n"
            f"  Moto speed={self._moto_speed * 3.6:.0f}km/h  "
            f"trigger={self._trigger_distance:.0f}m  turn_commit={self._turn_commit_yaw_deg:.0f}deg\n"
            f"  Hang target=({self._hang_transform.location.x:.1f}, "
            f"{self._hang_transform.location.y:.1f}, {self._hang_transform.location.z:.1f})\n"
            f"  friction={self._road_friction_scale:.2f}  "
            f"side_wind={self._side_wind_force:.0f}N@{self._side_wind_yaw_deg:.0f}deg\n",
            flush=True,
        )

    def _spawn_display_motorcycle(self, transform):
        if self._display_motorcycle is not None and self._display_motorcycle.is_alive:
            return self._display_motorcycle

        blueprint = self._world.get_blueprint_library().find(self._motorcycle_model)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hanging_motorcycle_display")

        spawn_attempts = [
            transform,
            carla.Transform(
                carla.Location(
                    x=transform.location.x,
                    y=transform.location.y,
                    z=transform.location.z + 0.8,
                ),
                transform.rotation,
            ),
        ]
        for spawn_tf in spawn_attempts:
            actor = self._world.try_spawn_actor(blueprint, spawn_tf)
            if actor is None:
                continue
            actor.set_simulate_physics(False)
            actor.set_target_velocity(carla.Vector3D(0, 0, 0))
            actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            actor.set_transform(spawn_tf)
            self._display_motorcycle = actor
            self.other_actors.append(actor)
            loc = spawn_tf.location
            print(
                f"[MotoHang] Spawned clean display motorcycle at "
                f"({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})",
                flush=True,
            )
            return actor

        print("[MotoHang] WARNING: display motorcycle spawn failed; using collision actor", flush=True)
        return None

    def _spawn_hang_display_motorcycle(self):
        if self._hang_display_motorcycle is not None and self._hang_display_motorcycle.is_alive:
            return self._hang_display_motorcycle

        blueprint = self._world.get_blueprint_library().find(self._motorcycle_model)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hanging_motorcycle_held_display")

        actor = self._world.try_spawn_actor(blueprint, self._hang_transform)
        if actor is None:
            print("[MotoHang] WARNING: held hang display spawn failed", flush=True)
            return None

        actor.set_simulate_physics(False)
        actor.set_target_velocity(carla.Vector3D(0, 0, 0))
        actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        actor.set_transform(self._hang_transform)
        self._hang_display_motorcycle = actor
        self.other_actors.append(actor)
        loc = self._hang_transform.location
        print(
            f"[MotoHang] Spawned held display motorcycle at "
            f"({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})",
            flush=True,
        )
        return actor

    def _get_post_impact_exit_transform(self, start_transform):
        start_loc = start_transform.location
        impact = self._impact_point
        away_x = start_loc.x - impact.x
        away_y = start_loc.y - impact.y
        mag = math.hypot(away_x, away_y)
        if mag < 0.1:
            yaw_rad = math.radians(start_transform.rotation.yaw)
            away_x = math.cos(yaw_rad)
            away_y = math.sin(yaw_rad)
            mag = 1.0
        away_x /= mag
        away_y /= mag
        loc = carla.Location(
            x=start_loc.x + away_x * self._post_impact_bump_distance,
            y=start_loc.y + away_y * self._post_impact_bump_distance,
            z=max(start_loc.z + 0.15, self._intersection_center.z + 0.8),
        )
        return carla.Transform(
            loc,
            carla.Rotation(
                pitch=-28.0,
                yaw=math.degrees(math.atan2(away_y, away_x)),
                roll=104.0,
            ),
        )

    def _on_collision(self, event):
        other = event.other_actor
        if self._motorcycle is not None and other.id != self._motorcycle.id:
            print(f"[MotoHang] Ignoring non-motorcycle collision: {other.type_id}", flush=True)
            return
        if not self._impact_detected:
            self._impact_detected = True
            self._impact_time = GameTime.get_time()
            print(
                f"[MotoHang] *** COLLISION: {event.other_actor.type_id} ***",
                flush=True,
            )

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateMotorcycleHanging",
        )

    def _create_behavior(self):
        root = py_trees.composites.Sequence("MotorcycleHangingSequence")

        # Phase 0: sync — hold motorcycle until ego starts moving
        sync = py_trees.composites.Parallel(
            "Phase0_Sync",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
        )
        sync.add_child(HoldUntilEgoMoves(self._motorcycle, self._ego))
        root.add_child(sync)

        # Phase 1: main loop — motorcycle attack + ego speed control
        main_loop = py_trees.composites.Parallel(
            "Phase1_MainLoop",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        main_loop.add_child(_RunMotorcycleAttack(self))
        main_loop.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            max_brake=self._ego_cap_brake,
        ))
        main_loop.add_child(_EgoMinSpeedForcer(
            self._ego, min_speed_kmh=self._ego_min_speed_kmh))
        main_loop.add_child(_EgoIntersectionLeftTurnForcer(self))
        main_loop.add_child(_IntersectionTrafficCleaner(self))
        main_loop.add_child(TimeOut(self._max_run_time))
        root.add_child(main_loop)

        # Phase 2: post-impact hold (capture motorcycle in the air)
        root.add_child(_PostImpactHold(
            self,
            self._post_impact_hold,
            freeze_delay=self._post_impact_capture_delay,
            flight_duration=self._post_impact_flight_duration,
            arc_height=self._post_impact_arc_height,
        ))
        if self._route_completion_hold > 0.0:
            root.add_child(TimeOut(self._route_completion_hold, name="RouteCompletionHold"))

        # Cleanup
        root.add_child(ActorDestroy(
            self._motorcycle, name="DestroyMotorcycle"))

        ego_wp = self._map.get_waypoint(
            self._trigger_wp.transform.location,
            project_to_road=True, lane_type=carla.LaneType.Driving,
        )
        lane_wps = [ego_wp]
        left = ego_wp.get_left_lane()
        if left and left.lane_type == carla.LaneType.Driving:
            lane_wps.append(left)
        right = ego_wp.get_right_lane()
        if right and right.lane_type == carla.LaneType.Driving:
            lane_wps.append(right)

        outer = py_trees.composites.Parallel(
            "MotorcycleHanging_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        if self.route_mode:
            outer.add_child(_WaitForMotorcyclePostImpact(self))
        outer.add_child(CarlaRolloutLogger(
            actors=[("ego", self._ego), ("motorcycle", self._motorcycle)],
            scenario_id="moto_hang",
            route_id="motorcycle_hanging_route",
            name="CarlaRolloutLogger"))
        outer.add_child(_SideWindForce(self))
        if self._tl_dict:
            outer.add_child(TrafficLightFreezer(self._tl_dict))
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego, self._motorcycle],
            lane_waypoints=lane_wps,
            center_location=self._trigger_wp.transform.location,
            radius=180.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        return outer

    def _create_test_criteria(self):
        criteria = py_trees.composites.Parallel(
            "MotorcycleHangingCriteria",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        criteria.add_child(CollisionTest(self._ego))
        return criteria

    def remove_all_actors(self):
        if self._collision_sensor is not None and self._collision_sensor.is_alive:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()
            self._collision_sensor = None
        if self._display_motorcycle is not None and self._display_motorcycle.is_alive:
            try:
                self._display_motorcycle.destroy()
            except RuntimeError:
                pass
            self._display_motorcycle = None
        if self._hang_display_motorcycle is not None and self._hang_display_motorcycle.is_alive:
            try:
                self._hang_display_motorcycle.destroy()
            except RuntimeError:
                pass
            self._hang_display_motorcycle = None
        super().remove_all_actors()
