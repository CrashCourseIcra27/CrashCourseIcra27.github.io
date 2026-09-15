"""Urban overtake cut in and right turn (Town13).

A car overtakes on the left, cuts in ahead of the ego, and immediately turns right.
"""

import math

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
)
from srunner.scenariomanager.scenarioatomics.custom_atomics_urban_overtake_cut_in_and_right_turn_town13 import (
    CarlaRolloutLogger,
    EgoSpeedGovernor,
    HoldUntilEgoMoves,
    LaneCleaner,
    PeriodicStatusLogger,
    StopActorsOnEgoCollision,
)
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenarios.basic_scenario import BasicScenario


def _speed_mps(actor):
    if actor is None or not actor.is_alive:
        return 0.0
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _planar_distance(loc_a, loc_b):
    return math.hypot(loc_a.x - loc_b.x, loc_a.y - loc_b.y)


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    """Keep ego committed enough that the adjacent cut-in remains visible."""

    def __init__(self, ego, min_speed_kmh=18.0, throttle=0.65, name="EgoMinSpeedForcer"):
        super().__init__(name)
        self._ego = ego
        self._min_mps = float(min_speed_kmh) / 3.6
        self._throttle = float(throttle)

    def update(self):
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.RUNNING
        speed = _speed_mps(self._ego)
        if 3.0 < speed < self._min_mps:
            control = self._ego.get_control()
            control.brake = 0.0
            control.throttle = max(control.throttle, self._throttle)
            self._ego.apply_control(control)
        return py_trees.common.Status.RUNNING


class _UrbanCutInController(py_trees.behaviour.Behaviour):
    """Overtake from the left lane, then cut right and turn away."""

    def __init__(self, scenario, name="UrbanCutInController"):
        super().__init__(name)
        self._scenario = scenario
        self._last_time = None
        self._cut_start_time = None

    def initialise(self):
        self._last_time = GameTime.get_time()
        self._cut_start_time = None

    def update(self):
        scenario = self._scenario
        actor = scenario._primary_actor
        if actor is None or not actor.is_alive:
            return py_trees.common.Status.SUCCESS

        now = GameTime.get_time()
        dt = max(0.02, min(0.10, now - self._last_time)) if self._last_time else 0.05
        self._last_time = now

        ego_loc = scenario._ego.get_location()
        dist_to_trigger = _planar_distance(ego_loc, scenario._trigger_wp.transform.location)
        if self._cut_start_time is None and dist_to_trigger <= scenario._cut_start_distance:
            self._cut_start_time = now
            scenario._start_primary_cut_in()
            print(
                f"[UrbanCutIn] Ego entering crosswalk zone "
                f"(dist={dist_to_trigger:.1f}m) - left-lane car cuts right",
                flush=True,
            )

        if self._cut_start_time is None:
            scenario._pace_primary_with_ego(dist_to_trigger)
            return py_trees.common.Status.RUNNING

        elapsed = now - self._cut_start_time
        scenario._advance_primary_cut_in(dt, elapsed)

        total_duration = scenario._cut_duration + scenario._post_cut_duration
        if elapsed >= total_duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class UrbanOvertakeCutInAndRightTurnTown13(BasicScenario):
    """Left-lane traffic overtakes ego, cuts across, then immediately turns right."""

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

    @staticmethod
    def _walk_waypoint(waypoint, distance, forward=True):
        remaining = abs(float(distance))
        cursor = waypoint
        while remaining > 0.2 and cursor is not None:
            step = min(6.0, remaining)
            candidates = cursor.next(step) if forward else cursor.previous(step)
            if not candidates:
                break
            cursor = candidates[0]
            remaining -= step
        return cursor

    @staticmethod
    def _vector_add(location, vector, scale):
        return carla.Location(
            x=location.x + vector.x * scale,
            y=location.y + vector.y * scale,
            z=location.z + vector.z * scale,
        )

    @staticmethod
    def _yaw_from_forward(vector):
        return math.degrees(math.atan2(vector.y, vector.x))

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=120):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._ego = ego_vehicles[0]

        op = dict(config.other_parameters) if hasattr(config, "other_parameters") else {}
        if hasattr(config, "scenario") and hasattr(config.scenario, "other_parameters"):
            op.update(config.scenario.other_parameters)
        if isinstance(op, dict) and isinstance(op.get("other_parameters"), dict):
            op = op["other_parameters"]
        self._activation_distance = self._get_param(op, "activation_distance_m", 62.0)
        self._primary_model = str(self._get_param(
            op, "primary_model", "vehicle.dodge.charger_2020", cast=str))
        self._primary_side_sign = self._get_param(op, "primary_side_sign", 1.0)
        self._primary_spawn_ahead = self._get_param(op, "primary_spawn_ahead_m", -22.0)
        self._primary_cruise_ahead = self._get_param(op, "primary_cruise_ahead_m", 15.0)
        self._primary_lateral_offset = self._get_param(op, "primary_lateral_offset_m", 3.4)
        self._primary_speed_kmh = self._get_param(op, "primary_speed_kmh", 26.0)
        self._cut_start_distance = self._get_param(op, "cut_start_distance_m", 12.0)
        self._cut_forward_speed_kmh = self._get_param(op, "cut_forward_speed_kmh", 18.0)
        self._cut_lateral_speed = self._get_param(op, "cut_lateral_speed_ms", 3.1)
        self._cut_intrusion_offset = self._get_param(op, "cut_intrusion_offset_m", 0.7)
        self._cut_duration = self._get_param(op, "cut_duration_s", 1.35)
        self._cut_yaw_deg = self._get_param(op, "cut_yaw_deg", 16.0)
        self._right_turn_forward_speed_kmh = self._get_param(op, "right_turn_forward_speed_kmh", 17.0)
        self._right_turn_lateral_offset = self._get_param(op, "right_turn_lateral_offset_m", 4.4)
        self._right_turn_duration = self._get_param(op, "right_turn_duration_s", 2.2)
        self._right_turn_yaw_deg = self._get_param(op, "right_turn_yaw_deg", 72.0)
        self._post_cut_duration = self._get_param(op, "post_cut_duration_s", 5.0)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 31.0)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 19.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.08)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._primary_actor = None
        self._primary_lane_wp = None
        self._primary_loc = None
        self._primary_forward = None
        self._primary_right = None
        self._primary_base_yaw = None
        self._cut_origin_loc = None
        self._cut_origin_forward = None
        self._cut_origin_right = None
        self._cut_origin_yaw = None
        self._toward_ego_sign = 1.0
        self._away_from_ego_sign = -1.0

        super().__init__(
            name="UrbanOvertakeCutInAndRightTurnTown13",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            terminate_on_failure=True,
            criteria_enable=criteria_enable,
        )

    def _spawn_actor(self, models, transform, rolename):
        for model in dict.fromkeys(models):
            actor = CarlaDataProvider.request_new_actor(model, transform, rolename=rolename)
            if actor is not None:
                return actor
        return None

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        self._primary_lane_wp = ego_wp
        if abs(self._primary_spawn_ahead) > 0.2:
            spawn_wp = self._walk_waypoint(
                ego_wp, self._primary_spawn_ahead, forward=self._primary_spawn_ahead >= 0.0)
            if spawn_wp is None:
                spawn_wp = ego_wp
        else:
            spawn_wp = ego_wp
        spawn_options = []
        for side_sign in (self._primary_side_sign, -self._primary_side_sign, 0.0):
            for offset in (self._primary_lateral_offset, max(1.2, self._primary_lateral_offset * 0.55)):
                spawn_options.append((spawn_wp, side_sign, offset))
        for ahead_delta in (-8.0, 8.0, -16.0):
            alt_wp = self._walk_waypoint(
                spawn_wp, ahead_delta, forward=ahead_delta >= 0.0)
            if alt_wp is not None:
                spawn_options.append((alt_wp, self._primary_side_sign, self._primary_lateral_offset))
                spawn_options.append((alt_wp, -self._primary_side_sign, self._primary_lateral_offset))
                spawn_options.append((alt_wp, self._primary_side_sign, max(1.2, self._primary_lateral_offset * 0.55)))
                spawn_options.append((alt_wp, -self._primary_side_sign, max(1.2, self._primary_lateral_offset * 0.55)))
        spawn_options.append((spawn_wp, 0.0, 0.0))

        self._primary_actor = None
        selected_spawn = (spawn_wp, self._primary_side_sign, self._primary_lateral_offset)
        for candidate_wp, side_sign, offset in spawn_options:
            candidate_forward = candidate_wp.transform.get_forward_vector()
            candidate_right = candidate_wp.transform.get_right_vector()
            primary_location = self._vector_add(
                candidate_wp.transform.location, candidate_right, offset * side_sign)
            primary_transform = carla.Transform(
                carla.Location(
                    primary_location.x,
                    primary_location.y,
                    candidate_wp.transform.location.z + 0.5,
                ),
                candidate_wp.transform.rotation,
            )
            self._primary_actor = self._spawn_actor(
                [
                    self._primary_model,
                    "vehicle.audi.tt",
                    "vehicle.tesla.model3",
                    "vehicle.dodge.charger_2020",
                ],
                primary_transform,
                "urban_crosswalk_cut_in_vehicle",
            )
            if self._primary_actor is not None:
                self._primary_side_sign = side_sign if abs(side_sign) > 0.01 else self._primary_side_sign
                self._primary_lateral_offset = offset
                selected_spawn = (candidate_wp, side_sign, offset)
                break
        if self._primary_actor is None:
            raise RuntimeError("[UrbanCutIn] Primary cut-in vehicle spawn failed")
        self._primary_actor.set_simulate_physics(True)
        self.other_actors.append(self._primary_actor)

        self._away_from_ego_sign = 1.0 if self._primary_side_sign >= 0.0 else -1.0
        self._toward_ego_sign = -self._away_from_ego_sign

        self._capture_primary_pose()

        print(
            f"\n[UrbanCutIn] Spawn\n"
            f"  Ego road={ego_wp.road_id} lane={ego_wp.lane_id} "
            f"spawn_road={selected_spawn[0].road_id} spawn_lane={selected_spawn[0].lane_id}\n"
            f"  Primary virtual left-lane side={self._primary_side_sign:+.1f} "
            f"initial_longitudinal={self._primary_spawn_ahead:.1f}m "
            f"overtake_target={self._primary_cruise_ahead:.1f}m "
            f"offset={self._primary_lateral_offset:.1f}m\n"
            f"  toward_ego_sign={self._toward_ego_sign:+.1f} "
            f"cut_start_distance={self._cut_start_distance:.1f}m "
            f"cut_lat={self._cut_lateral_speed:.1f}m/s\n",
            flush=True,
        )

    def _capture_primary_pose(self):
        transform = self._primary_actor.get_transform()
        self._primary_loc = transform.location
        self._primary_forward = transform.get_forward_vector()
        self._primary_right = transform.get_right_vector()
        self._primary_base_yaw = self._yaw_from_forward(self._primary_forward)

    def _start_primary_cut_in(self):
        self._capture_primary_pose()
        self._cut_origin_loc = self._primary_loc
        self._cut_origin_forward = self._primary_forward
        self._cut_origin_right = self._primary_right
        self._cut_origin_yaw = self._primary_base_yaw

    def _set_actor_pose(self, actor, location, base_yaw, yaw_offset_deg,
                        forward_speed, lateral_speed, forward_vector, right_vector):
        actor.set_transform(carla.Transform(location, carla.Rotation(yaw=base_yaw + yaw_offset_deg)))
        vx = forward_vector.x * forward_speed + right_vector.x * lateral_speed
        vy = forward_vector.y * forward_speed + right_vector.y * lateral_speed
        actor.set_target_velocity(carla.Vector3D(vx, vy, 0.0))

    def _set_actor_heading_pose(self, actor, location, yaw_deg, speed):
        actor.set_transform(carla.Transform(location, carla.Rotation(yaw=yaw_deg)))
        yaw_rad = math.radians(yaw_deg)
        actor.set_target_velocity(carla.Vector3D(
            math.cos(yaw_rad) * speed,
            math.sin(yaw_rad) * speed,
            0.0,
        ))

    def _pace_primary_with_ego(self, dist_to_trigger=None):
        ego_transform = self._ego.get_transform()
        ego_forward = ego_transform.get_forward_vector()
        ego_right = ego_transform.get_right_vector()
        if dist_to_trigger is None:
            overtake_progress = 1.0
        else:
            start_distance = max(self._activation_distance, self._cut_start_distance + 1.0)
            overtake_span = max(1.0, start_distance - self._cut_start_distance)
            overtake_progress = max(
                0.0,
                min(1.0, (start_distance - dist_to_trigger) / overtake_span),
            )
        overtake_ease = overtake_progress * overtake_progress * (3.0 - 2.0 * overtake_progress)
        longitudinal = (
            self._primary_spawn_ahead * (1.0 - overtake_ease)
            + self._primary_cruise_ahead * overtake_ease
        )
        speed = max(_speed_mps(self._ego), self._primary_speed_kmh / 3.6)
        lateral = self._away_from_ego_sign * self._primary_lateral_offset
        target = carla.Location(
            x=ego_transform.location.x + ego_forward.x * longitudinal + ego_right.x * lateral,
            y=ego_transform.location.y + ego_forward.y * longitudinal + ego_right.y * lateral,
            z=ego_transform.location.z + 0.15,
        )
        forward = ego_forward
        right = ego_right
        base_yaw = ego_transform.rotation.yaw
        self._primary_loc = target
        self._primary_forward = forward
        self._primary_right = right
        self._primary_base_yaw = base_yaw
        self._set_actor_pose(self._primary_actor, target, base_yaw, 0.0, speed, 0.0, forward, right)

    def _advance_primary_cut_in(self, dt, elapsed):
        forward = self._cut_origin_forward or self._primary_forward
        right = self._cut_origin_right or self._primary_right
        base_yaw = self._cut_origin_yaw if self._cut_origin_yaw is not None else self._primary_base_yaw
        origin = self._cut_origin_loc or self._primary_loc
        forward_speed = self._cut_forward_speed_kmh / 3.6
        cut_lateral_travel = self._toward_ego_sign * (
            self._primary_lateral_offset + self._cut_intrusion_offset)

        if elapsed <= self._cut_duration:
            ratio = max(0.0, min(1.0, elapsed / max(0.01, self._cut_duration)))
            ease = ratio * ratio * (3.0 - 2.0 * ratio)
            lateral_speed = self._toward_ego_sign * self._cut_lateral_speed
            yaw_offset = self._toward_ego_sign * self._cut_yaw_deg * math.sin(math.pi * ratio)
            forward_offset = forward_speed * elapsed
            lateral_offset = cut_lateral_travel * ease
            self._primary_loc = carla.Location(
                x=origin.x + forward.x * forward_offset + right.x * lateral_offset,
                y=origin.y + forward.y * forward_offset + right.y * lateral_offset,
                z=origin.z,
            )
            self._set_actor_pose(
                self._primary_actor,
                self._primary_loc,
                base_yaw,
                yaw_offset,
                forward_speed,
                lateral_speed,
                forward,
                right,
            )
        else:
            turn_elapsed = elapsed - self._cut_duration
            turn_ratio = max(0.0, min(1.0, turn_elapsed / max(0.01, self._right_turn_duration)))
            ease = turn_ratio * turn_ratio * (3.0 - 2.0 * turn_ratio)
            turn_speed = self._right_turn_forward_speed_kmh / 3.6
            turn_angle = math.radians(self._right_turn_yaw_deg) * self._toward_ego_sign * ease
            final_angle = max(0.05, abs(math.radians(self._right_turn_yaw_deg)))
            radius = max(
                4.0,
                abs(self._right_turn_lateral_offset) / max(0.05, 1.0 - math.cos(final_angle)),
            )
            turn_start_forward = forward_speed * self._cut_duration
            turn_start = carla.Location(
                x=origin.x + forward.x * turn_start_forward + right.x * cut_lateral_travel,
                y=origin.y + forward.y * turn_start_forward + right.y * cut_lateral_travel,
                z=origin.z,
            )
            forward_offset = radius * math.sin(abs(turn_angle))
            lateral_offset = self._toward_ego_sign * radius * (1.0 - math.cos(abs(turn_angle)))
            self._primary_loc = carla.Location(
                x=turn_start.x + forward.x * forward_offset + right.x * lateral_offset,
                y=turn_start.y + forward.y * forward_offset + right.y * lateral_offset,
                z=turn_start.z,
            )
            yaw = base_yaw + math.degrees(turn_angle)
            self._set_actor_heading_pose(self._primary_actor, self._primary_loc, yaw, turn_speed)
        self._primary_actor.apply_control(carla.VehicleControl(throttle=0.20, brake=0.0, steer=0.26 * self._toward_ego_sign))

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateUrbanRightLaneCutInAtCrosswalk",
        )

    def _create_behavior(self):
        P = py_trees.common.ParallelPolicy
        root = py_trees.composites.Sequence("UrbanRightLaneCutInAtCrosswalk_Root")

        phase0 = py_trees.composites.Parallel("Phase0_Sync", policy=P.SUCCESS_ON_ONE)
        phase0.add_child(TimeOut(8.0, name="Phase0_Timeout"))
        phase0.add_child(HoldUntilEgoMoves(self._primary_actor, self._ego, name="HoldPrimaryUntilEgoMoves"))
        root.add_child(phase0)

        root.add_child(_UrbanCutInController(self))

        cleanup = py_trees.composites.Sequence("Cleanup")
        cleanup.add_child(ActorDestroy(self._primary_actor, name="DestroyUrbanCutInVehicle"))
        root.add_child(cleanup)

        lane_waypoints = [self._trigger_wp]
        if self._primary_lane_wp is not None:
            lane_waypoints.append(self._primary_lane_wp)
        protected_actors = [self._ego, self._primary_actor]

        outer = py_trees.composites.Parallel(
            "UrbanRightLaneCutInAtCrosswalk_Outer",
            policy=P.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        outer.add_child(CarlaRolloutLogger(
            actors=[("ego", self._ego), ("primary_cut_in", self._primary_actor)],
            scenario_id="urban_right_lane_cut_in_at_crosswalk",
            route_id="urban_right_lane_cut_in_at_crosswalk_route",
            name="CarlaRolloutLogger",
        ))
        outer.add_child(LaneCleaner(
            protected_actors=protected_actors,
            lane_waypoints=lane_waypoints,
            center_location=self._trigger_wp.transform.location,
            radius=180.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        outer.add_child(PeriodicStatusLogger(
            self._ego, self._primary_actor, None, interval=1.0, name="StatusLogger"))
        outer.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            max_brake=self._ego_cap_brake,
            name="EgoSpeedGovernor",
        ))
        outer.add_child(_EgoMinSpeedForcer(
            self._ego,
            min_speed_kmh=self._ego_min_speed_kmh,
            name="EgoMinSpeedForcer",
        ))
        outer.add_child(StopActorsOnEgoCollision(
            self._ego,
            actors_to_stop=[self._primary_actor],
            name="StopOnCollision",
        ))
        return outer

    def _create_test_criteria(self):
        return [CollisionTest(self._ego)]

    def __del__(self):
        self.remove_all_actors()
