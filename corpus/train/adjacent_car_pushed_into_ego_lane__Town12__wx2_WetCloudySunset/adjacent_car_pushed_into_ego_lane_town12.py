"""Adjacent car pushed into ego lane (Town12).

A car in the adjacent lane is forced sideways into the ego's lane during a merge.
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
from srunner.scenariomanager.scenarioatomics.custom_atomics_adjacent_car_pushed_into_ego_lane_town12 import (
    EgoSpeedGovernor,
    LaneCleaner,
    PeriodicStatusLogger,
)
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenarios.basic_scenario import BasicScenario


def _speed_mps(actor):
    if actor is None or not actor.is_alive:
        return 0.0
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


class _RunForcedMergePitRebound(py_trees.behaviour.Behaviour):
    """Stage a right-side vehicle being hit and drifting into ego."""

    def __init__(self, scenario_ref, name="RunForcedMergePitRebound"):
        super().__init__(name)
        self._scenario = scenario_ref
        self._last_time = None
        self._action_t0 = None  # game-time when action fires (ego reached speed)

    def initialise(self):
        self._last_time = GameTime.get_time()
        self._action_t0 = None

    def update(self):
        scenario = self._scenario
        if scenario._pickup is None or scenario._merge_car is None:
            return py_trees.common.Status.FAILURE
        if not scenario._pickup.is_alive or not scenario._merge_car.is_alive:
            return py_trees.common.Status.SUCCESS

        now = GameTime.get_time()
        dt = max(0.02, min(0.10, now - self._last_time)) if self._last_time else 0.05
        self._last_time = now

        # --- Phase 0: wait for ego to reach freeway speed ---
        if self._action_t0 is None:
            ego_speed = _speed_mps(scenario._ego)
            if ego_speed >= scenario._trigger_speed_mps:
                self._action_t0 = now
                print(
                    f"[ForcedMerge] Ego at {ego_speed * 3.6:.1f} km/h "
                    f"— ACCIDENT SEQUENCE STARTS",
                    flush=True,
                )
            else:
                scenario._cruise_vehicles(dt)
                return py_trees.common.Status.RUNNING

        # --- Phase 1+: run the accident sequence ---
        act_elapsed = max(0.0, now - self._action_t0)
        scenario._advance_pickup(dt, act_elapsed)
        scenario._advance_merge_car(dt, act_elapsed)

        total = (
            scenario._pre_merge_duration
            + scenario._merge_duration
            + scenario._pit_duration
            + scenario._rebound_duration
            + scenario._settle_duration
        )
        if act_elapsed >= total:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    def __init__(self, ego, min_speed_kmh=58.0, throttle=0.9, name="EgoMinSpeedForcer"):
        super().__init__(name)
        self._ego = ego
        self._min_speed_mps = float(min_speed_kmh) / 3.6
        self._throttle = float(throttle)

    def update(self):
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.RUNNING
        if _speed_mps(self._ego) < self._min_speed_mps:
            control = self._ego.get_control()
            control.brake = 0.0
            control.throttle = max(control.throttle, self._throttle)
            self._ego.apply_control(control)
        return py_trees.common.Status.RUNNING


class AdjacentCarPushedIntoEgoLaneTown12(BasicScenario):
    """Vehicle beside ego is struck from its right and drifts into ego."""

    timeout = 180

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
            step = min(8.0, remaining)
            candidates = cursor.next(step) if forward else cursor.previous(step)
            if not candidates:
                break
            cursor = candidates[0]
            remaining -= step
        return cursor

    @staticmethod
    def _vector_add(location, vec, scale):
        return carla.Location(
            x=location.x + vec.x * scale,
            y=location.y + vec.y * scale,
            z=location.z + vec.z * scale,
        )

    @staticmethod
    def _yaw_from_forward(vec):
        return math.degrees(math.atan2(vec.y, vec.x))

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._ego = ego_vehicles[0]

        op = config.other_parameters if hasattr(config, "other_parameters") else {}
        self._activation_distance = self._get_param(op, "activation_distance_m", 95.0)
        self._pickup_spawn_ahead = self._get_param(op, "victim_spawn_ahead_m", 7.0)
        self._merge_spawn_ahead = self._get_param(op, "striker_spawn_ahead_m", 8.5)
        self._pickup_speed_kmh = self._get_param(op, "victim_speed_kmh", 48.0)
        self._merge_cruise_speed_kmh = self._get_param(op, "striker_speed_kmh", 78.0)
        self._merge_forward_speed_kmh = self._get_param(op, "striker_impact_speed_kmh", 82.0)
        self._pit_forward_speed_kmh = self._get_param(op, "victim_drift_speed_kmh", 34.0)
        self._rebound_forward_speed_kmh = self._get_param(op, "victim_post_impact_speed_kmh", 18.0)
        self._settle_forward_speed_kmh = self._get_param(op, "settle_forward_speed_kmh", 20.0)
        self._merge_lateral_speed_ms = self._get_param(op, "striker_lateral_speed_ms", 4.8)
        self._pit_lateral_speed_ms = self._get_param(op, "victim_lateral_speed_ms", 8.4)
        self._rebound_lateral_speed_ms = self._get_param(op, "victim_final_lateral_speed_ms", 3.8)
        self._settle_lateral_speed_ms = self._get_param(op, "settle_lateral_speed_ms", 0.3)
        self._pre_merge_duration = self._get_param(op, "parallel_duration_s", 0.25)
        self._merge_duration = self._get_param(op, "right_side_impact_duration_s", 0.45)
        self._pit_duration = self._get_param(op, "victim_drift_duration_s", 0.95)
        self._rebound_duration = self._get_param(op, "ego_lane_intrusion_duration_s", 0.95)
        self._settle_duration = self._get_param(op, "settle_duration_s", 1.1)
        self._merge_yaw_deg = self._get_param(op, "striker_yaw_deg", 10.0)
        self._pit_yaw_deg = self._get_param(op, "victim_yaw_deg", 24.0)
        self._rebound_yaw_deg = self._get_param(op, "victim_final_yaw_deg", 12.0)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 76.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.10)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 0.0)
        self._ego_force_throttle = self._get_param(op, "ego_force_throttle", 0.90)
        self._aftermath_duration = self._get_param(op, "aftermath_duration_s", 5.0)
        self._trigger_speed_mps = self._get_param(op, "trigger_speed_kmh", 58.0) / 3.6
        self._cruise_ahead_m = self._get_param(op, "cruise_ahead_m", 35.0)
        self._cruise_right_m = 3.6  # right-lane lateral offset from ego lane centre

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)

        self._pickup = None
        self._merge_car = None
        self._pickup_loc = None
        self._merge_loc = None
        self._pickup_base_yaw = None
        self._merge_base_yaw = None
        self._pickup_forward = None
        self._pickup_right = None
        self._merge_forward = None
        self._merge_right = None
        self._toward_ego_sign = -1.0
        self._toward_barrier_sign = 1.0
        self._striker_side_offset = self._get_param(op, "striker_side_offset_m", 3.7)
        self._merge_side_wp = None

        super().__init__(
            name="AdjacentCarPushedIntoEgoLaneTown12",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            terminate_on_failure=True,
            criteria_enable=criteria_enable,
        )

    def _cruise_vehicles(self, dt):
        """Keep victim and striker pacing alongside ego until the action fires."""
        self._advance_pickup(dt, 0.0)
        self._advance_merge_car(dt, 0.0)

    def _choose_merge_lane(self, ego_wp):
        right_lane = ego_wp.get_right_lane()
        if right_lane is not None and right_lane.lane_type == carla.LaneType.Driving:
            return right_lane
        left_lane = ego_wp.get_left_lane()
        if left_lane is not None and left_lane.lane_type == carla.LaneType.Driving:
            return left_lane
        return None

    def _spawn_actor(self, models, transform, rolename):
        for model in models:
            actor = CarlaDataProvider.request_new_actor(model, transform, rolename=rolename)
            if actor is not None:
                return actor
        return None

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        try:
            self._ego.set_light_state(carla.VehicleLightState(
                carla.VehicleLightState.Position |
                carla.VehicleLightState.LowBeam |
                carla.VehicleLightState.HighBeam
            ))
        except Exception:
            pass
        self._merge_side_wp = self._choose_merge_lane(ego_wp)
        if self._merge_side_wp is None:
            raise RuntimeError("[ForcedMerge] No adjacent lane available for right-side vehicle")
        pickup_wp = self._walk_waypoint(self._merge_side_wp, self._pickup_spawn_ahead, forward=True) or self._merge_side_wp
        merge_wp = self._walk_waypoint(self._merge_side_wp, self._merge_spawn_ahead, forward=True) or self._merge_side_wp
        merge_right = merge_wp.transform.get_right_vector()

        pickup_t = carla.Transform(
            carla.Location(
                x=pickup_wp.transform.location.x,
                y=pickup_wp.transform.location.y,
                z=pickup_wp.transform.location.z + 0.5,
            ),
            pickup_wp.transform.rotation,
        )
        merge_t = carla.Transform(
            carla.Location(
                x=merge_wp.transform.location.x + merge_right.x * self._striker_side_offset,
                y=merge_wp.transform.location.y + merge_right.y * self._striker_side_offset,
                z=merge_wp.transform.location.z + 0.5,
            ),
            merge_wp.transform.rotation,
        )

        self._pickup = self._spawn_actor(
            [
                "vehicle.nissan.patrol",
                "vehicle.audi.etron",
                "vehicle.tesla.model3",
            ],
            pickup_t,
            "right_side_victim_vehicle",
        )
        if self._pickup is None:
            raise RuntimeError("[ForcedMerge] Right-side victim spawn failed")

        self._merge_car = self._spawn_actor(
            [
                "vehicle.audi.tt",
                "vehicle.dodge.charger_2020",
                "vehicle.tesla.model3",
            ],
            merge_t,
            "outer_striking_vehicle",
        )
        if self._merge_car is None:
            raise RuntimeError("[ForcedMerge] Outer striking vehicle spawn failed")

        self._pickup.set_simulate_physics(True)
        self._pickup.set_enable_gravity(True)
        self._merge_car.set_simulate_physics(True)
        self._merge_car.set_enable_gravity(True)

        self.other_actors.append(self._pickup)
        self.other_actors.append(self._merge_car)

        self._pickup_loc = pickup_t.location
        self._merge_loc = merge_t.location
        self._pickup_forward = pickup_wp.transform.get_forward_vector()
        self._pickup_right = pickup_wp.transform.get_right_vector()
        self._merge_forward = merge_wp.transform.get_forward_vector()
        self._merge_right = merge_wp.transform.get_right_vector()
        self._pickup_base_yaw = self._yaw_from_forward(self._pickup_forward)
        self._merge_base_yaw = self._yaw_from_forward(self._merge_forward)

        to_ego = carla.Vector3D(
            ego_wp.transform.location.x - merge_wp.transform.location.x,
            ego_wp.transform.location.y - merge_wp.transform.location.y,
            0.0,
        )
        lateral_dot = to_ego.x * self._merge_right.x + to_ego.y * self._merge_right.y
        self._toward_ego_sign = 1.0 if lateral_dot >= 0.0 else -1.0
        self._toward_barrier_sign = -self._toward_ego_sign

        print(
            f"\n[ForcedMerge] Spawn\n"
            f"  Ego road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  Victim road={pickup_wp.road_id} lane={pickup_wp.lane_id} ahead={self._pickup_spawn_ahead:.1f}m\n"
            f"  Striker road={merge_wp.road_id} lane={merge_wp.lane_id} ahead={self._merge_spawn_ahead:.1f}m offset={self._striker_side_offset:.1f}m\n"
            f"  toward_ego_sign={self._toward_ego_sign:+.1f} toward_barrier_sign={self._toward_barrier_sign:+.1f}\n",
            flush=True,
        )

    def _set_actor_pose(self, actor, location, base_yaw, yaw_offset_deg, forward_speed, lateral_speed, forward_vec, right_vec):
        rotation = carla.Rotation(yaw=base_yaw + yaw_offset_deg)
        actor.set_transform(carla.Transform(location, rotation))
        vx = forward_vec.x * forward_speed + right_vec.x * lateral_speed
        vy = forward_vec.y * forward_speed + right_vec.y * lateral_speed
        actor.set_target_velocity(carla.Vector3D(vx, vy, 0.0))

    def _advance_pickup(self, dt, act_elapsed):
        impact_start = self._pre_merge_duration + self._merge_duration * 0.6
        drift_end = impact_start + self._pit_duration
        intrusion_end = drift_end + self._rebound_duration

        if act_elapsed < impact_start:
            ego_t = self._ego.get_transform()
            ego_fwd = ego_t.get_forward_vector()
            ego_right = ego_t.get_right_vector()
            pace = max(_speed_mps(self._ego), 8.0)
            lat_dir = -self._toward_ego_sign
            cx = ego_t.location.x + ego_fwd.x * self._cruise_ahead_m + ego_right.x * (lat_dir * self._cruise_right_m)
            cy = ego_t.location.y + ego_fwd.y * self._cruise_ahead_m + ego_right.y * (lat_dir * self._cruise_right_m)
            snap_wp = self._map.get_waypoint(carla.Location(x=cx, y=cy, z=ego_t.location.z),
                                             project_to_road=True, lane_type=carla.LaneType.Driving)
            if snap_wp is not None:
                cz = snap_wp.transform.location.z + 0.1
                fwd = snap_wp.transform.get_forward_vector()
                rgt = snap_wp.transform.get_right_vector()
                base_yaw = snap_wp.transform.rotation.yaw
            else:
                cz = ego_t.location.z + 0.1
                fwd = ego_fwd
                rgt = ego_right
                base_yaw = ego_t.rotation.yaw
            loc = carla.Location(x=cx, y=cy, z=cz)
            self._pickup_loc = loc
            self._pickup_forward = fwd
            self._pickup_right = rgt
            self._pickup_base_yaw = base_yaw
            self._set_actor_pose(self._pickup, loc, base_yaw, 0.0, pace, 0.0, fwd, rgt)
            return

        # Absolute position tracking from impact onwards.
        if act_elapsed < drift_end:
            ratio = (act_elapsed - impact_start) / max(0.01, self._pit_duration)
            fwd_speed = self._pit_forward_speed_kmh / 3.6
            lat_speed = self._toward_ego_sign * self._pit_lateral_speed_ms
            yaw_offset = self._toward_ego_sign * self._pit_yaw_deg * ratio
        elif act_elapsed < intrusion_end:
            ratio = (act_elapsed - drift_end) / max(0.01, self._rebound_duration)
            fwd_speed = self._rebound_forward_speed_kmh / 3.6
            lat_speed = self._toward_ego_sign * self._rebound_lateral_speed_ms
            yaw_offset = self._toward_ego_sign * (
                self._pit_yaw_deg * (1.0 - ratio) + self._rebound_yaw_deg * ratio
            )
        else:
            fwd_speed = self._settle_forward_speed_kmh / 3.6
            lat_speed = self._toward_ego_sign * self._settle_lateral_speed_ms
            yaw_offset = self._toward_ego_sign * self._rebound_yaw_deg * 0.5

        self._pickup_loc = self._vector_add(self._pickup_loc, self._pickup_forward, fwd_speed * dt)
        self._pickup_loc = self._vector_add(self._pickup_loc, self._pickup_right, lat_speed * dt)
        self._set_actor_pose(
            self._pickup, self._pickup_loc, self._pickup_base_yaw, yaw_offset,
            fwd_speed, lat_speed, self._pickup_forward, self._pickup_right,
        )

    def _advance_merge_car(self, dt, act_elapsed):
        stage1_end = self._pre_merge_duration
        stage2_end = stage1_end + self._merge_duration

        if act_elapsed < stage1_end:
            # Cruise: keep striker in outer adjacent lane (same vector approach as victim).
            ego_t = self._ego.get_transform()
            ego_fwd = ego_t.get_forward_vector()
            ego_right = ego_t.get_right_vector()
            pace = max(_speed_mps(self._ego) + 1.0, 8.0)
            lat_dir = -self._toward_ego_sign
            striker_right_m = lat_dir * (self._cruise_right_m + self._striker_side_offset)
            cx = ego_t.location.x + ego_fwd.x * self._cruise_ahead_m + ego_right.x * striker_right_m
            cy = ego_t.location.y + ego_fwd.y * self._cruise_ahead_m + ego_right.y * striker_right_m
            snap_wp = self._map.get_waypoint(carla.Location(x=cx, y=cy, z=ego_t.location.z),
                                             project_to_road=True, lane_type=carla.LaneType.Driving)
            if snap_wp is not None:
                cz = snap_wp.transform.location.z + 0.1
                fwd = snap_wp.transform.get_forward_vector()
                rgt = snap_wp.transform.get_right_vector()
                base_yaw = snap_wp.transform.rotation.yaw
            else:
                cz = ego_t.location.z + 0.1
                fwd = ego_fwd
                rgt = ego_right
                base_yaw = ego_t.rotation.yaw
            loc = carla.Location(x=cx, y=cy, z=cz)
            self._merge_loc = loc
            self._merge_forward = fwd
            self._merge_right = rgt
            self._merge_base_yaw = base_yaw
            self._set_actor_pose(self._merge_car, loc, base_yaw, 0.0, pace, 0.0, fwd, rgt)
            return

        # Absolute tracking: striker swerves into victim.
        forward_speed = 0.0
        lateral_speed = 0.0
        yaw_offset = 0.0
        if act_elapsed < stage2_end:
            ratio = (act_elapsed - stage1_end) / max(0.01, self._merge_duration)
            forward_speed = self._merge_forward_speed_kmh / 3.6
            lateral_speed = -self._toward_barrier_sign * self._merge_lateral_speed_ms * (0.35 + 0.65 * ratio)
            yaw_offset = -self._toward_barrier_sign * self._merge_yaw_deg * ratio
        elif act_elapsed < stage2_end + self._pit_duration:
            forward_speed = self._pit_forward_speed_kmh / 3.6
            lateral_speed = -self._toward_barrier_sign * 0.8
            yaw_offset = -self._toward_barrier_sign * self._merge_yaw_deg
        else:
            forward_speed = self._settle_forward_speed_kmh / 3.6
            lateral_speed = 0.0
            yaw_offset = -self._toward_barrier_sign * self._merge_yaw_deg * 0.5

        self._merge_loc = self._vector_add(self._merge_loc, self._merge_forward, forward_speed * dt)
        self._merge_loc = self._vector_add(self._merge_loc, self._merge_right, lateral_speed * dt)
        self._set_actor_pose(
            self._merge_car, self._merge_loc, self._merge_base_yaw, yaw_offset,
            forward_speed, lateral_speed, self._merge_forward, self._merge_right,
        )

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateForcedMergePitRebound",
        )

    def _create_behavior(self):
        root = py_trees.composites.Sequence("ForcedMergePitRebound_Root")
        root.add_child(_RunForcedMergePitRebound(self, name="MergePitRebound"))
        root.add_child(TimeOut(self._aftermath_duration, name="Aftermath"))

        cleanup = py_trees.composites.Sequence("Cleanup")
        cleanup.add_child(ActorDestroy(self._merge_car, name="DestroyOuterStriker"))
        cleanup.add_child(ActorDestroy(self._pickup, name="DestroyRightSideVictim"))
        root.add_child(cleanup)

        protected = [self._ego, self._pickup, self._merge_car]
        lanes = [self._trigger_wp]
        if self._merge_side_wp is not None:
            lanes.append(self._merge_side_wp)

        outer = py_trees.composites.Parallel(
            "ForcedMergePitRebound_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=protected,
            lane_waypoints=lanes,
            center_location=self._trigger_wp.transform.location,
            radius=450.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        outer.add_child(PeriodicStatusLogger(
            self._ego,
            self._merge_car,
            self._pickup,
            interval=1.0,
            name="StatusLogger",
        ))
        outer.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            max_brake=self._ego_cap_brake,
            name="EgoSpeedGovernor",
        ))
        if self._ego_min_speed_kmh > 0.0:
            outer.add_child(_EgoMinSpeedForcer(
                self._ego,
                min_speed_kmh=self._ego_min_speed_kmh,
                throttle=self._ego_force_throttle,
            ))
        return outer

    def _create_test_criteria(self):
        return [CollisionTest(self._ego)]

    def __del__(self):
        self.remove_all_actors()
