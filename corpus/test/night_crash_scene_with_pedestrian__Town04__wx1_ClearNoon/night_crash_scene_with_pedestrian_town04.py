"""Night crash scene with pedestrian (Town04).

A night-time crash scene with stopped wrecks, debris, and a pedestrian in the roadway.
"""

import math

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy, WaypointFollower
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
)
from srunner.scenariomanager.scenarioatomics.custom_atomics_night_crash_scene_with_pedestrian_town04 import (
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


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    def __init__(self, ego, min_speed_kmh=0.0, throttle=0.85, name="EgoMinSpeedForcer"):
        super().__init__(name)
        self._ego = ego
        self._min_mps = float(min_speed_kmh) / 3.6
        self._throttle = float(throttle)

    def update(self):
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.RUNNING
        if _speed_mps(self._ego) < self._min_mps:
            ctrl = self._ego.get_control()
            ctrl.brake = 0.0
            ctrl.throttle = self._throttle
            self._ego.apply_control(ctrl)
        return py_trees.common.Status.RUNNING
class _EgoTimedThrottleForcer(py_trees.behaviour.Behaviour):
    def __init__(self, ego, start_after_s, duration_s, throttle=0.85,
                 name="EgoTimedThrottleForcer"):
        super().__init__(name)
        self._ego = ego
        self._start_after_s = float(start_after_s)
        self._duration_s = float(duration_s)
        self._throttle = float(throttle)
        self._started_at = None

    def initialise(self):
        self._started_at = GameTime.get_time()

    def update(self):
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.RUNNING
        elapsed = GameTime.get_time() - self._started_at
        if (elapsed < self._start_after_s or
                elapsed >= self._start_after_s + self._duration_s):
            return py_trees.common.Status.RUNNING
        control = self._ego.get_control()
        control.brake = 0.0
        control.throttle = self._throttle
        self._ego.apply_control(control)
        return py_trees.common.Status.RUNNING


class _RunADASVariant(py_trees.behaviour.Behaviour):
    def __init__(self, scenario_ref, name="RunADASVariant"):
        super().__init__(name)
        self._scenario = scenario_ref
        self._started_at = None
        self._last_time = None

    def initialise(self):
        now = GameTime.get_time()
        self._started_at = now
        self._last_time = now

    def update(self):
        scenario = self._scenario
        elapsed = 0.0 if self._started_at is None else max(0.0, GameTime.get_time() - self._started_at)
        dt = 0.05 if self._last_time is None else max(0.02, min(0.10, GameTime.get_time() - self._last_time))
        self._last_time = GameTime.get_time()

        scenario._run_variant(elapsed, dt)

        if elapsed >= scenario._scenario_duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class _OneShot(py_trees.behaviour.Behaviour):
    def __init__(self, callback, name="OneShot"):
        super().__init__(name)
        self._callback = callback
        self._done = False

    def update(self):
        if not self._done:
            self._callback()
            self._done = True
        return py_trees.common.Status.SUCCESS


class _WaitUntilAlongside(py_trees.behaviour.Behaviour):
    def __init__(self, ego, actor, max_forward_m=6.0, name="WaitUntilAlongside"):
        super().__init__(name)
        self._ego = ego
        self._actor = actor
        self._max_forward = float(max_forward_m)

    def update(self):
        if not (self._ego and self._actor and self._ego.is_alive and self._actor.is_alive):
            return py_trees.common.Status.RUNNING
        actor_tf = self._actor.get_transform()
        ego_loc = self._ego.get_location()
        actor_loc = actor_tf.location
        forward = actor_tf.get_forward_vector()
        forward_offset = ((ego_loc.x - actor_loc.x) * forward.x +
                          (ego_loc.y - actor_loc.y) * forward.y)
        if abs(forward_offset) <= self._max_forward:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class _ForcedLateralIntrusion(py_trees.behaviour.Behaviour):
    def __init__(self, actor, forward_speed_mps, lateral_speed_mps,
                 duration, lateral_sign, steer=0.28, name="ForcedLateralIntrusion"):
        super().__init__(name)
        self._actor = actor
        self._forward_speed = float(forward_speed_mps)
        self._lateral_speed = float(lateral_speed_mps)
        self._duration = float(duration)
        self._lateral_sign = float(lateral_sign)
        self._steer = float(steer)
        self._start_time = None

    def initialise(self):
        self._start_time = GameTime.get_time()
        # Temporarily disable physics for instant lateral response;
        # will re-enable before collision window so sensors detect impact.
        if self._actor is not None and self._actor.is_alive:
            try:
                self._actor.set_simulate_physics(False)
            except Exception:
                pass
        self._physics_restored = False

    def update(self):
        if self._actor is None or not self._actor.is_alive:
            return py_trees.common.Status.SUCCESS
        elapsed = GameTime.get_time() - self._start_time if self._start_time is not None else 0.0
        if elapsed >= self._duration:
            # Ensure physics is on when we finish
            if not self._physics_restored and self._actor is not None and self._actor.is_alive:
                try:
                    self._actor.set_simulate_physics(True)
                except Exception:
                    pass
            return py_trees.common.Status.SUCCESS
        transform = self._actor.get_transform()
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()
        lateral = self._lateral_speed * self._lateral_sign
        # Kinematic phase: instant velocity (first ~0.15s)
        if not self._physics_restored:
            if elapsed >= 0.15:
                # Restore physics so collision sensors detect the impact
                try:
                    self._actor.set_simulate_physics(True)
                except Exception:
                    pass
                self._physics_restored = True
            else:
                self._actor.set_target_velocity(carla.Vector3D(
                    forward.x * self._forward_speed + right.x * lateral,
                    forward.y * self._forward_speed + right.y * lateral,
                    0.0,
                ))
                return py_trees.common.Status.RUNNING
        # Physics phase: steer + throttle to maintain lateral push
        self._actor.apply_control(carla.VehicleControl(
            throttle=0.30, brake=0.0, steer=self._steer * self._lateral_sign,
        ))
        return py_trees.common.Status.RUNNING


class NightCrashSceneWithPedestrianTown04(BasicScenario):
    """Single reusable class for highway ADAS benchmark subcases."""

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
    def _vector_add(location, vector, scale):
        return carla.Location(
            x=location.x + vector.x * scale,
            y=location.y + vector.y * scale,
            z=location.z + vector.z * scale,
        )

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._ego = ego_vehicles[0]

        op = dict(config.other_parameters) if hasattr(config, "other_parameters") else {}
        if hasattr(config, "scenario") and hasattr(config.scenario, "other_parameters"):
            op.update(config.scenario.other_parameters)
        self._variant = str(self._get_param(op, "variant", "lead_reveal_stop", cast=str))
        self._activation_distance = self._get_param(op, "activation_distance_m", 120.0)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 74.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.10)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 0.0)
        self._ego_force_throttle = self._get_param(op, "ego_force_throttle", 0.85)
        self._scenario_duration = self._get_param(op, "scenario_duration_s", 6.0)
        self._shoulder_truck_spawn_ahead_m = self._get_param(
            op, "shoulder_truck_spawn_ahead_m", 34.0)
        self._shoulder_truck_reveal_distance_m = self._get_param(
            op, "shoulder_truck_reveal_distance_m", 0.0)
        self._shoulder_truck_reveal_after_s = self._get_param(
            op, "shoulder_truck_reveal_after_s", 0.0)
        self._shoulder_truck_ego_hold_s = self._get_param(
            op, "shoulder_truck_ego_hold_s", 0.0)
        self._shoulder_truck_ego_hold_throttle = self._get_param(
            op, "shoulder_truck_ego_hold_throttle", 0.85)
        self._animal_track_ego = bool(self._get_param(
            op, "animal_track_ego_target", 0, cast=int))
        self._animal_intercept_lead = self._get_param(
            op, "animal_intercept_lead_m", 0.8)
        self._animal_cross_speed = self._get_param(
            op, "animal_cross_speed_mps", 3.0)
        self._animal_cross_start_delay_s = self._get_param(
            op, "animal_cross_start_delay_s", 0.0)
        self._animal_spawn_ahead_m = self._get_param(
            op, "animal_spawn_ahead_m", 34.0)
        self._animal_cross_distance_limit = self._get_param(
            op, "animal_cross_distance_limit_m", 8.0)
        self._animal_start_lateral = self._get_param(
            op, "animal_start_lateral_m", 5.0)
        self._animal_disable_physics = bool(self._get_param(
            op, "animal_disable_physics", 0, cast=int))
        self._animal_disable_collision_on_hit = bool(self._get_param(
            op, "animal_disable_collision_on_hit", 0, cast=int))
        self._animal_stabilize_ego_on_collision = bool(self._get_param(
            op, "animal_stabilize_ego_on_collision", 0, cast=int))
        self._animal_stop_on_collision = bool(self._get_param(
            op, "animal_stop_on_collision", 1, cast=int))
        self._road_friction_scale = self._get_param(op, "road_friction_scale", 1.0)

        self._merge_spawn_ahead_m = self._get_param(op, "merge_spawn_ahead_m", 18.0)
        self._merge_initial_speed_mps = self._get_param(op, "merge_initial_speed_kmh", 79.2) / 3.6
        self._merge_cut_speed_mps = self._get_param(op, "merge_cut_speed_kmh", 86.4) / 3.6
        self._merge_settle_speed_mps = self._get_param(op, "merge_settle_speed_kmh", 64.8) / 3.6
        self._merge_cut_start_s = self._get_param(op, "merge_cut_start_s", 0.9)
        self._merge_cut_duration_s = self._get_param(op, "merge_cut_duration_s", 0.8)
        self._merge_lateral_speed_mps = self._get_param(op, "merge_lateral_speed_mps", -5.8)
        self._merge_yaw_deg = self._get_param(op, "merge_yaw_deg", -18.0)
        self._merge_physics_restore_progress = self._get_param(
            op, "merge_physics_restore_progress", 0.75)

        self._semi_spawn_ahead_m = self._get_param(op, "semi_spawn_ahead_m", 18.0)
        self._semi_spawn_lateral_offset_m = self._get_param(op, "semi_spawn_lateral_offset_m", 0.0)
        self._semi_initial_speed_mps = self._get_param(op, "semi_initial_speed_kmh", 72.0) / 3.6
        self._semi_intrusion_speed_mps = self._get_param(op, "semi_intrusion_speed_kmh", 79.2) / 3.6
        self._semi_settle_speed_mps = self._get_param(op, "semi_settle_speed_kmh", 75.6) / 3.6
        self._semi_cut_start_s = self._get_param(op, "semi_cut_start_s", 0.9)
        self._semi_cut_duration_s = self._get_param(op, "semi_cut_duration_s", 0.65)
        self._semi_settle_duration_s = self._get_param(op, "semi_settle_duration_s", 0.55)
        self._semi_lateral_speed_mps = self._get_param(op, "semi_lateral_speed_mps", 4.2)
        self._semi_yaw_deg = self._get_param(op, "semi_yaw_deg", 14.0)
        self._semi_lateral_sign_override = self._get_param(op, "semi_lateral_sign", 0.0)
        self._semi_spawn_side = str(self._get_param(op, "semi_spawn_side", "auto", cast=str)).lower()
        self._semi_invert_intrusion = self._get_param(op, "semi_invert_intrusion", 0.0) > 0.5
        self._semi_alongside_threshold_m = self._get_param(op, "semi_alongside_threshold_m", 7.0)
        self._semi_aftermath_duration_s = self._get_param(op, "semi_aftermath_duration_s", 5.0)
        self._semi_model = str(self._get_param(
            op, "semi_model", "vehicle.carlamotors.european_hgv", cast=str))
        self._semi_color = str(self._get_param(op, "semi_color", "30,80,210", cast=str))

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._left_lane_wp = None
        self._right_lane_wp = None
        self._actors = []
        self._walker = None
        self._lead = None
        self._semi_plan = []
        self._hazard = None
        self._support = None
        self._extra_props = []
        self._merge_collision_sensor = None
        self._workzone_collision_sensor = None
        self._state = {}

        super().__init__(
            name="NightCrashSceneWithPedestrianTown04",
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
            world.spawn_actor(bp, carla.Transform(carla.Location(-10000.0, -10000.0, 0.0)))
        except RuntimeError:
            pass

    def _spawn_vehicle(self, models, transform, rolename, color=None, simulate_physics=True):
        for model in models:
            actor = CarlaDataProvider.request_new_actor(model, transform, rolename=rolename, color=color)
            if actor is not None:
                actor.set_simulate_physics(simulate_physics)
                if simulate_physics:
                    actor.set_enable_gravity(True)
                else:
                    actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                    actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                self.other_actors.append(actor)
                self._actors.append(actor)
                return actor
        return None

    def _on_merge_collision(self, event):
        if self._ego is None or event.other_actor is None:
            return
        if event.other_actor.id != self._ego.id:
            return
        if not self._state.get("merge_collision_seen", False):
            self._state["merge_collision_seen"] = True
            print("[ADASuite] right merge actor collided with ego", flush=True)


    def _on_workzone_collision(self, event):
        if self._ego is None or event.other_actor is None:
            return
        if event.other_actor.id == self._ego.id:
            return
        if not self._state.get("workzone_collision_seen", False):
            self._state["workzone_collision_seen"] = True
            print(
                f"[ADASuite] ego collided with workzone obstacle "
                f"type={event.other_actor.type_id}",
                flush=True,
            )
    @staticmethod
    def _set_vehicle_lights(actor, hazard=False, brake=False, position=True):
        if actor is None or not actor.is_alive:
            return
        flags = 0
        if position:
            flags |= carla.VehicleLightState.Position
        if brake:
            flags |= carla.VehicleLightState.Brake
        if hazard:
            flags |= carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker
        try:
            actor.set_light_state(carla.VehicleLightState(flags))
        except RuntimeError:
            pass

    def _spawn_prop(self, models, transform, rolename):
        for model in models:
            actor = CarlaDataProvider.request_new_actor(model, transform, rolename=rolename)
            if actor is not None:
                actor.set_simulate_physics(False)
                self.other_actors.append(actor)
                self._extra_props.append(actor)
                return actor
        return None

    def _spawn_walker(self, transform):
        candidates = [
            "walker.pedestrian.0013",
            "walker.pedestrian.0014",
            "walker.animal.1007",
            "walker.animal.1008",
        ]
        lib = self._world.get_blueprint_library()
        existing = {bp.id for bp in lib}
        for model in candidates:
            if model not in existing:
                continue
            actor = CarlaDataProvider.request_new_actor(model, transform, rolename="adas_animal")
            if actor is not None:
                self.other_actors.append(actor)
                self._walker = actor
                return actor
        return None

    def _spawn_pedestrian(self, transform):
        candidates = [
            "walker.pedestrian.0013",
            "walker.pedestrian.0014",
            "walker.pedestrian.0001",
            "walker.pedestrian.0005",
        ]
        lib = self._world.get_blueprint_library()
        existing = {bp.id for bp in lib}
        for model in candidates:
            if model not in existing:
                continue
            actor = CarlaDataProvider.request_new_actor(model, transform, rolename="adas_pedestrian")
            if actor is not None:
                self.other_actors.append(actor)
                self._walker = actor
                return actor
        return None

    def _build_plan(self, start_wp, distance=800.0, step=2.0):
        waypoints = []
        cursor = start_wp
        travelled = 0.0
        while cursor is not None and travelled < distance:
            candidates = cursor.next(step)
            if not candidates:
                break
            cursor = candidates[0]
            waypoints.append(cursor)
            travelled += step
        return waypoints

    def _register_pose(self, actor, waypoint):
        return {
            "loc": carla.Location(
                x=waypoint.transform.location.x,
                y=waypoint.transform.location.y,
                z=waypoint.transform.location.z + 0.5,
            ),
            "forward": waypoint.transform.get_forward_vector(),
            "right": waypoint.transform.get_right_vector(),
            "yaw": waypoint.transform.rotation.yaw,
            "actor": actor,
        }

    def _set_pose(self, pose, yaw_offset_deg, forward_speed, lateral_speed):
        actor = pose["actor"]
        if actor is None or not actor.is_alive:
            return
        actor.set_transform(carla.Transform(
            pose["loc"],
            carla.Rotation(yaw=pose["yaw"] + yaw_offset_deg),
        ))
        actor.set_target_velocity(carla.Vector3D(
            pose["forward"].x * forward_speed + pose["right"].x * lateral_speed,
            pose["forward"].y * forward_speed + pose["right"].y * lateral_speed,
            0.0,
        ))

    def _advance_pose(self, pose, dt, forward_speed, lateral_speed):
        pose["loc"] = self._vector_add(pose["loc"], pose["forward"], forward_speed * dt)
        pose["loc"] = self._vector_add(pose["loc"], pose["right"], lateral_speed * dt)

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if self._variant in (
                "night_shoulder_truck",
                "dark_crash_blockage",
                "hazard_vehicle_appears",
                "crash_scene_pedestrian_escape",
                "smoke_cloud_crash_scene"):
            self._set_vehicle_lights(self._ego, hazard=False, brake=False, position=True)
            try:
                self._ego.set_light_state(carla.VehicleLightState(
                    carla.VehicleLightState.Position |
                    carla.VehicleLightState.LowBeam |
                    carla.VehicleLightState.HighBeam
                ))
            except RuntimeError:
                pass
        self._left_lane_wp = ego_wp.get_left_lane()
        self._right_lane_wp = ego_wp.get_right_lane()
        if self._left_lane_wp is not None and self._left_lane_wp.lane_type != carla.LaneType.Driving:
            self._left_lane_wp = None
        if self._right_lane_wp is not None and self._right_lane_wp.lane_type != carla.LaneType.Driving:
            self._right_lane_wp = None

        handler = getattr(self, f"_setup_{self._variant}", None)
        if handler is None:
            raise RuntimeError(f"[ADASuite] Unknown variant: {self._variant}")
        handler(ego_wp)

        print(
            f"\n[ADASuite] Spawn\n"
            f"  variant={self._variant}\n"
            f"  ego road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  actors={len(self._actors)} props={len(self._extra_props)} walker={'yes' if self._walker else 'no'}\n",
            flush=True,
        )

    def _setup_lead_reveal_stop(self, ego_wp):
        lead_wp = self._walk_waypoint(ego_wp, 22.0, forward=True) or ego_wp
        stopped_wp = self._walk_waypoint(ego_wp, 42.0, forward=True) or lead_wp
        lead = self._spawn_vehicle(["vehicle.tesla.model3", "vehicle.audi.tt"], carla.Transform(
            carla.Location(lead_wp.transform.location.x, lead_wp.transform.location.y, lead_wp.transform.location.z + 0.5),
            lead_wp.transform.rotation,
        ), "adas_lead")
        stopped = self._spawn_vehicle(["vehicle.carlamotors.european_hgv", "vehicle.tesla.cybertruck", "vehicle.tesla.model3"], carla.Transform(
            carla.Location(stopped_wp.transform.location.x, stopped_wp.transform.location.y, stopped_wp.transform.location.z),
            stopped_wp.transform.rotation,
        ), "adas_stopped", color="40,40,40", simulate_physics=False)
        self._lead = self._register_pose(lead, lead_wp)
        self._hazard = self._register_pose(stopped, stopped_wp)
        # Override z to road surface — kinematic HGV has mesh origin at road level,
        # so no +0.5 offset is needed (unlike physics-enabled actors that need clearance)
        if self._hazard is not None:
            self._hazard["loc"] = carla.Location(
                x=stopped_wp.transform.location.x,
                y=stopped_wp.transform.location.y,
                z=stopped_wp.transform.location.z,
            )
        if self._left_lane_wp is not None:
            blocker_wp = self._walk_waypoint(self._left_lane_wp, 10.0, forward=True) or self._left_lane_wp
            blocker = self._spawn_vehicle(["vehicle.nissan.patrol", "vehicle.audi.tt"], carla.Transform(
                carla.Location(blocker_wp.transform.location.x, blocker_wp.transform.location.y, blocker_wp.transform.location.z + 0.5),
                blocker_wp.transform.rotation,
            ), "adas_left_blocker", color="80,80,120")
            self._support = self._register_pose(blocker, blocker_wp)
        self._state["lead_speed"] = 21.0

    def _setup_workzone_merge(self, ego_wp):
        is_compact_town_route = self._world.get_map().name.endswith(("Town12", "Town13"))
        distances = [62.0] * 4 if is_compact_town_route else [116.0, 118.0]
        forward = not is_compact_town_route
        right = ego_wp.transform.get_right_vector()

        def placement_transform(distance):
            if is_compact_town_route:
                location = self._vector_add(
                    ego_wp.transform.location,
                    ego_wp.transform.get_forward_vector(),
                    distance,
                )
                return carla.Transform(location, ego_wp.transform.rotation)
            wp = self._walk_waypoint(ego_wp, distance, forward=forward) or ego_wp
            return wp.transform

        for index, dist in enumerate(distances):
            placement = placement_transform(dist)
            lateral_offset = (index - 1.5) * 1.2 if is_compact_town_route else index * 0.85
            loc = carla.Location(
                x=placement.location.x + right.x * lateral_offset,
                y=placement.location.y + right.y * lateral_offset,
                z=placement.location.z + 0.05,
            )
            prop = self._spawn_prop(
                ["static.prop.streetbarrier", "static.prop.chainbarrierend"],
                carla.Transform(loc, placement.rotation),
                f"adas_workzone_{index}",
            )
            if prop is not None:
                try:
                    prop.set_simulate_physics(True)
                    prop.set_enable_gravity(False)
                except RuntimeError:
                    pass
                if index == 0:
                    self._hazard = {"actor": prop}

        sign_transform = placement_transform(48.0 if is_compact_town_route else 90.0)
        sign_right = sign_transform.get_right_vector()
        sign_loc = carla.Location(
            x=sign_transform.location.x + sign_right.x * 3.2,
            y=sign_transform.location.y + sign_right.y * 3.2,
            z=sign_transform.location.z + 0.1,
        )
        self._spawn_prop(
            ["static.prop.warningconstruction", "static.prop.trafficwarning"],
            carla.Transform(sign_loc, sign_transform.rotation),
            "adas_workzone_warning",
        )
        cone_distances = (30.0, 42.0, 54.0) if is_compact_town_route else (78.0, 86.0, 96.0)
        for index, dist in enumerate(cone_distances):
            cone_transform = placement_transform(dist)
            cone_right = cone_transform.get_right_vector()
            cone_loc = carla.Location(
                x=cone_transform.location.x + cone_right.x * 2.8,
                y=cone_transform.location.y + cone_right.y * 2.8,
                z=cone_transform.location.z + 0.05,
            )
            self._spawn_prop(
                ["static.prop.constructioncone"],
                carla.Transform(cone_loc, cone_transform.rotation),
                f"adas_workzone_cone_{index}",
            )
        collision_bp = self._world.get_blueprint_library().find(
            "sensor.other.collision")
        try:
            self._workzone_collision_sensor = self._world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=self._ego)
            self._workzone_collision_sensor.listen(self._on_workzone_collision)
            self.other_actors.append(self._workzone_collision_sensor)
        except RuntimeError:
            self._workzone_collision_sensor = None

    def _setup_night_shoulder_truck(self, ego_wp):
        truck_wp = self._walk_waypoint(
            ego_wp, self._shoulder_truck_spawn_ahead_m, forward=True) or ego_wp
        right = truck_wp.transform.get_right_vector()
        loc = carla.Location(
            x=truck_wp.transform.location.x,
            y=truck_wp.transform.location.y,
            z=truck_wp.transform.location.z + 0.5,
        )
        if (self._shoulder_truck_reveal_after_s > 0.0 or
                self._shoulder_truck_reveal_distance_m > 0.0):
            self._state["shoulder_truck_reveal_pose"] = {
                "loc": carla.Location(x=loc.x, y=loc.y, z=loc.z),
                "forward": truck_wp.transform.get_forward_vector(),
                "right": truck_wp.transform.get_right_vector(),
                "yaw": truck_wp.transform.rotation.yaw,
            }
            loc = carla.Location(
                x=loc.x + right.x * 30.0,
                y=loc.y + right.y * 30.0,
                z=loc.z,
            )
        truck = self._spawn_vehicle(["vehicle.carlamotors.european_hgv", "vehicle.carlamotors.carlacola"], carla.Transform(
            loc, truck_wp.transform.rotation,
        ), "adas_shoulder_truck", color="20,20,20", simulate_physics=False)
        self._hazard = self._register_pose(truck, truck_wp)
        self._hazard["loc"] = loc
        self._state["shoulder_truck_hidden"] = (
            self._shoulder_truck_reveal_after_s > 0.0 or self._shoulder_truck_reveal_distance_m > 0.0)
        sign_wp = self._walk_waypoint(ego_wp, 24.0, forward=True) or ego_wp
        self._spawn_prop(["static.prop.streetbarrier", "static.prop.chainbarrier"], carla.Transform(
            carla.Location(sign_wp.transform.location.x + right.x * 2.0, sign_wp.transform.location.y + right.y * 2.0, sign_wp.transform.location.z + 0.05),
            sign_wp.transform.rotation,
        ), "adas_shoulder_warning")

    def _setup_dark_crash_blockage(self, ego_wp):
        crash_wp = self._walk_waypoint(ego_wp, 36.0, forward=True) or ego_wp
        crash = self._spawn_vehicle(["vehicle.tesla.model3", "vehicle.audi.tt"], carla.Transform(
            carla.Location(crash_wp.transform.location.x, crash_wp.transform.location.y, crash_wp.transform.location.z + 0.5),
            carla.Rotation(yaw=crash_wp.transform.rotation.yaw + 24.0),
        ), "adas_dark_crash", color="10,10,10")
        self._hazard = self._register_pose(crash, crash_wp)
        if self._left_lane_wp is not None:
            left_wp = self._walk_waypoint(self._left_lane_wp, 40.0, forward=True) or self._left_lane_wp
            blocker = self._spawn_vehicle(["vehicle.nissan.patrol", "vehicle.tesla.model3"], carla.Transform(
                carla.Location(left_wp.transform.location.x, left_wp.transform.location.y, left_wp.transform.location.z + 0.5),
                carla.Rotation(yaw=left_wp.transform.rotation.yaw - 12.0),
            ), "adas_dark_blocker", color="20,20,20")
            self._support = self._register_pose(blocker, left_wp)

    def _setup_hazard_vehicle_appears(self, ego_wp):
        lead_wp = self._walk_waypoint(ego_wp, 34.0, forward=True) or ego_wp
        hazard_wp = self._walk_waypoint(ego_wp, 38.0, forward=True) or lead_wp
        lead = self._spawn_vehicle(["vehicle.audi.tt", "vehicle.tesla.model3"], carla.Transform(
            carla.Location(lead_wp.transform.location.x, lead_wp.transform.location.y, lead_wp.transform.location.z + 0.5),
            lead_wp.transform.rotation,
        ), "adas_hazard_occluder", color="170,170,175")

        right = hazard_wp.transform.get_right_vector()
        hazard_loc = carla.Location(
            x=hazard_wp.transform.location.x + right.x * 1.2,
            y=hazard_wp.transform.location.y + right.y * 1.2,
            z=hazard_wp.transform.location.z,
        )
        hazard = self._spawn_vehicle(["vehicle.nissan.patrol", "vehicle.carlamotors.carlacola", "vehicle.tesla.model3"], carla.Transform(
            hazard_loc,
            carla.Rotation(yaw=hazard_wp.transform.rotation.yaw + 7.0),
        ), "adas_hazard_vehicle", color="245,245,245", simulate_physics=False)

        self._lead = self._register_pose(lead, lead_wp)
        self._hazard = self._register_pose(hazard, hazard_wp)
        self._hazard["loc"] = hazard_loc
        if hazard is not None:
            self._set_vehicle_lights(hazard, hazard=True, brake=True, position=True)

        if self._left_lane_wp is not None:
            blocker_wp = self._walk_waypoint(self._left_lane_wp, 12.0, forward=True) or self._left_lane_wp
            blocker = self._spawn_vehicle(["vehicle.audi.tt", "vehicle.nissan.micra"], carla.Transform(
                carla.Location(blocker_wp.transform.location.x, blocker_wp.transform.location.y, blocker_wp.transform.location.z + 0.5),
                blocker_wp.transform.rotation,
            ), "adas_hazard_escape_blocker", color="100,110,140")
            self._support = self._register_pose(blocker, blocker_wp)

        self._state["lead_speed"] = 16.0

    def _setup_semi_intrusion_clear_escape(self, ego_wp):
        if self._semi_spawn_side == "right" and self._right_lane_wp is not None:
            source_lane = self._right_lane_wp
        elif self._semi_spawn_side == "left" and self._left_lane_wp is not None:
            source_lane = self._left_lane_wp
        else:
            source_lane = self._left_lane_wp if self._left_lane_wp is not None else self._right_lane_wp
        if source_lane is None:
            raise RuntimeError("[ADASuite] semi_intrusion_clear_escape requires an adjacent lane")

        truck_wp = self._walk_waypoint(
            source_lane,
            abs(self._semi_spawn_ahead_m),
            forward=self._semi_spawn_ahead_m >= 0.0,
        ) or source_lane
        ego_loc = ego_wp.transform.location
        truck_loc = truck_wp.transform.location
        right = truck_wp.transform.get_right_vector()
        ego_lateral_offset = ((ego_loc.x - truck_loc.x) * right.x +
                              (ego_loc.y - truck_loc.y) * right.y)
        intrusion_sign = 1.0 if ego_lateral_offset >= 0.0 else -1.0
        if abs(self._semi_lateral_sign_override) > 0.1:
            override_sign = 1.0 if self._semi_lateral_sign_override > 0.0 else -1.0
            if override_sign * ego_lateral_offset > 0.0:
                intrusion_sign = override_sign
        if self._semi_invert_intrusion:
            intrusion_sign *= -1.0
        truck = self._spawn_vehicle([
            self._semi_model,
            "vehicle.carlamotors.european_hgv",
            "vehicle.carlamotors.carlacola",
            "vehicle.tesla.cybertruck",
        ], carla.Transform(
            carla.Location(truck_wp.transform.location.x, truck_wp.transform.location.y, truck_wp.transform.location.z + 0.5),
            truck_wp.transform.rotation,
        ), "adas_semi_intrusion", color=self._semi_color)
        self._lead = self._register_pose(truck, truck_wp)
        if self._lead is not None and abs(self._semi_spawn_lateral_offset_m) > 0.01:
            self._lead["loc"] = self._vector_add(
                self._lead["loc"],
                self._lead["right"],
                self._semi_spawn_lateral_offset_m,
            )
            truck_transform = truck.get_transform()
            truck_transform.location = carla.Location(
                self._lead["loc"].x,
                self._lead["loc"].y,
                self._lead["loc"].z + 0.5,
            )
            truck.set_transform(truck_transform)
        self._semi_plan = self._build_plan(truck_wp)
        self._state["intrusion_sign"] = intrusion_sign
        self._state["lead_speed"] = self._semi_initial_speed_mps
        self._state["semi_lanechange_active"] = True

    def _setup_crash_scene_pedestrian_escape(self, ego_wp):
        crash_wp = self._walk_waypoint(ego_wp, 30.0, forward=True) or ego_wp
        right = crash_wp.transform.get_right_vector()
        crash_loc = carla.Location(
            x=crash_wp.transform.location.x + right.x * 0.9,
            y=crash_wp.transform.location.y + right.y * 0.9,
            z=crash_wp.transform.location.z + 0.08,
        )
        crash = self._spawn_vehicle(["vehicle.tesla.model3", "vehicle.audi.tt"], carla.Transform(
            crash_loc,
            carla.Rotation(yaw=crash_wp.transform.rotation.yaw + 20.0),
        ), "adas_crash_scene_car", color="120,120,120", simulate_physics=True)
        self._hazard = self._register_pose(crash, crash_wp)
        self._hazard["loc"] = crash_loc
        self._set_vehicle_lights(crash, hazard=True, brake=True, position=True)

        debris_loc = carla.Location(
            x=crash_wp.transform.location.x + right.x * 1.5,
            y=crash_wp.transform.location.y + right.y * 1.5,
            z=crash_wp.transform.location.z + 0.02,
        )
        self._spawn_prop(
            ["static.prop.dirtdebris01", "static.prop.streetbarrier", "static.prop.chainbarrierend"],
            carla.Transform(debris_loc, crash_wp.transform.rotation),
            "adas_crash_scene_debris",
        )

        walker_loc = carla.Location(
            x=crash_wp.transform.location.x + right.x * 2.2,
            y=crash_wp.transform.location.y + right.y * 2.2,
            z=crash_wp.transform.location.z + 0.08,
        )
        self._spawn_pedestrian(carla.Transform(walker_loc, crash_wp.transform.rotation))

        white_base = self._vector_add(crash_wp.transform.location, crash_wp.transform.get_forward_vector(), 8.0)
        white_loc = carla.Location(
            x=white_base.x + right.x * 2.8,
            y=white_base.y + right.y * 2.8,
            z=crash_wp.transform.location.z + 0.08,
        )
        white = self._spawn_vehicle(["vehicle.tesla.model3", "vehicle.audi.tt"], carla.Transform(
            white_loc,
            carla.Rotation(yaw=crash_wp.transform.rotation.yaw - 8.0),
        ), "adas_crash_scene_white_sedan", color="245,245,245", simulate_physics=False)
        if white is not None:
            self._set_vehicle_lights(white, hazard=True, brake=True, position=True)
            white_wp = self._walk_waypoint(crash_wp, 8.0, forward=True) or crash_wp
            self._state["crash_white"] = self._register_pose(white, white_wp)
            self._state["crash_white"]["loc"] = white_loc

        if self._left_lane_wp is not None:
            pass_wp = self._walk_waypoint(self._left_lane_wp, 16.0, forward=True) or self._left_lane_wp
            pass_loc = carla.Location(
                pass_wp.transform.location.x,
                pass_wp.transform.location.y,
                pass_wp.transform.location.z + 0.08,
            )
            passer = self._spawn_vehicle(["vehicle.audi.tt", "vehicle.tesla.model3"], carla.Transform(
                pass_loc,
                pass_wp.transform.rotation,
            ), "adas_crash_scene_passer", color="70,90,140")
            self._support = self._register_pose(passer, pass_wp)
            if self._support is not None:
                self._support["loc"] = pass_loc

    def _setup_smoke_cloud_crash_scene(self, ego_wp):
        crash_wp = self._walk_waypoint(ego_wp, 28.0, forward=True) or ego_wp
        right = crash_wp.transform.get_right_vector()
        smoke_wp = self._walk_waypoint(ego_wp, 18.0, forward=True) or ego_wp
        for index, offset in enumerate([-2.2, -1.1, 0.0, 1.1, 2.2]):
            loc = carla.Location(
                x=smoke_wp.transform.location.x + right.x * offset,
                y=smoke_wp.transform.location.y + right.y * offset,
                z=smoke_wp.transform.location.z + 0.05,
            )
            self._spawn_prop(
                ["static.prop.dirtdebris01", "static.prop.streetbarrier", "static.prop.chainbarrierend"],
                carla.Transform(loc, carla.Rotation(yaw=smoke_wp.transform.rotation.yaw + 15.0 * index)),
                f"adas_smoke_debris_{index}",
            )

        overturned_loc = carla.Location(
            x=crash_wp.transform.location.x + right.x * 0.9,
            y=crash_wp.transform.location.y + right.y * 0.9,
            z=crash_wp.transform.location.z + 0.5,
        )
        overturned = self._spawn_vehicle(["vehicle.audi.tt", "vehicle.tesla.model3"], carla.Transform(
            overturned_loc,
            carla.Rotation(yaw=crash_wp.transform.rotation.yaw + 68.0),
        ), "adas_smoke_overturned", color="45,45,45", simulate_physics=False)
        self._hazard = self._register_pose(overturned, crash_wp)
        self._state["smoke_overturned_transform"] = carla.Transform(
            overturned_loc,
            carla.Rotation(yaw=crash_wp.transform.rotation.yaw + 68.0),
        )

        sedan_base = self._vector_add(crash_wp.transform.location, crash_wp.transform.get_forward_vector(), 7.0)
        sedan_loc = carla.Location(
            x=sedan_base.x + right.x * 2.4,
            y=sedan_base.y + right.y * 2.4,
            z=crash_wp.transform.location.z + 0.5,
        )
        sedan = self._spawn_vehicle(["vehicle.tesla.model3", "vehicle.audi.tt"], carla.Transform(
            sedan_loc,
            carla.Rotation(yaw=crash_wp.transform.rotation.yaw - 16.0),
        ), "adas_smoke_white_sedan", color="245,245,245", simulate_physics=False)
        sedan_wp = self._walk_waypoint(crash_wp, 7.0, forward=True) or crash_wp
        self._support = self._register_pose(sedan, sedan_wp)
        self._support["loc"] = sedan_loc
        self._set_vehicle_lights(sedan, hazard=True, brake=True, position=True)

    def _setup_aggressive_merge(self, ego_wp):
        merge_wp = self._right_lane_wp
        if merge_wp is not None:
            merge_wp = self._walk_waypoint(
                merge_wp, self._merge_spawn_ahead_m, forward=True,
            ) or self._right_lane_wp
            merge_transform = carla.Transform(
                carla.Location(
                    merge_wp.transform.location.x,
                    merge_wp.transform.location.y,
                    merge_wp.transform.location.z + 0.5,
                ),
                merge_wp.transform.rotation,
            )
        else:
            # Some highway segments expose no adjacent lane in the map even
            # though the shoulder is wide enough for a right-side merge.
            forward = ego_wp.transform.get_forward_vector()
            right = ego_wp.transform.get_right_vector()
            merge_loc = carla.Location(
                ego_wp.transform.location.x + forward.x * self._merge_spawn_ahead_m + right.x * 5.5,
                ego_wp.transform.location.y + forward.y * self._merge_spawn_ahead_m + right.y * 5.5,
                ego_wp.transform.location.z + 0.5,
            )
            merge_transform = carla.Transform(merge_loc, ego_wp.transform.rotation)
        merge = self._spawn_vehicle(
            ["vehicle.audi.tt", "vehicle.tesla.model3"],
            merge_transform,
            "adas_aggressive_merge",
            color="220,30,30",
            simulate_physics=True,
        )
        self._lead = self._register_pose(merge, ego_wp if merge_wp is None else merge_wp)
        if merge_wp is None:
            self._lead["loc"] = merge_transform.location
            self._lead["forward"] = ego_wp.transform.get_forward_vector()
            self._lead["right"] = ego_wp.transform.get_right_vector()
        if merge is not None:
            collision_bp = self._world.get_blueprint_library().find(
                "sensor.other.collision")
            try:
                self._merge_collision_sensor = self._world.spawn_actor(
                    collision_bp, carla.Transform(), attach_to=merge)
                self._merge_collision_sensor.listen(self._on_merge_collision)
                self.other_actors.append(self._merge_collision_sensor)
            except RuntimeError:
                self._merge_collision_sensor = None

    def _setup_animal_cross(self, ego_wp):
        animal_wp = self._walk_waypoint(
            ego_wp, self._animal_spawn_ahead_m, forward=True) or ego_wp
        right = animal_wp.transform.get_right_vector()
        loc = carla.Location(
            x=animal_wp.transform.location.x + right.x * 5.0,
            y=animal_wp.transform.location.y + right.y * 5.0,
            z=animal_wp.transform.location.z + 0.5,
        )
        self._state["animal_start_loc"] = carla.Location(loc.x, loc.y, loc.z)
        self._state["animal_cross_distance_m"] = 0.0
        self._state["animal_cross_started"] = False
        walker = self._spawn_walker(
            carla.Transform(loc, animal_wp.transform.rotation))
        if walker is not None and self._animal_disable_physics:
            walker.set_simulate_physics(False)
        if self._left_lane_wp is not None:
            blocker_wp = self._walk_waypoint(self._left_lane_wp, 4.0, forward=True) or self._left_lane_wp
            blocker = self._spawn_vehicle(["vehicle.audi.tt", "vehicle.tesla.model3"], carla.Transform(
                carla.Location(blocker_wp.transform.location.x, blocker_wp.transform.location.y, blocker_wp.transform.location.z + 0.5),
                blocker_wp.transform.rotation,
            ), "adas_animal_blocker", color="120,120,120")
            self._support = self._register_pose(blocker, blocker_wp)

    def _run_variant(self, elapsed, dt):
        handler = getattr(self, f"_step_{self._variant}", None)
        if handler is not None:
            handler(elapsed, dt)

    def _step_lead_reveal_stop(self, elapsed, dt):
        lead_speed = self._state.get("lead_speed", 21.0)
        lateral_speed = 0.0
        yaw = 0.0
        if elapsed > 1.2:
            ratio = min(1.0, (elapsed - 1.2) / 0.7)
            lateral_speed = -4.8
            yaw = -14.0 * ratio
        if elapsed > 1.9:
            lead_speed = 16.0
            lateral_speed = 0.0
        self._advance_pose(self._lead, dt, lead_speed, lateral_speed)
        self._set_pose(self._lead, yaw, lead_speed, lateral_speed)
        if self._support is not None:
            support_speed = 20.0
            self._advance_pose(self._support, dt, support_speed, 0.0)
            self._set_pose(self._support, 0.0, support_speed, 0.0)
        # Pin the kinematic stopped truck at road surface each tick
        self._set_pose(self._hazard, 0.0, 0.0, 0.0)

    def _step_workzone_merge(self, elapsed, dt):
        return


    def _step_night_shoulder_truck(self, elapsed, dt):
        if self._state.get("shoulder_truck_hidden", False):
            if (self._shoulder_truck_reveal_after_s > 0.0 and
                    elapsed < self._shoulder_truck_reveal_after_s):
                self._set_pose(self._hazard, 0.0, 0.0, 0.0)
                return

            ego_loc = self._ego.get_location()
            trigger_loc = self._trigger_wp.transform.location
            distance = math.hypot(
                ego_loc.x - trigger_loc.x, ego_loc.y - trigger_loc.y)
            if (self._shoulder_truck_reveal_distance_m > 0.0 and
                    distance > self._shoulder_truck_reveal_distance_m):
                self._set_pose(self._hazard, 0.0, 0.0, 0.0)
                return
            reveal_pose = self._state["shoulder_truck_reveal_pose"]
            self._hazard["loc"] = reveal_pose["loc"]
            self._hazard["forward"] = reveal_pose["forward"]
            self._hazard["right"] = reveal_pose["right"]
            self._hazard["yaw"] = reveal_pose["yaw"]
            self._state["shoulder_truck_hidden"] = False
            self._state["shoulder_truck_revealed_at"] = elapsed
        self._set_pose(self._hazard, 0.0, 0.0, 0.0)

    def _step_dark_crash_blockage(self, elapsed, dt):
        self._set_pose(self._hazard, 24.0, 0.0, 0.0)
        if self._support is not None:
            self._set_pose(self._support, -12.0, 0.0, 0.0)

    def _step_hazard_vehicle_appears(self, elapsed, dt):
        lead_speed = self._state.get("lead_speed", 16.0)
        lateral_speed = 0.0
        yaw = 0.0
        if elapsed > 1.0:
            ratio = min(1.0, (elapsed - 1.0) / 0.8)
            lateral_speed = -4.0
            yaw = -12.0 * ratio
        if elapsed > 1.8:
            lead_speed = 20.0
            lateral_speed = 0.0

        self._advance_pose(self._lead, dt, lead_speed, lateral_speed)
        self._set_pose(self._lead, yaw, lead_speed, lateral_speed)
        self._set_pose(self._hazard, 7.0, 0.0, 0.0)

        if self._support is not None:
            support_speed = max(18.0, _speed_mps(self._ego))
            self._advance_pose(self._support, dt, support_speed, 0.0)
            self._set_pose(self._support, 0.0, support_speed, 0.0)

    def _step_semi_intrusion_clear_escape(self, elapsed, dt):
        if not self._state.get("semi_lanechange_active", False):
            return
        if self._lead is None:
            return
        actor = self._lead["actor"]
        if actor is None or not actor.is_alive:
            return

        intrusion_sign = self._state.get("intrusion_sign", 1.0)
        forward_speed = self._state.get("lead_speed", self._semi_initial_speed_mps)
        lateral_speed = 0.0
        yaw = 0.0
        cut_start = max(0.0, self._semi_cut_start_s)
        cut_end = cut_start + max(0.1, self._semi_cut_duration_s)
        settle_end = cut_end + max(0.1, self._semi_settle_duration_s)

        # Pre-cut: track alongside ego so the semi doesn't race ahead while ego
        # accelerates from rest.  Capture the lane lateral offset on the first
        # tick, then pin the semi at precut_fwd metres ahead of ego every frame.
        if elapsed <= cut_start:
            if actor is not None and actor.is_alive and self._ego is not None and self._ego.is_alive:
                ego_tf = self._ego.get_transform()
                ego_loc = ego_tf.location
                ego_fwd_v = ego_tf.get_forward_vector()
                ego_right = ego_tf.get_right_vector()
                ego_vel = self._ego.get_velocity()
                ego_speed = (ego_vel.x ** 2 + ego_vel.y ** 2) ** 0.5

                if not self._state.get("precut_lat_captured", False):
                    actor_loc = actor.get_location()
                    rel_x = actor_loc.x - ego_loc.x
                    rel_y = actor_loc.y - ego_loc.y
                    precut_lat = rel_x * ego_right.x + rel_y * ego_right.y
                    self._state["precut_lat"] = precut_lat
                    self._state["precut_lat_captured"] = True
                    print(f"[SemiPrecut] spawn lat={precut_lat:.2f}m", flush=True)

                precut_lat = self._state["precut_lat"]
                precut_fwd = 3.0  # keep semi 3 m ahead of ego

                semi_loc = carla.Location(
                    x=ego_loc.x + ego_fwd_v.x * precut_fwd + ego_right.x * precut_lat,
                    y=ego_loc.y + ego_fwd_v.y * precut_fwd + ego_right.y * precut_lat,
                    z=actor.get_location().z,
                )
                actor.set_transform(carla.Transform(semi_loc, carla.Rotation(yaw=ego_tf.rotation.yaw)))
                semi_speed = max(ego_speed + 1.5, 3.0)
                actor.set_target_velocity(carla.Vector3D(
                    ego_fwd_v.x * semi_speed,
                    ego_fwd_v.y * semi_speed,
                    0.0,
                ))
                self._lead["loc"] = semi_loc
                self._lead["yaw"] = ego_tf.rotation.yaw
                self._lead["forward"] = ego_fwd_v
                self._lead["right"] = ego_right
            return

        if cut_start < elapsed <= cut_end:
            # V12b SMOOTH DRIFT: one-time capture of actual semi position at cut_start,
            # then linearly interpolate toward ego's lane — no teleport, correct directions.
            if self._ego is not None and self._ego.is_alive:
                ego_tf = self._ego.get_transform()
                ego_loc = ego_tf.location
                ego_forward = ego_tf.get_forward_vector()
                ego_right = ego_tf.get_right_vector()

                # --- One-time initialisation at the first frame of cut phase ---
                if not self._state.get("drift_start_captured", False):
                    actor_loc = actor.get_location()
                    rel_x = actor_loc.x - ego_loc.x
                    rel_y = actor_loc.y - ego_loc.y
                    # Project actual semi position into ego's local frame.
                    # intrusion_sign=-1 when semi is in right lane (ego is to semi's left).
                    # init_lat > 0 means semi is to ego's RIGHT (expected ~2–3 m).
                    init_lat = rel_x * ego_right.x + rel_y * ego_right.y
                    init_fwd = rel_x * ego_forward.x + rel_y * ego_forward.y
                    self._state["drift_start_lateral"] = init_lat
                    self._state["drift_start_fwd"] = init_fwd
                    self._state["drift_start_captured"] = True
                    print(
                        f"[SemiDrift] cut_start captured: lat={init_lat:.2f}m "
                        f"fwd={init_fwd:.2f}m sign={intrusion_sign}",
                        flush=True,
                    )

                progress = min(1.0, (elapsed - cut_start) / max(0.1, self._semi_cut_duration_s))

                initial_lateral = self._state["drift_start_lateral"]  # e.g. +2.7 m (right of ego)
                initial_fwd    = self._state["drift_start_fwd"]       # e.g. -2 m (behind ego)

                # Target: semi intrudes 1.5 m into ego's lane.
                # For intrusion_sign=-1 (semi on right): target = -(-1)*1.5 = +1.5 m to ego's right.
                # Semi half-width ≈1.25 m → left edge at 1.5-1.25=0.25 m inside ego's lane. ✓
                target_lateral = -intrusion_sign * 1.5

                current_lateral = initial_lateral + progress * (target_lateral - initial_lateral)

                # Keep forward offset fixed relative to ego (no snap forward/backward).
                semi_loc = carla.Location(
                    x=ego_loc.x + ego_forward.x * initial_fwd + ego_right.x * current_lateral,
                    y=ego_loc.y + ego_forward.y * initial_fwd + ego_right.y * current_lateral,
                    z=actor.get_location().z,
                )

                # Semi leans its nose into ego's lane progressively.
                # intrusion_sign=-1 → yaw_offset decreases → semi nose turns left. ✓
                yaw_offset = intrusion_sign * self._semi_yaw_deg * progress
                semi_yaw = ego_tf.rotation.yaw + yaw_offset

                actor.set_transform(carla.Transform(semi_loc, carla.Rotation(yaw=semi_yaw)))

                # Lateral velocity component directed toward ego.
                # intrusion_sign=-1 → lateral_speed negative → leftward toward ego. ✓
                forward_speed = self._semi_intrusion_speed_mps
                lateral_speed = intrusion_sign * self._semi_lateral_speed_mps
                actor.set_target_velocity(carla.Vector3D(
                    ego_forward.x * forward_speed + ego_right.x * lateral_speed,
                    ego_forward.y * forward_speed + ego_right.y * lateral_speed,
                    0.0,
                ))

                # Gradual steering toward ego (negative = left turn when intrusion_sign=-1). ✓
                steer = intrusion_sign * 0.15 * progress
                actor.apply_control(carla.VehicleControl(
                    throttle=0.35,
                    brake=0.0,
                    steer=steer,
                ))

                self._lead["loc"] = semi_loc
                self._lead["yaw"] = semi_yaw
                self._lead["forward"] = ego_forward
                self._lead["right"] = ego_right
            return

        if cut_end < elapsed <= settle_end:
            ratio = min(1.0, (elapsed - cut_end) / max(0.1, self._semi_settle_duration_s))
            lateral_speed = 0.0
            yaw = intrusion_sign * (self._semi_yaw_deg - 8.0 * ratio)
            forward_speed = self._semi_settle_speed_mps

        self._advance_pose(self._lead, dt, forward_speed, lateral_speed)
        self._set_pose(self._lead, yaw, forward_speed, lateral_speed)

    def _step_crash_scene_pedestrian_escape(self, elapsed, dt):
        self._set_pose(self._hazard, 20.0, 0.0, 0.0)
        if self._state.get("crash_white") is not None:
            self._set_pose(self._state["crash_white"], -8.0, 0.0, 0.0)
        if self._support is not None:
            support_speed = max(18.0, _speed_mps(self._ego))
            self._advance_pose(self._support, dt, support_speed, 0.0)
            self._set_pose(self._support, 0.0, support_speed, 0.0)
        if self._walker is None or not self._walker.is_alive:
            return
        walker_loc = self._walker.get_location()
        wp = self._map.get_waypoint(self._trigger_wp.transform.location)
        forward = wp.transform.get_forward_vector()
        right = wp.transform.get_right_vector()
        self._walker.set_location(carla.Location(
            x=walker_loc.x + forward.x * 3.0 * dt - right.x * 1.4 * dt,
            y=walker_loc.y + forward.y * 3.0 * dt - right.y * 1.4 * dt,
            z=walker_loc.z,
        ))

    def _step_smoke_cloud_crash_scene(self, elapsed, dt):
        transform = self._state.get("smoke_overturned_transform")
        actor = self._hazard["actor"] if self._hazard is not None else None
        if transform is not None and actor is not None and actor.is_alive:
            actor.set_transform(transform)
            actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        if self._support is not None:
            self._set_pose(self._support, -16.0, 0.0, 0.0)

    def _step_aggressive_merge(self, elapsed, dt):
        if self._lead is None or self._lead["actor"] is None:
            return
        actor = self._lead["actor"]
        if not actor.is_alive or self._ego is None or not self._ego.is_alive:
            return

        ego_tf = self._ego.get_transform()
        ego_loc = ego_tf.location
        ego_forward = ego_tf.get_forward_vector()
        ego_right = ego_tf.get_right_vector()
        intrusion_sign = self._state.get("merge_intrusion_sign", -1.0)
        cut_start = max(0.0, self._merge_cut_start_s)
        cut_duration = max(0.1, self._merge_cut_duration_s)
        cut_end = cut_start + cut_duration

        if self._state.get("merge_collision_seen", False):
            initial_lateral = self._state.get("merge_precut_lateral", 0.0)
            away_sign = 1.0 if initial_lateral >= 0.0 else -1.0
            escape_speed = max(_speed_mps(self._ego), 8.0)
            actor.set_target_velocity(carla.Vector3D(
                ego_forward.x * escape_speed + ego_right.x * away_sign * 5.0,
                ego_forward.y * escape_speed + ego_right.y * away_sign * 5.0,
                0.0,
            ))
            actor.apply_control(carla.VehicleControl(
                throttle=0.35, brake=0.0, steer=0.0,
            ))
            self._state["merge_physics_restored"] = True
            return

        if self._state.get("merge_released", False):
            return
        if self._state.get("merge_physics_restored", False):
            return

        if elapsed <= cut_start:
            precut_lateral = self._state.get("merge_precut_lateral")
            if precut_lateral is None:
                actor_loc = actor.get_location()
                precut_lateral = (
                    (actor_loc.x - ego_loc.x) * ego_right.x
                    + (actor_loc.y - ego_loc.y) * ego_right.y
                )
                self._state["merge_precut_lateral"] = precut_lateral
                print(
                    f"[ADASuite] right merge precut lateral={precut_lateral:.2f}m",
                    flush=True,
                )
            merge_loc = carla.Location(
                x=ego_loc.x + ego_forward.x * 4.0 + ego_right.x * precut_lateral,
                y=ego_loc.y + ego_forward.y * 4.0 + ego_right.y * precut_lateral,
                z=actor.get_location().z,
            )
            actor.set_transform(carla.Transform(
                merge_loc, carla.Rotation(yaw=ego_tf.rotation.yaw),
            ))
            actor.set_target_velocity(carla.Vector3D(
                ego_forward.x * 3.5,
                ego_forward.y * 3.5,
                0.0,
            ))
            return

        if elapsed >= cut_end + 0.10:
            actor.apply_control(carla.VehicleControl(
                throttle=0.0, brake=0.55, steer=0.0,
            ))
            self._state["merge_released"] = True
            print("[ADASuite] right merge actor left in physics after impact", flush=True)
            return

        progress = min(1.0, (elapsed - cut_start) / cut_duration)
        initial_lateral = self._state.get("merge_precut_lateral", 3.0)
        current_lateral = initial_lateral * (1.0 - progress)
        forward_gap = 4.0 - 4.5 * progress
        merge_loc = carla.Location(
            x=ego_loc.x + ego_forward.x * forward_gap + ego_right.x * current_lateral,
            y=ego_loc.y + ego_forward.y * forward_gap + ego_right.y * current_lateral,
            z=actor.get_location().z,
        )
        actor.set_transform(carla.Transform(
            merge_loc,
            carla.Rotation(yaw=ego_tf.rotation.yaw + intrusion_sign * 24.0 * progress),
        ))
        if progress >= self._merge_physics_restore_progress and not self._state.get("merge_physics_restored", False):
            try:
                actor.set_collisions(True)
            except (AttributeError, RuntimeError):
                pass
            try:
                actor.set_simulate_physics(True)
            except RuntimeError:
                pass
            actor.apply_control(carla.VehicleControl(
                throttle=0.30, brake=0.0, steer=0.28 * intrusion_sign,
            ))
            self._state["merge_physics_restored"] = True
            return
        actor.set_target_velocity(carla.Vector3D(
            ego_forward.x * 3.5 + ego_right.x * 4.0 * intrusion_sign,
            ego_forward.y * 3.5 + ego_right.y * 4.0 * intrusion_sign,
            0.0,
        ))
        actor.apply_control(carla.VehicleControl(
            throttle=0.30, brake=0.0, steer=0.28 * intrusion_sign,
        ))
        self._lead["loc"] = merge_loc
        self._lead["yaw"] = ego_tf.rotation.yaw + intrusion_sign * 24.0 * progress
        self._lead["forward"] = ego_forward
        self._lead["right"] = ego_right

    def _step_animal_cross(self, elapsed, dt):
        if self._support is not None:
            support_speed = max(16.0, _speed_mps(self._ego))
            self._advance_pose(self._support, dt, support_speed, 0.0)
            self._set_pose(self._support, 0.0, support_speed, 0.0)
        if self._walker is None or not self._walker.is_alive:
            return
        if not self._animal_track_ego and _speed_mps(self._ego) < 12.5:
            return
        forward = self._trigger_wp.transform.get_forward_vector()
        ego_loc = self._ego.get_location()
        walker_loc = self._walker.get_location()
        ego_gap = ((walker_loc.x - ego_loc.x) * forward.x +
                    (walker_loc.y - ego_loc.y) * forward.y)
        if not self._state.get("animal_cross_started", False):
            if "animal_cross_triggered_at" not in self._state:
                if self._animal_cross_start_delay_s <= 0.0 and ego_gap > 26.0:
                    return
                self._state["animal_cross_triggered_at"] = elapsed
            if elapsed - self._state["animal_cross_triggered_at"] < self._animal_cross_start_delay_s:
                return
            self._state["animal_cross_started"] = True
            self._state["animal_cross_start_loc"] = carla.Location(
                x=walker_loc.x, y=walker_loc.y, z=walker_loc.z)
        wp = self._map.get_waypoint(self._trigger_wp.transform.location)
        direction = wp.transform.get_right_vector()
        cross_distance = min(
            self._animal_cross_distance_limit,
            self._state.get("animal_cross_distance_m", 0.0)
            + self._animal_cross_speed * dt,
        )
        self._state["animal_cross_distance_m"] = cross_distance
        start_loc = self._state.get(
            "animal_cross_start_loc",
            self._state.get("animal_start_loc", walker_loc),
        )
        if self._animal_track_ego:
            ego_loc = self._ego.get_location()
            longitudinal = carla.Location(
                x=ego_loc.x + forward.x * self._animal_intercept_lead,
                y=ego_loc.y + forward.y * self._animal_intercept_lead,
                z=start_loc.z,
            )
        else:
            longitudinal = start_loc
        if self._animal_track_ego:
            lateral_offset = max(0.0, self._animal_start_lateral - cross_distance)
            target_x = longitudinal.x + direction.x * lateral_offset
            target_y = longitudinal.y + direction.y * lateral_offset
        else:
            target_x = start_loc.x - direction.x * cross_distance
            target_y = start_loc.y - direction.y * cross_distance
        self._walker.set_location(carla.Location(
            x=target_x,
            y=target_y,
            z=longitudinal.z,
        ))

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateHighwayADASObstacleSuite",
        )

    def _launch_semi(self):
        if self._lead is None:
            return
        actor = self._lead.get("actor")
        if actor is None or not actor.is_alive:
            return
        forward = actor.get_transform().get_forward_vector()
        speed = self._semi_initial_speed_mps
        actor.set_target_velocity(carla.Vector3D(forward.x * speed, forward.y * speed, 0.0))

    def _start_semi_kinematic_intrusion(self):
        if self._lead is None:
            return
        actor = self._lead.get("actor")
        if actor is not None and actor.is_alive:
            self._lead["loc"] = actor.get_location()
            self._lead["yaw"] = actor.get_transform().rotation.yaw
            actor.set_simulate_physics(False)
        self._state["semi_lanechange_active"] = True

    def _create_semi_intrusion_behavior(self):
        from agents.navigation.local_planner import RoadOption

        semi_actor = self._lead["actor"] if self._lead is not None else None
        plan = [(wp, RoadOption.LANEFOLLOW) for wp in self._semi_plan]
        intrusion_sign = self._state.get("intrusion_sign", 1.0)
        P = py_trees.common.ParallelPolicy

        root = py_trees.composites.Sequence("SemiIntrusion_Root")
        phase0 = py_trees.composites.Parallel("Phase0_Sync", policy=P.SUCCESS_ON_ONE)
        phase0.add_child(TimeOut(8.0, name="Phase0_Timeout"))
        phase0.add_child(HoldUntilEgoMoves(semi_actor, self._ego, name="HoldSemi"))
        root.add_child(phase0)
        root.add_child(_OneShot(self._launch_semi, name="LaunchSemi"))

        phase1 = py_trees.composites.Parallel("Phase1_CruiseAlongside", policy=P.SUCCESS_ON_ONE)
        phase1.add_child(WaypointFollower(
            semi_actor,
            target_speed=max(0.1, self._semi_initial_speed_mps),
            plan=plan,
            name="SemiLaneFollow",
        ))
        phase1.add_child(_WaitUntilAlongside(
            self._ego,
            semi_actor,
            max_forward_m=self._semi_alongside_threshold_m,
            name="SemiAlongsideTrigger",
        ))
        phase1.add_child(TimeOut(max(7.0, self._semi_cut_start_s + 3.0), name="Phase1_MaxTimeout"))
        root.add_child(phase1)

        phase2 = py_trees.composites.Sequence("Phase2_SemiIntrusion")
        phase2.add_child(_ForcedLateralIntrusion(
            semi_actor,
            forward_speed_mps=self._semi_intrusion_speed_mps,
            lateral_speed_mps=self._semi_lateral_speed_mps,
            duration=self._semi_cut_duration_s,
            lateral_sign=intrusion_sign,
            steer=0.40,
            name="SemiIntrusion",
        ))
        root.add_child(phase2)

        root.add_child(_ForcedLateralIntrusion(
            semi_actor,
            forward_speed_mps=self._semi_settle_speed_mps,
            lateral_speed_mps=0.0,
            duration=self._semi_aftermath_duration_s,
            lateral_sign=intrusion_sign,
            steer=0.0,
            name="SemiAftermathRoll",
        ))

        cleanup = py_trees.composites.Sequence("Cleanup")
        for index, actor in enumerate(self._actors + self._extra_props):
            cleanup.add_child(ActorDestroy(actor, name=f"Destroy{index}"))
        root.add_child(cleanup)
        return root

    def _create_behavior(self):
        root = py_trees.composites.Sequence("ADASuite_Root")
        root.add_child(_RunADASVariant(self))
        root.add_child(TimeOut(1.8, name="Aftermath"))

        cleanup = py_trees.composites.Sequence("Cleanup")
        for index, actor in enumerate(self._actors + self._extra_props):
            cleanup.add_child(ActorDestroy(actor, name=f"Destroy{index}"))
        if self._walker is not None:
            cleanup.add_child(ActorDestroy(self._walker, name="DestroyWalker"))
        root.add_child(cleanup)

        lane_waypoints = [self._trigger_wp]
        if self._left_lane_wp is not None:
            lane_waypoints.append(self._left_lane_wp)
        if self._right_lane_wp is not None:
            lane_waypoints.append(self._right_lane_wp)

        outer = py_trees.composites.Parallel(
            "ADASuite_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego] + self._actors,
            lane_waypoints=lane_waypoints,
            center_location=self._trigger_wp.transform.location,
            radius=450.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        outer.add_child(PeriodicStatusLogger(
            self._ego,
            self._actors[0] if self._actors else None,
            self._actors[1] if len(self._actors) > 1 else None,
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
        if self._ego_min_speed_kmh > 0.1:
            outer.add_child(_EgoMinSpeedForcer(
                self._ego,
                min_speed_kmh=self._ego_min_speed_kmh,
                throttle=self._ego_force_throttle,
                name="EgoMinSpeedForcer",
            ))
        if (self._variant == "night_shoulder_truck" and
                self._shoulder_truck_ego_hold_s > 0.0):
            outer.add_child(_EgoTimedThrottleForcer(
                self._ego,
                start_after_s=self._shoulder_truck_reveal_after_s,
                duration_s=self._shoulder_truck_ego_hold_s,
                throttle=self._shoulder_truck_ego_hold_throttle,
                name="ShoulderTruckEgoThrottleHold",
            ))
        if (self._variant == "semi_intrusion_clear_escape" or
                (self._variant == "animal_cross" and
                 (self._animal_stop_on_collision or
                  self._animal_disable_collision_on_hit or
                  self._animal_stabilize_ego_on_collision))) and self._actors:
            outer.add_child(StopActorsOnEgoCollision(
                self._ego,
                actors_to_stop=[self._ego] + self._actors,
                name="StopOnCollision",
                disable_other_collision=self._animal_disable_collision_on_hit,
                stop_ego_on_collision=self._animal_stop_on_collision,
                stabilize_ego_on_collision=self._animal_stabilize_ego_on_collision,
            ))
        return outer

    def _create_test_criteria(self):
        return []

    def __del__(self):
        self.remove_all_actors()
