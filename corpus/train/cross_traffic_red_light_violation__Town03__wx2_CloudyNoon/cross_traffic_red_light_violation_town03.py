"""Cross traffic red light violation (Town03).

A vehicle runs a red light and crosses the ego's path at the junction.
"""

import math

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
from srunner.scenariomanager.scenarioatomics.custom_atomics_cross_traffic_red_light_violation_town03 import (
    EgoSpeedGovernor,
    HoldUntilEgoMoves,
    LaneCleaner,
)
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import HandleJunctionScenario
from srunner.tools.scenario_helper import (
    filter_junction_wp_direction,
    generate_target_waypoint,
    get_closest_traffic_light,
    get_geometric_linear_intersection,
    get_junction_topology,
)


def _speed_mps(actor):
    if actor is None or not actor.is_alive:
        return 0.0
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _planar_distance(loc_a, loc_b):
    return math.hypot(loc_a.x - loc_b.x, loc_a.y - loc_b.y)


def _yaw_delta_deg(a_deg, b_deg):
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def _ground_vehicle_to_road(world_map, actor, heading_dx=None, heading_dy=None, z_offset=0.08):
    if actor is None or not actor.is_alive:
        return
    transform = actor.get_transform()
    waypoint = world_map.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return
    yaw = transform.rotation.yaw
    if heading_dx is not None and heading_dy is not None and math.hypot(heading_dx, heading_dy) > 0.1:
        yaw = math.degrees(math.atan2(heading_dy, heading_dx))
    actor.set_transform(carla.Transform(
        carla.Location(transform.location.x, transform.location.y, waypoint.transform.location.z + z_offset),
        carla.Rotation(yaw=yaw, pitch=0.0, roll=0.0),
    ))


class _RunRedLightNearMiss(py_trees.behaviour.Behaviour):
    def __init__(self, scenario, name="RunRedLightNearMiss"):
        super().__init__(name)
        self._scenario = scenario
        self._start_time = None
        self._attack_started = False
        self._hesitation_start = None
        self._last_log = -999.0
        self._runner_reached_conflict = False

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        scenario = self._scenario
        if scenario._impact_detected:
            print("[RedLightLeft] *** IMPACT DETECTED ***", flush=True)
            return py_trees.common.Status.SUCCESS
        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0
        ego_loc = scenario._ego.get_location()
        dist_to_conflict = _planar_distance(ego_loc, scenario._collision_location)
        ego_speed = _speed_mps(scenario._ego)
        ttc = float("inf") if ego_speed < 0.5 else dist_to_conflict / ego_speed

        if (not self._attack_started and
                (dist_to_conflict <= scenario._trigger_distance or ttc <= scenario._sync_time)):
            self._attack_started = True
            self._hesitation_start = elapsed
            print(
                f"[RedLightLeft] Ego reached conflict approach "
                f"(dist={dist_to_conflict:.1f}m, ttc={ttc:.2f}s) — violator GO, ego hesitate.",
                flush=True,
            )

        runner = scenario._runner
        if runner is not None and runner.is_alive:
            if self._attack_started:
                runner_loc = runner.get_location()
                ego_tf = scenario._ego.get_transform()
                ego_forward = ego_tf.get_forward_vector()
                target_location = scenario._sink_location
                if scenario._runner_intercept_ahead_m <= 0.0:
                    target_location = scenario._collision_location
                elif _planar_distance(runner_loc, ego_loc) <= scenario._runner_intercept_distance:
                    ego_right = ego_tf.get_right_vector()
                    target_location = carla.Location(
                        x=ego_loc.x + ego_forward.x * scenario._runner_intercept_ahead_m,
                        y=ego_loc.y + ego_forward.y * scenario._runner_intercept_ahead_m,
                        z=scenario._sink_location.z,
                    )
                    target_location.x -= ego_right.x * scenario._runner_intercept_left_m
                    target_location.y -= ego_right.y * scenario._runner_intercept_left_m
                dx = target_location.x - runner_loc.x
                dy = target_location.y - runner_loc.y
                magnitude = math.hypot(dx, dy)
                if magnitude > 0.8:
                    if scenario._ground_runner_during_attack:
                        _ground_vehicle_to_road(
                            scenario._map, runner, heading_dx=dx, heading_dy=dy,
                            z_offset=scenario._runner_ground_z_offset,
                        )
                    runner.set_target_velocity(carla.Vector3D(
                        scenario._runner_speed * dx / magnitude,
                        scenario._runner_speed * dy / magnitude,
                        0.0,
                    ))
                    runner.apply_control(carla.VehicleControl(throttle=scenario._runner_throttle, brake=0.0, steer=0.0))
                else:
                    runner.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            else:
                runner.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                runner.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))

        if self._hesitation_start is not None:
            pause_elapsed = elapsed - self._hesitation_start
            if pause_elapsed <= scenario._hesitation_duration:
                control = scenario._ego.get_control()
                control.throttle = 0.0
                control.brake = max(control.brake, scenario._hesitation_brake)
                scenario._ego.apply_control(control)

        if elapsed - self._last_log >= 1.0:
            self._last_log = elapsed
            runner_spd = _speed_mps(runner) * 3.6 if runner else 0.0
            print(
                f"[RedLightLeft] t={elapsed:.1f}s ego={ego_speed * 3.6:.0f}km/h "
                f"runner={runner_spd:.0f}km/h dist={dist_to_conflict:.1f}m "
                f"attack={self._attack_started}",
                flush=True,
            )

        if self._attack_started and runner is not None and runner.is_alive:
            runner_distance = _planar_distance(runner.get_location(), scenario._collision_location)
            if runner_distance <= scenario._runner_clear_distance:
                self._runner_reached_conflict = True
            if (self._runner_reached_conflict and runner_distance >= scenario._runner_clear_distance):
                if self._hesitation_start is not None and elapsed > self._hesitation_start + scenario._hesitation_duration + 1.0:
                    return py_trees.common.Status.SUCCESS

        if elapsed >= scenario._max_run_time:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    def __init__(self, scenario, name="EgoMinSpeedForcer"):
        super().__init__(name)
        self._scenario = scenario

    def update(self):
        scenario = self._scenario
        ego = scenario._ego
        if ego is None or not ego.is_alive:
            return py_trees.common.Status.RUNNING
        if _speed_mps(ego) < scenario._ego_min_speed_mps:
            control = ego.get_control()
            control.brake = 0.0
            control.throttle = scenario._ego_force_throttle
            ego.apply_control(control)
        return py_trees.common.Status.RUNNING


class _ClearIntersectionTraffic(py_trees.behaviour.Behaviour):
    def __init__(self, scenario, name="ClearIntersectionTraffic"):
        super().__init__(name)
        self._scenario = scenario
        self._last_clean = -999.0

    def update(self):
        scenario = self._scenario
        now = GameTime.get_time()
        if now - self._last_clean < 0.25:
            return py_trees.common.Status.RUNNING
        self._last_clean = now
        protected_ids = {scenario._ego.id}
        if scenario._runner is not None:
            protected_ids.add(scenario._runner.id)
        center = scenario._collision_location
        if center is None:
            return py_trees.common.Status.RUNNING
        for actor in scenario._world.get_actors().filter("vehicle.*"):
            if actor.id in protected_ids:
                continue
            if _planar_distance(actor.get_location(), center) <= scenario._clear_intersection_radius:
                try:
                    actor_type = actor.type_id
                    actor.destroy()
                    print(f"[RedLightLeft] removed intersection NPC {actor_type}", flush=True)
                except RuntimeError:
                    pass
        return py_trees.common.Status.RUNNING


class _EgoLeftTurnForcer(py_trees.behaviour.Behaviour):
    def __init__(self, scenario, name="EgoLeftTurnForcer"):
        super().__init__(name)
        self._scenario = scenario
        self._active_since = None
        self._initial_yaw = None
        self._last_log = -999.0

    def update(self):
        scenario = self._scenario
        ego = scenario._ego
        if ego is None or not ego.is_alive:
            return py_trees.common.Status.RUNNING
        if scenario._forced_turn_start_distance <= 0.0:
            return py_trees.common.Status.RUNNING

        now = GameTime.get_time()
        ego_tf = ego.get_transform()
        dist_to_conflict = _planar_distance(ego_tf.location, scenario._collision_location)
        if self._initial_yaw is None:
            self._initial_yaw = ego_tf.rotation.yaw
        turn_delta = _yaw_delta_deg(ego_tf.rotation.yaw, self._initial_yaw)

        if self._active_since is None and dist_to_conflict <= scenario._forced_turn_start_distance:
            self._active_since = now
            print(
                f"[RedLightLeft] Ego forced LEFT turn start "
                f"(dist={dist_to_conflict:.1f}m yaw_delta={turn_delta:.1f})",
                flush=True,
            )

        active = (
            self._active_since is not None and
            now - self._active_since <= scenario._forced_turn_duration and
            turn_delta < scenario._forced_turn_end_yaw_deg
        )
        if active:
            ego.apply_control(carla.VehicleControl(
                throttle=scenario._forced_turn_throttle,
                steer=scenario._forced_turn_steer,
                brake=0.0,
                hand_brake=False,
                manual_gear_shift=False,
            ))
            if now - self._last_log >= 0.5:
                self._last_log = now
                print(
                    f"[RedLightLeft] forcing left: dist={dist_to_conflict:.1f}m "
                    f"yaw_delta={turn_delta:.1f} steer={scenario._forced_turn_steer:.2f}",
                    flush=True,
                )
        return py_trees.common.Status.RUNNING


class CrossTrafficRedLightViolationTown03(BasicScenario):
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
        self._activation_distance = self._get_param(op, "activation_distance_m", 70.0)
        self._direction = str(self._get_param(op, "direction", "right", cast=str)).lower()
        self._source_distance = self._get_param(op, "source_distance_m", 20.0)
        self._sink_distance = self._get_param(op, "sink_distance_m", 16.0)
        self._runner_speed = self._get_param(op, "runner_speed_kmh", 56.0) / 3.6
        self._trigger_distance = self._get_param(op, "trigger_distance_m", 13.0)
        self._sync_time = self._get_param(op, "sync_time_s", 2.2)
        self._runner_model = str(self._get_param(op, "runner_model", "vehicle.audi.tt", cast=str))
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 14.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.20)
        self._ego_min_speed_mps = self._get_param(op, "ego_min_speed_kmh", 0.0) / 3.6
        self._ego_force_throttle = self._get_param(op, "ego_force_throttle", 0.0)
        self._forced_turn_start_distance = self._get_param(op, "forced_turn_start_distance_m", 0.0)
        self._forced_turn_steer = self._get_param(op, "forced_turn_steer", -0.75)
        self._forced_turn_throttle = self._get_param(op, "forced_turn_throttle", 0.42)
        self._forced_turn_duration = self._get_param(op, "forced_turn_duration_s", 3.0)
        self._forced_turn_end_yaw_deg = self._get_param(op, "forced_turn_end_yaw_deg", 75.0)
        self._hesitation_duration = self._get_param(op, "hesitation_duration_s", 1.8)
        self._hesitation_brake = self._get_param(op, "hesitation_brake", 0.55)
        self._runner_clear_distance = self._get_param(op, "runner_clear_distance_m", 10.0)
        self._runner_intercept_distance = self._get_param(op, "runner_intercept_distance_m", 28.0)
        self._runner_intercept_ahead_m = self._get_param(op, "runner_intercept_ahead_m", 2.5)
        self._runner_intercept_left_m = self._get_param(op, "runner_intercept_left_m", 0.0)
        self._runner_throttle = self._get_param(op, "runner_throttle", 0.65)
        self._ground_runner_during_attack = bool(self._get_param(op, "ground_runner_during_attack", 1, cast=int))
        self._runner_ground_z_offset = self._get_param(op, "runner_ground_z_offset_m", 0.08)
        self._block_distance = self._get_param(op, "block_distance_m", 3.5)
        self._clear_intersection_radius = self._get_param(op, "clear_intersection_radius_m", 35.0)
        self._max_run_time = self._get_param(op, "max_run_time", 18.0)
        self._post_event_hold = self._get_param(op, "post_event_hold_s", 2.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._junction = None
        self._spawn_wp = None
        self._sink_wp = None
        self._spawn_location = None
        self._sink_location = None
        self._collision_location = None
        self._runner = None
        self._collision_sensor = None
        self._impact_detected = False
        self._tl_dict = {}

        super().__init__(
            name="CrossTrafficRedLightViolationTown03",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            terminate_on_failure=True,
            criteria_enable=criteria_enable,
        )

    def _initialize_actors(self, config):
        ego_location = config.trigger_points[0].location
        ego_wp = self._map.get_waypoint(ego_location)
        starting_wp = ego_wp
        while not starting_wp.is_junction:
            next_wps = starting_wp.next(1.0)
            if not next_wps:
                raise ValueError("[RedLightLeft] Failed to find a junction ahead of ego")
            starting_wp = next_wps[0]
        self._junction = starting_wp.get_junction()

        entry_wps, _ = get_junction_topology(self._junction)
        source_entry_wps = filter_junction_wp_direction(starting_wp, entry_wps, self._direction)
        if not source_entry_wps:
            raise ValueError(f"[RedLightLeft] No cross-traffic entry for direction={self._direction}")

        ego_forward_yaw = math.radians(starting_wp.transform.rotation.yaw)
        ego_forward = (math.cos(ego_forward_yaw), math.sin(ego_forward_yaw))

        spawn_wp = source_entry_wps[0]
        collision_location = None
        best_progress = float("-inf")
        for candidate_wp in source_entry_wps:
            candidate_collision = get_geometric_linear_intersection(
                starting_wp.transform.location,
                candidate_wp.transform.location,
                True,
            )
            if candidate_collision is None:
                continue
            delta_x = candidate_collision.x - self._trigger_wp.transform.location.x
            delta_y = candidate_collision.y - self._trigger_wp.transform.location.y
            progress_along_ego = delta_x * ego_forward[0] + delta_y * ego_forward[1]
            if progress_along_ego > best_progress:
                best_progress = progress_along_ego
                spawn_wp = candidate_wp
                collision_location = candidate_collision

        staged_distance = 0.0
        while staged_distance < self._source_distance:
            prev_wps = spawn_wp.previous(1.0)
            if not prev_wps or prev_wps[0].is_junction:
                break
            spawn_wp = prev_wps[0]
            staged_distance += 1.0
        self._spawn_wp = spawn_wp
        self._spawn_location = carla.Transform(
            spawn_wp.transform.location + carla.Location(z=0.4),
            spawn_wp.transform.rotation,
        )

        sink_exit_wp = generate_target_waypoint(self._map.get_waypoint(spawn_wp.transform.location), 0)
        sink_wps = sink_exit_wp.next(self._sink_distance)
        if not sink_wps:
            raise ValueError("[RedLightLeft] Failed to compute runner sink waypoint")
        self._sink_wp = sink_wps[0]
        self._sink_location = self._sink_wp.transform.location

        self._collision_location = collision_location
        if self._collision_location is None:
            raise ValueError("[RedLightLeft] Failed to compute conflict point")
        collision_wp = self._map.get_waypoint(self._collision_location)
        self._collision_location.z = collision_wp.transform.location.z

        traffic_lights = self._world.get_traffic_lights_in_junction(self._junction.id)
        if traffic_lights:
            ego_tl = get_closest_traffic_light(ego_wp, traffic_lights)
            for traffic_light in traffic_lights:
                self._tl_dict[traffic_light] = (
                    carla.TrafficLightState.Green if traffic_light == ego_tl else carla.TrafficLightState.Red
                )

        models = [self._runner_model, "vehicle.audi.tt", "vehicle.mercedes.coupe_2020", "vehicle.tesla.model3"]
        for model in dict.fromkeys(models):
            self._runner = CarlaDataProvider.request_new_actor(
                model, self._spawn_location, rolename="red_light_left_runner",
            )
            if self._runner is not None:
                break
        if self._runner is None:
            raise RuntimeError("[RedLightLeft] Runner spawn failed")

        self._runner.set_simulate_physics(True)
        self._runner.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self._runner.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        self.other_actors.append(self._runner)

        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            bp, carla.Transform(), attach_to=self._ego)
        self._collision_sensor.listen(lambda event: self._on_collision(event))

        print(
            f"\n[RedLightLeft] Spawn\n"
            f"  Runner at ({self._spawn_location.location.x:.1f}, {self._spawn_location.location.y:.1f}) model={self._runner_model}\n"
            f"  Conflict=({self._collision_location.x:.1f}, {self._collision_location.y:.1f}) sink=({self._sink_location.x:.1f}, {self._sink_location.y:.1f})\n"
            f"  runner_speed={self._runner_speed * 3.6:.0f}km/h hesitation={self._hesitation_duration:.1f}s\n",
            flush=True,
        )

    def _on_collision(self, event):
        other = event.other_actor
        is_runner = self._runner is not None and other.id == self._runner.id
        is_vehicle_conflict = (
            other.type_id.startswith("vehicle.") and
            _planar_distance(self._ego.get_location(), self._collision_location) <= self._block_distance + 4.0
        )
        if not is_runner:
            print(f"[RedLightLeft] Ignoring non-runner collision: {other.type_id}", flush=True)
            return
        if not self._impact_detected:
            self._impact_detected = True
            print(f"[RedLightLeft] *** COLLISION: {other.type_id} ***", flush=True)

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateRedLightRunnerLeftTurn",
        )

    def _create_behavior(self):
        root = py_trees.composites.Sequence("RedLightLeftSequence")
        if self.route_mode and self._spawn_wp is not None and self._sink_wp is not None:
            root.add_child(HandleJunctionScenario(
                clear_junction=True,
                clear_ego_entry=True,
                remove_entries=[self._spawn_wp],
                remove_exits=[self._sink_wp],
                stop_entries=False,
                extend_road_exit=40,
            ))

        sync = py_trees.composites.Parallel(
            "Phase0_Sync",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
        )
        sync.add_child(HoldUntilEgoMoves(self._runner, self._ego))
        root.add_child(sync)

        main_loop = py_trees.composites.Parallel(
            "Phase1_MainLoop",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        main_loop.add_child(_RunRedLightNearMiss(self))
        main_loop.add_child(_ClearIntersectionTraffic(self))
        main_loop.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            max_brake=self._ego_cap_brake,
        ))
        if self._ego_min_speed_mps > 0.1 and self._ego_force_throttle > 0.0:
            main_loop.add_child(_EgoMinSpeedForcer(self))
        if self._forced_turn_start_distance > 0.0:
            main_loop.add_child(_EgoLeftTurnForcer(self))
        main_loop.add_child(TimeOut(self._max_run_time))
        root.add_child(main_loop)
        root.add_child(TimeOut(self._post_event_hold, name="PostEventHold"))
        root.add_child(ActorDestroy(self._runner, name="DestroyRunner"))

        ego_wp = self._map.get_waypoint(
            self._trigger_wp.transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        lane_wps = [ego_wp]
        left = ego_wp.get_left_lane()
        if left and left.lane_type == carla.LaneType.Driving:
            lane_wps.append(left)
        right = ego_wp.get_right_lane()
        if right and right.lane_type == carla.LaneType.Driving:
            lane_wps.append(right)

        outer = py_trees.composites.Parallel(
            "RedLightLeft_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        if self.route_mode:
            outer.add_child(CollisionTest(self._ego, name="RedLightLeftCollisionEnd"))
        if self._tl_dict:
            outer.add_child(TrafficLightFreezer(self._tl_dict))
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego, self._runner],
            lane_waypoints=lane_wps,
            center_location=self._trigger_wp.transform.location,
            radius=120.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        return outer

    def _create_test_criteria(self):
        criteria = py_trees.composites.Parallel(
            "RedLightLeftCriteria",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        criteria.add_child(CollisionTest(self._ego))
        return criteria

    def remove_all_actors(self):
        if self._collision_sensor is not None and self._collision_sensor.is_alive:
            try:
                self._collision_sensor.stop()
            except RuntimeError:
                pass
            try:
                self._collision_sensor.destroy()
            except RuntimeError:
                pass
            self._collision_sensor = None
        return super().remove_all_actors()
