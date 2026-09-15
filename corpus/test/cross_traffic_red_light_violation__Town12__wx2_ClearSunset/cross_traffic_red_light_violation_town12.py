"""Cross traffic red light violation (Town12).

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
from srunner.scenariomanager.scenarioatomics.custom_atomics_cross_traffic_red_light_violation_town12 import (
    CarlaRolloutLogger,
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


class _RunRedLightAttack(py_trees.behaviour.Behaviour):
    """Launch the violator through the junction once ego commits to the conflict."""

    def __init__(self, scenario, name="RunRedLightAttack"):
        super().__init__(name)
        self._scenario = scenario
        self._start_time = None
        self._attack_started = False
        self._last_log = -999.0

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        scenario = self._scenario
        if scenario._impact_detected:
            print("[RedLight] *** T-BONE COLLISION ***", flush=True)
            self._apply_post_impact_response()
            return py_trees.common.Status.SUCCESS

        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0

        ego_loc = scenario._ego.get_location()
        dist_to_collision = _planar_distance(ego_loc, scenario._collision_location)
        ego_speed = _speed_mps(scenario._ego)
        time_to_collision = float("inf") if ego_speed < 0.5 else dist_to_collision / ego_speed

        if (not self._attack_started and
                (dist_to_collision <= scenario._trigger_distance or
                 time_to_collision <= scenario._sync_time)):
            self._attack_started = True
            print(
                f"[RedLight] Ego committed to junction "
                f"(dist={dist_to_collision:.1f}m, ttc={time_to_collision:.2f}s) — runner GO!",
                flush=True,
            )

        runner = scenario._runner
        if runner is not None and runner.is_alive:
            if self._attack_started:
                runner_loc = runner.get_location()
                ego_transform = scenario._ego.get_transform()
                ego_forward = ego_transform.get_forward_vector()
                intercept_location = ego_loc + carla.Location(
                    x=ego_forward.x * 2.5,
                    y=ego_forward.y * 2.5,
                    z=0.0,
                )

                # Bias the runner toward the ego's through-lane so the crossing
                # remains a believable red-light run but still produces contact.
                target_location = scenario._sink_location
                if _planar_distance(runner_loc, ego_loc) <= 28.0:
                    target_location = carla.Location(
                        x=intercept_location.x,
                        y=intercept_location.y,
                        z=scenario._sink_location.z,
                    )

                dx = target_location.x - runner_loc.x
                dy = target_location.y - runner_loc.y
                magnitude = math.hypot(dx, dy)
                if magnitude > 0.8:
                    runner.set_target_velocity(carla.Vector3D(
                        scenario._runner_speed * dx / magnitude,
                        scenario._runner_speed * dy / magnitude,
                        0.0,
                    ))
                    runner.apply_control(carla.VehicleControl(
                        throttle=0.55, brake=0.0, steer=0.0,
                    ))
                else:
                    print("[RedLight] Runner cleared the conflict zone without impact", flush=True)
                    return py_trees.common.Status.FAILURE
            else:
                runner.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                runner.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=1.0, steer=0.0,
                ))

        if elapsed - self._last_log >= 1.0:
            self._last_log = elapsed
            ego_spd = _speed_mps(scenario._ego) * 3.6
            runner_spd = _speed_mps(runner) * 3.6 if runner else 0.0
            ttc_txt = "inf" if not math.isfinite(time_to_collision) else f"{time_to_collision:.2f}s"
            print(
                f"[RedLight] t={elapsed:.1f}s  ego={ego_spd:.0f}km/h  "
                f"runner={runner_spd:.0f}km/h  dist_col={dist_to_collision:.1f}m  "
                f"ttc={ttc_txt}  attack={self._attack_started}  impact={scenario._impact_detected}",
                flush=True,
            )

        if elapsed >= scenario._max_run_time:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING

    def _apply_post_impact_response(self):
        scenario = self._scenario
        runner = scenario._runner
        if runner is not None and runner.is_alive:
            runner.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            runner.apply_control(carla.VehicleControl(
                throttle=0.0, brake=1.0, steer=0.0, hand_brake=True,
            ))

        ego = scenario._ego
        if ego is None or not ego.is_alive:
            return

        control = ego.get_control()
        control.throttle = 0.0
        control.brake = max(control.brake, 0.6)
        ego.apply_control(control)

        if scenario._enable_flip and runner is not None and runner.is_alive:
            ego_velocity = ego.get_velocity()
            runner_velocity = runner.get_velocity()
            ego.set_target_velocity(carla.Vector3D(
                ego_velocity.x + runner_velocity.x * 0.25,
                ego_velocity.y + runner_velocity.y * 0.25,
                scenario._flip_launch_z,
            ))


class _PostImpactHold(py_trees.behaviour.Behaviour):
    """Keep the wreck scene alive for the capture window."""

    def __init__(self, duration, name="PostImpactHold"):
        super().__init__(name)
        self._duration = float(duration)
        self._start_time = None

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0
        if elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class _SideWindForce(py_trees.behaviour.Behaviour):
    """Apply a continuous lateral force to live scenario vehicles."""

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
        for actor in (scenario._ego, scenario._runner):
            if actor is not None and actor.is_alive and hasattr(actor, "add_force"):
                actor.add_force(force)

        now = GameTime.get_time()
        if now - self._last_log >= 2.0:
            self._last_log = now
            print(
                f"[RedLight] side_wind force={scenario._side_wind_force:.0f}N "
                f"yaw={scenario._side_wind_yaw_deg:.0f}deg",
                flush=True,
            )
        return py_trees.common.Status.RUNNING


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    """Prevent the autopilot from stopping completely before the junction."""

    def __init__(self, ego, min_speed_kmh=18.0, throttle=0.7, name="EgoMinSpeedForcer"):
        super().__init__(name)
        self._ego = ego
        self._min_mps = float(min_speed_kmh) / 3.6
        self._throttle = float(throttle)

    def update(self):
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.RUNNING
        speed = _speed_mps(self._ego)
        if 4.0 < speed < self._min_mps:
            control = self._ego.get_control()
            control.brake = 0.0
            control.throttle = self._throttle
            self._ego.apply_control(control)
        return py_trees.common.Status.RUNNING


class CrossTrafficRedLightViolationTown12(BasicScenario):
    """Signalized junction T-bone caused by one red-light-running vehicle."""

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
    def _get_bool_param(op, name, default):
        if not op or name not in op:
            return default
        raw = op[name]
        if isinstance(raw, dict):
            raw = raw.get("value", default)
        return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=120):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._ego = ego_vehicles[0]

        op = config.other_parameters if hasattr(config, "other_parameters") else {}
        self._activation_distance = self._get_param(op, "activation_distance_m", 80.0)
        self._direction = str(self._get_param(op, "direction", "right", cast=str)).lower()
        self._source_distance = self._get_param(op, "source_distance_m", 24.0)
        self._sink_distance = self._get_param(op, "sink_distance_m", 14.0)
        self._runner_speed = self._get_param(op, "runner_speed_kmh", 58.0) / 3.6
        self._trigger_distance = self._get_param(op, "trigger_distance_m", 13.0)
        self._sync_time = self._get_param(op, "sync_time_s", 2.0)
        self._runner_model = str(self._get_param(
            op, "runner_model", "vehicle.dodge.charger_2020", cast=str))
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 22.0)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 16.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.16)
        self._enable_flip = self._get_bool_param(op, "enable_flip", False)
        self._flip_launch_z = self._get_param(op, "flip_launch_z_mps", 6.0)
        self._road_friction_scale = self._get_param(op, "road_friction_scale", 1.0)
        self._side_wind_force = self._get_param(op, "side_wind_force_n", 0.0)
        self._side_wind_yaw_deg = self._get_param(op, "side_wind_yaw_deg", 0.0)
        self._max_run_time = self._get_param(op, "max_run_time", 16.0)
        self._post_impact_hold = self._get_param(op, "post_impact_hold", 4.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._ego_wp = None
        self._junction = None
        self._spawn_wp = None
        self._sink_wp = None
        self._spawn_location = None
        self._sink_location = None
        self._collision_location = None
        self._tl_dict = {}
        self._runner = None
        self._collision_sensor = None
        self._impact_detected = False

        super().__init__(
            name="CrossTrafficRedLightViolationTown12",
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
        ego_location = config.trigger_points[0].location
        self._ego_wp = self._map.get_waypoint(ego_location)

        starting_wp = self._ego_wp
        while not starting_wp.is_junction:
            next_wps = starting_wp.next(1.0)
            if not next_wps:
                raise ValueError("[RedLight] Failed to find a junction ahead of ego")
            starting_wp = next_wps[0]
        self._junction = starting_wp.get_junction()

        entry_wps, _ = get_junction_topology(self._junction)
        source_entry_wps = filter_junction_wp_direction(starting_wp, entry_wps, self._direction)
        if not source_entry_wps:
            raise ValueError(f"[RedLight] No cross-traffic entry for direction={self._direction}")

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

        sink_exit_wp = generate_target_waypoint(
            self._map.get_waypoint(spawn_wp.transform.location), 0)
        sink_wps = sink_exit_wp.next(self._sink_distance)
        if not sink_wps:
            raise ValueError("[RedLight] Failed to compute runner sink waypoint")
        self._sink_wp = sink_wps[0]
        self._sink_location = self._sink_wp.transform.location

        self._collision_location = collision_location
        if self._collision_location is None:
            raise ValueError("[RedLight] Failed to compute ego/runner conflict point")
        collision_wp = self._map.get_waypoint(self._collision_location)
        self._collision_location.z = collision_wp.transform.location.z

        traffic_lights = self._world.get_traffic_lights_in_junction(self._junction.id)
        if traffic_lights:
            ego_tl = get_closest_traffic_light(self._ego_wp, traffic_lights)
            for traffic_light in traffic_lights:
                self._tl_dict[traffic_light] = (
                    carla.TrafficLightState.Green if traffic_light == ego_tl
                    else carla.TrafficLightState.Red
                )

        runner_models = [
            self._runner_model,
            "vehicle.dodge.charger_2020",
            "vehicle.mercedes.coupe_2020",
            "vehicle.ford.mustang",
            "vehicle.tesla.model3",
        ]
        for model in dict.fromkeys(runner_models):
            self._runner = CarlaDataProvider.request_new_actor(
                model, self._spawn_location, rolename="red_light_runner",
            )
            if self._runner is not None:
                break
        if self._runner is None:
            raise RuntimeError("[RedLight] Runner vehicle spawn failed")

        self._runner.set_simulate_physics(True)
        self._runner.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self._runner.apply_control(carla.VehicleControl(
            throttle=0.0, brake=1.0, steer=0.0,
        ))
        self.other_actors.append(self._runner)

        collision_bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self._ego,
        )
        self._collision_sensor.listen(lambda event: self._on_collision(event))

        print(
            f"\n[RedLight] Spawn\n"
            f"  Runner at ({self._spawn_location.location.x:.1f}, "
            f"{self._spawn_location.location.y:.1f}) model={self._runner_model}\n"
            f"  Collision point=({self._collision_location.x:.1f}, "
            f"{self._collision_location.y:.1f})  sink=({self._sink_location.x:.1f}, "
            f"{self._sink_location.y:.1f})\n"
            f"  Runner speed={self._runner_speed * 3.6:.0f}km/h  "
            f"trigger_dist={self._trigger_distance:.0f}m  sync={self._sync_time:.1f}s  "
            f"signals={'locked' if self._tl_dict else 'none'}\n"
            f"  friction={self._road_friction_scale:.2f}  "
            f"side_wind={self._side_wind_force:.0f}N@{self._side_wind_yaw_deg:.0f}deg\n",
            flush=True,
        )

    def _on_collision(self, event):
        other = event.other_actor
        if self._runner is not None and other.id != self._runner.id:
            print(f"[RedLight] Ignoring non-runner collision: {other.type_id}", flush=True)
            return
        if not self._impact_detected:
            self._impact_detected = True
            print(f"[RedLight] *** COLLISION: {event.other_actor.type_id} ***", flush=True)

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateRedLightRunner",
        )

    def _create_behavior(self):
        root = py_trees.composites.Sequence("RedLightRunnerSequence")

        if self.route_mode and self._spawn_wp is not None and self._sink_wp is not None:
            root.add_child(HandleJunctionScenario(
                clear_junction=True,
                clear_ego_entry=True,
                remove_entries=[self._spawn_wp],
                remove_exits=[self._sink_wp],
                stop_entries=False,
                extend_road_exit=0,
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
        main_loop.add_child(_RunRedLightAttack(self))
        main_loop.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            max_brake=self._ego_cap_brake,
        ))
        main_loop.add_child(_EgoMinSpeedForcer(
            self._ego, min_speed_kmh=self._ego_min_speed_kmh))
        main_loop.add_child(TimeOut(self._max_run_time))
        root.add_child(main_loop)

        root.add_child(_PostImpactHold(self._post_impact_hold))
        root.add_child(ActorDestroy(self._runner, name="DestroyRunner"))

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
            "RedLightRunner_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        outer.add_child(CarlaRolloutLogger(
            actors=[("ego", self._ego), ("runner", self._runner)],
            scenario_id="red_light_runner_town13",
            route_id="red_light_runner_town13_route",
            name="CarlaRolloutLogger"))
        if self._tl_dict:
            outer.add_child(TrafficLightFreezer(self._tl_dict))
        outer.add_child(_SideWindForce(self))
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
            "RedLightRunnerCriteria",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        criteria.add_child(CollisionTest(self._ego))
        return criteria

    def remove_all_actors(self):
        if self._collision_sensor is not None and self._collision_sensor.is_alive:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()
            self._collision_sensor = None
        super().remove_all_actors()
