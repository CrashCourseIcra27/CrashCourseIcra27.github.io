"""Low grip adjacent vehicle spin blocks lane (Town04).

A vehicle in the adjacent lane spins on a reduced-grip surface and blocks the ego's lane.
"""

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
)
from srunner.scenariomanager.scenarioatomics.custom_atomics_low_grip_adjacent_vehicle_spin_blocks_lane_town04 import (
    EgoSpeedGovernor,
    HoldUntilEgoMoves,
    LaneCleaner,
    PeriodicStatusLogger,
)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario


def _speed_mps(actor):
    velocity = actor.get_velocity()
    return (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5


class LowGripAdjacentVehicleSpinBlocksLaneTown04(BasicScenario):
    """A vehicle in the adjacent lane loses control and spins into ego's lane."""

    timeout = 180

    class _ClearEgoFrontTraffic(py_trees.behaviour.Behaviour):
        def __init__(self, scenario, name="ClearEgoFrontTraffic"):
            super().__init__(name)
            self._scenario = scenario
            self._last_clean = -999.0

        def update(self):
            from srunner.scenariomanager.timer import GameTime
            now = GameTime.get_time()
            if now - self._last_clean < 0.25:
                return py_trees.common.Status.RUNNING
            self._last_clean = now
            scenario = self._scenario
            ego = scenario._ego
            if ego is None or not ego.is_alive:
                return py_trees.common.Status.RUNNING
            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_fwd = ego_tf.get_forward_vector()
            ego_right = ego_tf.get_right_vector()
            protected_ids = {ego.id}
            if scenario._adversarial is not None:
                protected_ids.add(scenario._adversarial.id)
            for actor in scenario._world.get_actors().filter("vehicle.*"):
                if actor.id in protected_ids:
                    continue
                loc = actor.get_location()
                dx = loc.x - ego_loc.x
                dy = loc.y - ego_loc.y
                ahead = dx * ego_fwd.x + dy * ego_fwd.y
                lateral = dx * ego_right.x + dy * ego_right.y
                if -8.0 <= ahead <= scenario._front_clear_ahead_m and abs(lateral) <= scenario._front_clear_half_width_m:
                    try:
                        actor_type = actor.type_id
                        actor.destroy()
                        print(f"[AdjacentSpin] removed front NPC {actor_type} ahead={ahead:.1f} lateral={lateral:.1f}", flush=True)
                    except RuntimeError:
                        pass
            return py_trees.common.Status.RUNNING

    class _AdjacentCutInSpin(py_trees.behaviour.Behaviour):
        def __init__(self, scenario, name="AdjacentCutInSpin"):
            super().__init__(name)
            self._scenario = scenario
            self._start_time = None
            self._started = False
            self._settled = False
            self._last_log = -999.0
            self._block_transform = None
            self._spin_start_ahead = 0.0
            self._spin_start_side = 0.0
            self._spin_start_time = None

        def initialise(self):
            self._start_time = None
            self._started = False
            self._settled = False
            self._last_log = -999.0
            self._block_transform = None
            self._spin_start_ahead = 0.0
            self._spin_start_side = 0.0
            self._spin_start_time = None

        def _relative_to_ego(self, ego_tf, location):
            ego_loc = ego_tf.location
            ego_fwd = ego_tf.get_forward_vector()
            ego_right = ego_tf.get_right_vector()
            dx = location.x - ego_loc.x
            dy = location.y - ego_loc.y
            return (
                dx * ego_fwd.x + dy * ego_fwd.y,
                dx * ego_right.x + dy * ego_right.y,
            )

        def _drive_adjacent_lane(self, scenario, adv, ego_tf):
            adv_wp = CarlaDataProvider.get_map().get_waypoint(
                adv.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            if adv_wp is not None:
                fwd = adv_wp.transform.get_forward_vector()
                yaw = adv_wp.transform.rotation.yaw
            else:
                fwd = ego_tf.get_forward_vector()
                yaw = ego_tf.rotation.yaw
            adv.set_target_velocity(carla.Vector3D(
                fwd.x * scenario._approach_speed,
                fwd.y * scenario._approach_speed,
                0.0,
            ))
            adv.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            adv.apply_control(carla.VehicleControl(throttle=0.32, brake=0.0, steer=0.0))
            return yaw

        def update(self):
            scenario = self._scenario
            ego = scenario._ego
            adv = scenario._adversarial
            if ego is None or adv is None or not ego.is_alive or not adv.is_alive:
                return py_trees.common.Status.FAILURE

            from srunner.scenariomanager.timer import GameTime
            current_time = GameTime.get_time()
            if self._start_time is None:
                if _speed_mps(ego) < 2.0:
                    adv.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
                    adv.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    return py_trees.common.Status.RUNNING
                self._start_time = current_time
            elapsed = current_time - self._start_time

            if getattr(scenario, "_ego_hit_adversary", False):
                scenario._hold_post_collision_static()
                if elapsed - self._last_log >= 1.0:
                    self._last_log = elapsed
                    print(f"[AdjacentSpin] t={elapsed:.1f}s post-impact static hold", flush=True)
                return py_trees.common.Status.RUNNING

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_fwd = ego_tf.get_forward_vector()
            ego_right = ego_tf.get_right_vector()
            ego_speed = _speed_mps(ego)
            adv_tf = adv.get_transform()
            adv_loc = adv_tf.location
            rel_ahead, rel_side = self._relative_to_ego(ego_tf, adv_loc)

            if not self._started:
                self._drive_adjacent_lane(scenario, adv, ego_tf)
                near_enough = -1.0 <= rel_ahead <= scenario._front_spin_ahead_m + 2.0
                timeout_close = (
                    elapsed >= scenario._front_spin_delay + 3.0
                    and -2.0 <= rel_ahead <= scenario._front_spin_ahead_m + 10.0
                )
                if elapsed >= scenario._front_spin_delay and (near_enough or timeout_close):
                    self._started = True
                    self._spin_start_ahead = rel_ahead
                    self._spin_start_side = rel_side
                    self._spin_start_time = current_time
                    print(
                        f"[AdjacentSpin] adjacent actor begins visible loss-of-control slide "
                        f"ahead={rel_ahead:.1f} side={rel_side:.1f}",
                        flush=True,
                    )

            if self._started:
                spin_elapsed = current_time - (self._spin_start_time or current_time)
                side = scenario._front_spin_side_m
                lateral = -1.0 if side > 0.0 else 1.0
                cut_duration = max(0.4, scenario._cut_in_duration)
                if spin_elapsed <= cut_duration:
                    progress = max(0.0, min(1.0, spin_elapsed / cut_duration))
                    target_ahead = scenario._front_spin_block_ahead_m
                    target_side = scenario._front_spin_cross_side_m
                    ahead = self._spin_start_ahead + progress * (target_ahead - self._spin_start_ahead)
                    side_offset = self._spin_start_side + progress * (target_side - self._spin_start_side)
                    loc = carla.Location(
                        x=ego_loc.x + ego_fwd.x * ahead + ego_right.x * side_offset,
                        y=ego_loc.y + ego_fwd.y * ahead + ego_right.y * side_offset,
                        z=max(adv_loc.z, ego_loc.z + 0.5),
                    )
                    yaw = ego_tf.rotation.yaw + lateral * scenario._front_spin_cut_yaw_deg * progress
                    adv.set_transform(carla.Transform(loc, carla.Rotation(yaw=yaw, pitch=0.0, roll=4.0 * progress)))
                    adv.set_target_velocity(carla.Vector3D(
                        ego_fwd.x * max(ego_speed * 0.45, 3.0) + ego_right.x * scenario._front_spin_lateral_mps * lateral,
                        ego_fwd.y * max(ego_speed * 0.45, 3.0) + ego_right.y * scenario._front_spin_lateral_mps * lateral,
                        0.0,
                    ))
                    adv.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, scenario._front_spin_yaw_rate * lateral))
                    adv.apply_control(carla.VehicleControl(throttle=scenario._cut_in_throttle, brake=0.05, steer=0.75 * lateral))
                else:
                    settle = max(0.0, min(1.0, (spin_elapsed - cut_duration) / max(0.2, scenario._spin_duration)))
                    yaw = ego_tf.rotation.yaw + lateral * (scenario._front_spin_cut_yaw_deg + scenario._front_spin_extra_spin_deg * settle)
                    if self._block_transform is None:
                        block_ahead = scenario._front_spin_block_ahead_m
                        block_side = scenario._front_spin_block_side_m
                        loc = carla.Location(
                            x=ego_loc.x + ego_fwd.x * block_ahead + ego_right.x * block_side,
                            y=ego_loc.y + ego_fwd.y * block_ahead + ego_right.y * block_side,
                            z=max(adv_loc.z, ego_loc.z + 0.5),
                        )
                        self._block_transform = carla.Transform(loc, carla.Rotation(yaw=yaw, pitch=0.0, roll=7.0))
                    adv.set_transform(self._block_transform)
                    adv.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    adv.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, scenario._front_spin_yaw_rate * lateral * (1.0 - settle)))
                    adv.apply_control(carla.VehicleControl(throttle=0.0, brake=0.85, steer=0.0, hand_brake=True))
                    if not self._settled and settle >= 1.0:
                        self._settled = True
                        print("[AdjacentSpin] actor spun across ego lane and blocked ahead", flush=True)

            if elapsed - self._last_log >= 1.0:
                self._last_log = elapsed
                print(
                    f"[AdjacentSpin] t={elapsed:.1f}s staged={self._started} "
                    f"settled={self._settled} rel_ahead={rel_ahead:.1f} rel_side={rel_side:.1f}",
                    flush=True,
                )

            if elapsed >= scenario._front_spin_delay + scenario._front_spin_duration:
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING

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
                 timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._ego = ego_vehicles[0]

        op = dict(config.other_parameters) if hasattr(config, "other_parameters") else {}
        if hasattr(config, "scenario") and hasattr(config.scenario, "other_parameters"):
            op.update(config.scenario.other_parameters)
        if isinstance(op, dict) and isinstance(op.get("other_parameters"), dict):
            op = op["other_parameters"]
        self._activation_distance = self._get_param(op, "activation_distance_m", 140.0)
        self._adv_spawn_behind = self._get_param(op, "adv_spawn_behind_ego", 55.0)
        self._adv_spawn_ahead = self._get_param(op, "adv_spawn_ahead_ego", None)
        self._approach_speed = self._get_param(op, "approach_speed_kmh", 104.0) / 3.6
        self._ahead_trigger_dist = self._get_param(op, "ahead_trigger_distance_m", 2.0)
        self._road_friction_scale = self._get_param(op, "road_friction_scale", 0.78)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 88.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.12)
        self._aftermath_duration = self._get_param(op, "aftermath_duration_s", 6.0)
        self._lead_distance = self._get_param(op, "ego_lead_vehicle_distance_m", 0.0)
        self._lead_speed_kmh = self._get_param(op, "ego_lead_vehicle_speed_kmh", 58.0)

        self._cut_in_steer = self._get_param(op, "cut_in_steer", 0.45)
        self._cut_in_throttle = self._get_param(op, "cut_in_throttle", 0.75)
        self._cut_in_duration = self._get_param(op, "cut_in_duration", 0.35)
        self._drift_steer = self._get_param(op, "drift_steer", 1.00)
        self._drift_steer_sign = self._get_param(op, "drift_steer_sign", -1.0)
        self._drift_throttle = self._get_param(op, "drift_throttle", 0.20)
        self._drift_brake = self._get_param(op, "drift_brake", 0.30)
        self._drift_duration = self._get_param(op, "drift_duration", 0.85)
        self._spin_throttle = self._get_param(op, "spin_throttle", 0.15)
        self._spin_duration = self._get_param(op, "spin_duration", 1.20)
        self._block_duration = self._get_param(op, "block_duration", 5.50)
        self._front_spin_delay = self._get_param(op, "front_spin_delay_s", 3.0)
        self._front_spin_ahead_m = self._get_param(op, "front_spin_ahead_m", 11.0)
        self._front_spin_side_m = self._get_param(op, "front_spin_side_m", -3.4)
        self._front_spin_lateral_mps = self._get_param(op, "front_spin_lateral_mps", 5.8)
        self._front_spin_yaw_rate = self._get_param(op, "front_spin_yaw_rate", 2.8)
        self._front_spin_duration = self._get_param(op, "front_spin_duration_s", 3.2)
        self._front_spin_forward_drop_m = self._get_param(op, "front_spin_forward_drop_m", 7.5)
        self._front_spin_block_ahead_m = self._get_param(op, "front_spin_block_ahead_m", 2.2)
        self._front_spin_block_side_m = self._get_param(op, "front_spin_block_side_m", 0.0)
        self._front_spin_cross_side_m = self._get_param(op, "front_spin_cross_side_m", 0.8)
        self._front_spin_cut_yaw_deg = self._get_param(op, "front_spin_cut_yaw_deg", 55.0)
        self._front_spin_extra_spin_deg = self._get_param(op, "front_spin_extra_spin_deg", 115.0)
        self._front_clear_ahead_m = self._get_param(op, "front_clear_ahead_m", 45.0)
        self._front_clear_half_width_m = self._get_param(op, "front_clear_half_width_m", 4.5)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._adversarial = None
        self._lead_vehicle = None
        self._left_lane_wps = []
        self._ego_collision_sensor = None
        self._ego_hit_logged = False
        self._ego_hit_adversary = False
        self._post_impact_static = False
        self._post_impact_transforms = {}

        super().__init__(
            name="LowGripAdjacentVehicleSpinBlocksLaneTown04",
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
        bp = world.get_blueprint_library().find("static.trigger.friction")
        bp.set_attribute("friction", str(friction))
        bp.set_attribute("extent_x", "1000000.0")
        bp.set_attribute("extent_y", "1000000.0")
        bp.set_attribute("extent_z", "1000000.0")
        try:
            world.spawn_actor(bp, carla.Transform(carla.Location(-10000.0, -10000.0, 0.0)))
        except RuntimeError:
            pass

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

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        left_wp = ego_wp.get_left_lane()
        if left_wp is None or left_wp.lane_type != carla.LaneType.Driving:
            raise RuntimeError(
                f"[AdjacentSpin] No drivable adjacent lane from "
                f"road={ego_wp.road_id} lane={ego_wp.lane_id}"
            )

        if self._adv_spawn_ahead is not None:
            adv_spawn_wp = self._walk_waypoint(left_wp, self._adv_spawn_ahead, forward=True) or left_wp
            spawn_desc = f"ahead={self._adv_spawn_ahead:.1f}m"
        else:
            adv_spawn_wp = self._walk_waypoint(left_wp, self._adv_spawn_behind, forward=False) or left_wp
            spawn_desc = f"behind={self._adv_spawn_behind:.1f}m"
        self._left_lane_wps = self._build_plan(adv_spawn_wp)
        adv_transform = carla.Transform(
            carla.Location(
                adv_spawn_wp.transform.location.x,
                adv_spawn_wp.transform.location.y,
                adv_spawn_wp.transform.location.z + 0.5,
            ),
            adv_spawn_wp.transform.rotation,
        )
        self._adversarial = CarlaDataProvider.request_new_actor(
            "vehicle.tesla.model3", adv_transform, rolename="unstable_adjacent_vehicle",
        )
        if self._adversarial is None:
            raise RuntimeError("[AdjacentSpin] Adversary spawn failed")
        self._adversarial.set_simulate_physics(True)
        self._adversarial.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.other_actors.append(self._adversarial)

        # Collision sensor on ego purely to LOG what the ego strikes, so we can
        # confirm a vehicle-vs-vehicle impact (not a barrier).
        try:
            adv_id = self._adversarial.id
            cbp = self._world.get_blueprint_library().find("sensor.other.collision")
            self._ego_collision_sensor = self._world.spawn_actor(
                cbp, carla.Transform(), attach_to=self._ego)

            self._ego_hit_logged = False
            self._ego_hit_adversary = False

            def _on_ego_collision(event, adv_id=adv_id):
                if self._ego_hit_logged:
                    return
                other = event.other_actor
                tag = "ADVERSARY" if other.id == adv_id else "OTHER"
                self._ego_hit_logged = True
                self._ego_hit_adversary = other.id == adv_id
                print(f"[AdjacentSpin] EGO COLLISION with {other.type_id} ({tag})", flush=True)
            self._ego_collision_sensor.listen(_on_ego_collision)
        except RuntimeError:
            self._ego_collision_sensor = None

        if self._lead_distance > 0.0:
            lead_wp = self._walk_waypoint(ego_wp, self._lead_distance, forward=True) or ego_wp
            lead_transform = carla.Transform(
                carla.Location(
                    lead_wp.transform.location.x,
                    lead_wp.transform.location.y,
                    lead_wp.transform.location.z + 0.5,
                ),
                lead_wp.transform.rotation,
            )
            self._lead_vehicle = CarlaDataProvider.request_new_actor(
                "vehicle.tesla.cybertruck", lead_transform, rolename="ego_lane_lead",
            )
            if self._lead_vehicle is not None:
                self._lead_vehicle.set_simulate_physics(True)
                tm_port = CarlaDataProvider.get_traffic_manager_port()
                tm = CarlaDataProvider.get_client().get_trafficmanager(tm_port)
                self._lead_vehicle.set_autopilot(True, tm_port)
                tm.set_desired_speed(self._lead_vehicle, self._lead_speed_kmh)
                tm.auto_lane_change(self._lead_vehicle, False)
                tm.ignore_vehicles_percentage(self._lead_vehicle, 100)
                tm.ignore_walkers_percentage(self._lead_vehicle, 100)
                tm.ignore_lights_percentage(self._lead_vehicle, 100)
                tm.ignore_signs_percentage(self._lead_vehicle, 100)
                self.other_actors.append(self._lead_vehicle)

        self._post_impact_static = False
        self._post_impact_transforms = {}

        print(
            f"\n[AdjacentSpin] Spawn\n"
            f"  Ego road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  ADV road={adv_spawn_wp.road_id} lane={adv_spawn_wp.lane_id} "
            f"{spawn_desc} approach={self._approach_speed * 3.6:.1f} km/h\n"
            f"  trigger_ahead={self._ahead_trigger_dist:.1f}m friction={self._road_friction_scale:.2f}\n",
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
        actors = [actor for actor in (self._ego, self._adversarial) if actor is not None and actor.is_alive]
        if not self._post_impact_static:
            self._post_impact_static = True
            self._post_impact_transforms = {actor.id: actor.get_transform() for actor in actors}
            for actor in actors:
                try:
                    actor.set_simulate_physics(False)
                except RuntimeError:
                    pass
            print("[AdjacentSpin] post-impact static hold engaged", flush=True)

        for actor in actors:
            transform = self._post_impact_transforms.get(actor.id)
            if transform is not None:
                try:
                    actor.set_transform(transform)
                except RuntimeError:
                    pass
            self._stop_actor_motion(actor)

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateAdjacentLaneSpinCollision",
        )

    def _create_behavior(self):
        P = py_trees.common.ParallelPolicy
        root = py_trees.composites.Sequence("AdjacentSpin_Root")

        # The cut-in spin self-gates on the ego having moved (see
        # _AdjacentCutInSpin.update), so it runs directly as the first step.
        # The old phase0 Parallel(HoldUntilEgoMoves) gate un-latched when the
        # ego slowed and prevented the spin from ever running.
        phase1 = py_trees.composites.Parallel("Phase1_AdjacentCutInSpin", policy=P.SUCCESS_ON_ONE)
        phase1.add_child(self._AdjacentCutInSpin(self))
        root.add_child(phase1)

        root.add_child(TimeOut(self._aftermath_duration, name="Phase3_Aftermath"))

        cleanup = py_trees.composites.Sequence("Cleanup")
        cleanup.add_child(ActorDestroy(self._adversarial, name="DestroyADV"))
        if self._lead_vehicle is not None:
            cleanup.add_child(ActorDestroy(self._lead_vehicle, name="DestroyLead"))
        root.add_child(cleanup)

        lanes = [self._trigger_wp]
        left_lane = self._trigger_wp.get_left_lane()
        if left_lane is not None and left_lane.lane_type == carla.LaneType.Driving:
            lanes.append(left_lane)
        protected = [self._ego, self._adversarial, self._lead_vehicle]

        outer = py_trees.composites.Parallel("AdjacentSpin_Outer", policy=P.SUCCESS_ON_ONE)
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=protected,
            lane_waypoints=lanes,
            center_location=self._trigger_wp.transform.location,
            radius=500.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        outer.add_child(PeriodicStatusLogger(
            self._ego, self._adversarial, self._lead_vehicle,
            interval=1.0, name="StatusLogger",
        ))
        outer.add_child(self._ClearEgoFrontTraffic(self))
        outer.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            name="EgoSpeedGovernor",
        ))
        return outer

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
