"""Heavy truck cut in and slows (Town12).

A heavy truck changes lane into the ego's path and slows.
"""

import math

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy,
    WaypointFollower,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
)
from srunner.scenariomanager.scenarioatomics.custom_atomics_heavy_truck_cut_in_and_slows_town12 import (
    CarlaRolloutLogger,
    EgoSpeedGovernor,
    HoldUntilEgoMoves,
    LaneCleaner,
    PeriodicStatusLogger,
    StopActorsOnEgoCollision,
)
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenarios.basic_scenario import BasicScenario


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
    """SUCCESS when the truck is alongside the ego (small forward offset)."""

    def __init__(self, ego, truck, max_forward_m=4.0,
                 name="WaitUntilAlongside"):
        super().__init__(name)
        self._ego = ego
        self._truck = truck
        self._max_fwd = float(max_forward_m)

    def update(self):
        if not (self._ego and self._truck
                and self._ego.is_alive and self._truck.is_alive):
            return py_trees.common.Status.RUNNING

        truck_t = self._truck.get_transform()
        ego_loc = self._ego.get_location()
        truck_loc = truck_t.location

        dx = ego_loc.x - truck_loc.x
        dy = ego_loc.y - truck_loc.y

        fwd = truck_t.get_forward_vector()
        forward_offset = dx * fwd.x + dy * fwd.y

        if abs(forward_offset) <= self._max_fwd:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class _ForcedLaneChange(py_trees.behaviour.Behaviour):
    """Override velocity each tick: forward + lateral drift via set_target_velocity."""

    def __init__(self, actor, forward_speed_ms, lateral_speed_ms=4.0,
                 duration=1.0, lateral_sign=1.0, name="ForcedLaneChange"):
        super().__init__(name)
        self._actor = actor
        self._forward_speed = float(forward_speed_ms)
        self._lateral_speed = float(lateral_speed_ms)
        self._duration = float(duration)
        self._lateral_sign = float(lateral_sign)
        self._start_time = None

    def initialise(self):
        from srunner.scenariomanager.timer import GameTime
        self._start_time = GameTime.get_time()

    def update(self):
        from srunner.scenariomanager.timer import GameTime
        if self._actor is None or not self._actor.is_alive:
            return py_trees.common.Status.SUCCESS
        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0
        if elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS

        t = self._actor.get_transform()
        fwd = t.get_forward_vector()
        right = t.get_right_vector()
        lat = self._lateral_speed * self._lateral_sign
        vx = self._forward_speed * fwd.x + lat * right.x
        vy = self._forward_speed * fwd.y + lat * right.y
        self._actor.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
        self._actor.apply_control(carla.VehicleControl(throttle=0.25, brake=0.0, steer=0.28 * self._lateral_sign))
        return py_trees.common.Status.RUNNING


class _SideWindForce(py_trees.behaviour.Behaviour):
    """Apply a continuous crosswind force to ego and truck."""

    def __init__(self, scenario, name="SideWindForce"):
        super().__init__(name)
        self._scenario = scenario
        self._last_log = -999.0
        self._activation_time = None

    def update(self):
        scenario = self._scenario
        if scenario._side_wind_force <= 0.0:
            return py_trees.common.Status.RUNNING

        if not scenario._is_side_wind_ready():
            self._activation_time = None
            return py_trees.common.Status.RUNNING

        now = GameTime.get_time()
        if self._activation_time is None:
            self._activation_time = now

        ramp = max(0.0, min(1.0, (now - self._activation_time) / scenario._side_wind_ramp_s))
        if ramp <= 0.0:
            return py_trees.common.Status.RUNNING

        yaw = math.radians(scenario._side_wind_yaw_deg)
        force = carla.Vector3D(
            math.cos(yaw) * scenario._side_wind_force * ramp,
            math.sin(yaw) * scenario._side_wind_force * ramp,
            0.0,
        )
        actors = [scenario._truck]
        if scenario._side_wind_affect_ego:
            actors.insert(0, scenario._ego)

        for actor in actors:
            if actor is not None and actor.is_alive and hasattr(actor, "add_force"):
                actor.add_force(force)

        if now - self._last_log >= 2.0:
            self._last_log = now
            print(
                f"[TruckLaneChange] side_wind force={scenario._side_wind_force * ramp:.0f}N "
                f"yaw={scenario._side_wind_yaw_deg:.0f}deg ramp={ramp:.2f}",
                flush=True,
            )
        return py_trees.common.Status.RUNNING


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    """Prevent the ego autopilot from braking below a minimum speed.

    Once the ego is moving (speed > 5 m/s), this behavior overrides the
    autopilot's brake and applies throttle whenever speed drops below the
    minimum.  This models a distracted / committed driver who does not
    brake in time.
    """

    def __init__(self, ego, min_speed_kmh=68.0, throttle=1.0,
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


class HeavyTruckCutInAndSlowsTown12(BasicScenario):
    """
    Truck in adjacent lane ahead; ego approaches and truck suddenly
    cuts in with hard braking — a panic lane-change collision.
    """
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

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._ego = ego_vehicles[0]

        op = config.other_parameters if hasattr(config, "other_parameters") else {}
        self._activation_distance = self._get_param(op, "activation_distance_m", 120.0)
        self._truck_spawn_ahead = self._get_param(op, "truck_spawn_ahead_m", 12.0)
        self._truck_speed_kmh = self._get_param(op, "truck_speed_kmh", 45.0)
        self._lane_change_duration = self._get_param(op, "truck_lane_change_duration_s", 1.2)
        self._aftermath_duration = self._get_param(op, "aftermath_duration_s", 6.0)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 65.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.15)
        self._road_friction_scale = self._get_param(op, "road_friction_scale", 1.0)
        self._side_wind_force = self._get_param(op, "side_wind_force_n", 0.0)
        self._side_wind_yaw_deg = self._get_param(op, "side_wind_yaw_deg", 0.0)
        self._side_wind_affect_ego = self._get_param(op, "side_wind_affect_ego", 0.0) > 0.5
        self._alongside_threshold_m = self._get_param(op, "alongside_threshold_m", 6.0)
        self._side_wind_activation_forward_m = self._get_param(
            op, "side_wind_activation_forward_m", self._alongside_threshold_m + 4.0)
        self._side_wind_ramp_s = self._get_param(op, "side_wind_ramp_s", 0.8)
        self._lateral_speed_ms = self._get_param(op, "lateral_speed_ms", 4.5)
        self._truck_lanechange_speed_kmh = self._get_param(op, "truck_lanechange_speed_kmh", 38.0)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 55.0)
        self._lateral_sign = self._get_param(op, "lateral_sign", 1.0)
        self._truck_model = str(self._get_param(
            op, "truck_model", "vehicle.carlamotors.carlacola", cast=str))

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._truck = None
        self._truck_plan = []
        self._truck_spawn_wp = None
        self._runtime_lateral_sign = self._lateral_sign

        super().__init__(
            name="HeavyTruckCutInAndSlowsTown12",
            ego_vehicles=ego_vehicles, config=config, world=world,
            debug_mode=debug_mode, terminate_on_failure=True,
            criteria_enable=criteria_enable)

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

    def _spawn_truck(self, transform, rolename):
        models = [self._truck_model,
                  "vehicle.carlamotors.carlacola",
                  "vehicle.carlamotors.european_hgv",
                  "vehicle.tesla.cybertruck",
                  "vehicle.carla.firetruck"]
        for model in dict.fromkeys(models):
            actor = CarlaDataProvider.request_new_actor(
                model, transform, rolename=rolename)
            if actor is not None:
                return actor
        return None

    def _build_plan(self, start_wp, distance=800.0, step=2.0):
        waypoints, cursor, travelled = [], start_wp, 0.0
        while cursor is not None and travelled < distance:
            candidates = cursor.next(step)
            if not candidates:
                break
            cursor = candidates[0]
            waypoints.append(cursor)
            travelled += step
        return waypoints

    def _find_adjacent_lane_reference(self):
        """Find the adjacent lane at the actual highway trigger point."""
        reference_wp = self._trigger_wp
        for adjacent_wp in (
                reference_wp.get_left_lane(),
                reference_wp.get_right_lane()):
            if (adjacent_wp is not None
                    and adjacent_wp.lane_type == carla.LaneType.Driving
                    and adjacent_wp.road_id == reference_wp.road_id):
                return reference_wp, adjacent_wp
        return reference_wp, None

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving)
        lane_reference_wp, spawn_lane = self._find_adjacent_lane_reference()
        if spawn_lane is None or spawn_lane.lane_type != carla.LaneType.Driving:
            raise RuntimeError("[TruckLaneChange] No adjacent driving lane for truck")

        ego_loc = ego_wp.transform.location
        trigger_loc = self._trigger_wp.transform.location
        ego_fwd = ego_wp.transform.get_forward_vector()
        to_trigger_x = trigger_loc.x - ego_loc.x
        to_trigger_y = trigger_loc.y - ego_loc.y
        trigger_gap_m = max(
            0.0, to_trigger_x * ego_fwd.x + to_trigger_y * ego_fwd.y)
        spawn_back = max(0.0, trigger_gap_m - self._truck_spawn_ahead)
        self._truck_spawn_wp = self._walk_waypoint(
            spawn_lane, spawn_back, forward=False) or spawn_lane

        self._truck_plan = self._build_plan(self._truck_spawn_wp)

        transform = carla.Transform(
            carla.Location(
                self._truck_spawn_wp.transform.location.x,
                self._truck_spawn_wp.transform.location.y,
                self._truck_spawn_wp.transform.location.z + 0.5),
            self._truck_spawn_wp.transform.rotation)
        self._truck = self._spawn_truck(transform, "lane_change_truck")
        if self._truck is None:
            raise RuntimeError("[TruckLaneChange] Truck spawn failed")
        self._truck.set_simulate_physics(True)
        self._truck.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.other_actors.append(self._truck)

        truck_right = self._truck_spawn_wp.transform.get_right_vector()
        target_loc = lane_reference_wp.transform.location
        truck_loc = self._truck_spawn_wp.transform.location
        target_dx = target_loc.x - truck_loc.x
        target_dy = target_loc.y - truck_loc.y
        self._runtime_lateral_sign = 1.0 if (
            target_dx * truck_right.x + target_dy * truck_right.y
        ) >= 0.0 else -1.0

        print(
            f"\n[TruckLaneChange] Spawn\n"
            f"  Ego road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  Lane reference road={lane_reference_wp.road_id} "
            f"lane={lane_reference_wp.lane_id} "
            f"trigger_gap={trigger_gap_m:.1f}m\n"
            f"  Truck road={self._truck_spawn_wp.road_id} "
            f"lane={self._truck_spawn_wp.lane_id} "
            f"ahead={self._truck_spawn_ahead:.1f}m "
            f"speed={self._truck_speed_kmh:.1f} km/h\n"
            f"  lateral_speed={self._lateral_speed_ms:.1f} m/s "
            f"duration={self._lane_change_duration:.2f}s "
            f"alongside_threshold={self._alongside_threshold_m:.1f}m\n"
            f"  lanechange_speed={self._truck_lanechange_speed_kmh:.1f} km/h"
            f"  ego_min_speed={self._ego_min_speed_kmh:.1f} km/h"
            f"  lateral_sign={self._runtime_lateral_sign:+.1f}\n"
            f"  friction={self._road_friction_scale:.2f}  "
            f"side_wind={self._side_wind_force:.0f}N@{self._side_wind_yaw_deg:.0f}deg "
            f"affect_ego={int(self._side_wind_affect_ego)} "
            f"activate_fwd={self._side_wind_activation_forward_m:.1f}m "
            f"ramp={self._side_wind_ramp_s:.1f}s\n",
            flush=True)

    def _is_side_wind_ready(self):
        if not (self._ego and self._truck and self._ego.is_alive and self._truck.is_alive):
            return False

        ego_speed = math.sqrt(sum(
            component * component for component in (
                self._ego.get_velocity().x,
                self._ego.get_velocity().y,
                self._ego.get_velocity().z,
            )
        ))
        if ego_speed < 8.0 / 3.6:
            return False

        truck_t = self._truck.get_transform()
        ego_loc = self._ego.get_location()
        truck_loc = truck_t.location
        dx = ego_loc.x - truck_loc.x
        dy = ego_loc.y - truck_loc.y
        fwd = truck_t.get_forward_vector()
        forward_offset = dx * fwd.x + dy * fwd.y
        return abs(forward_offset) <= self._side_wind_activation_forward_m

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego, self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateTruckSuddenLaneChange")

    def _launch_truck(self):
        if self._truck is None or not self._truck.is_alive:
            return
        fwd = self._truck_spawn_wp.transform.get_forward_vector()
        speed = self._truck_speed_kmh / 3.6
        self._truck.set_target_velocity(
            carla.Vector3D(fwd.x * speed, fwd.y * speed, 0.0))

    def _create_behavior(self):
        from agents.navigation.local_planner import RoadOption
        plan = [(wp, RoadOption.LANEFOLLOW) for wp in self._truck_plan]
        P = py_trees.common.ParallelPolicy
        root = py_trees.composites.Sequence("TruckLaneChange_Root")

        # Phase 0 — hold until ego moves
        phase0 = py_trees.composites.Parallel("Phase0_Sync", policy=P.SUCCESS_ON_ONE)
        phase0.add_child(TimeOut(8.0, name="Phase0_Timeout"))
        phase0.add_child(HoldUntilEgoMoves(self._truck, self._ego, name="HoldTruck"))
        root.add_child(phase0)

        root.add_child(_OneShot(self._launch_truck, name="LaunchTruck"))

        # Phase 1 — truck cruises in adjacent lane until alongside ego
        phase1 = py_trees.composites.Parallel("Phase1_Cruise", policy=P.SUCCESS_ON_ONE)
        phase1.add_child(WaypointFollower(
            self._truck, target_speed=self._truck_speed_kmh / 3.6,
            plan=plan, name="TruckLaneFollow"))
        phase1.add_child(_WaitUntilAlongside(
            self._ego, self._truck,
            max_forward_m=self._alongside_threshold_m,
            name="AlongsideTrigger"))
        phase1.add_child(TimeOut(30.0, name="Phase1_MaxTimeout"))
        root.add_child(phase1)

        # Phase 2 — forced lateral drift into ego's lane + hard braking (panic cut-in)
        lc_fwd = self._truck_lanechange_speed_kmh / 3.6
        phase2 = py_trees.composites.Parallel("Phase2_LaneChange", policy=P.SUCCESS_ON_ONE)
        phase2.add_child(_ForcedLaneChange(
            self._truck, forward_speed_ms=lc_fwd,
            lateral_speed_ms=self._lateral_speed_ms,
            duration=self._lane_change_duration,
            lateral_sign=self._runtime_lateral_sign,
            name="ForceTruckIntoEgoLane"))
        phase2.add_child(TimeOut(self._lane_change_duration + 1.0, name="Phase2_Timeout"))
        root.add_child(phase2)

        # Phase 3 — keep rolling forward slowly (aftermath)
        root.add_child(_ForcedLaneChange(
            self._truck, forward_speed_ms=lc_fwd,
            lateral_speed_ms=0.0, duration=self._aftermath_duration,
            name="TruckBlockEgoLane"))

        cleanup = py_trees.composites.Sequence("Cleanup")
        cleanup.add_child(ActorDestroy(self._truck, name="DestroyTruck"))
        root.add_child(cleanup)

        # Parallel helpers
        lanes = [self._trigger_wp]
        left_lane = self._trigger_wp.get_left_lane()
        if left_lane is not None and left_lane.lane_type == carla.LaneType.Driving:
            lanes.append(left_lane)

        outer = py_trees.composites.Parallel("TruckLaneChange_Outer", policy=P.SUCCESS_ON_ONE)
        outer.add_child(root)
        outer.add_child(CarlaRolloutLogger(
            actors=[("ego", self._ego), ("truck", self._truck)],
            scenario_id="truck_lane_change",
            route_id="truck_sudden_lane_change_route",
            name="CarlaRolloutLogger"))
        outer.add_child(_SideWindForce(self))
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego, self._truck],
            lane_waypoints=lanes,
            center_location=self._trigger_wp.transform.location,
            radius=500.0, interval=1.0, name="ScenarioLaneCleaner"))
        outer.add_child(PeriodicStatusLogger(
            self._ego, self._truck, None, interval=1.0, name="StatusLogger"))
        outer.add_child(EgoSpeedGovernor(
            self._ego, speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake, max_brake=self._ego_cap_brake,
            name="EgoSpeedGovernor"))
        outer.add_child(_EgoMinSpeedForcer(
            self._ego, min_speed_kmh=self._ego_min_speed_kmh,
            name="EgoMinSpeedForcer"))
        outer.add_child(StopActorsOnEgoCollision(
            self._ego, actors_to_stop=[self._truck],
            name="StopOnCollision"))
        return outer

    def _create_test_criteria(self):
        return [CollisionTest(self._ego)]

    def __del__(self):
        self.remove_all_actors()
