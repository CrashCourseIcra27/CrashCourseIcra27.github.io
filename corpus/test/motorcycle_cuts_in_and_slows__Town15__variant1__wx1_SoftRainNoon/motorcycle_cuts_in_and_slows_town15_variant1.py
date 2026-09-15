"""Motorcycle cuts in and slows (Town15).

A lane-filtering motorcycle cuts in ahead of the ego and slows.
"""

import math
import os

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
from srunner.scenariomanager.scenarioatomics.custom_atomics_motorcycle_cuts_in_and_slows_town15_variant1 import (
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


# ── helpers ──────────────────────────────────────────────────────────


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    """Override ego brake to keep speed above min."""

    def __init__(self, ego, min_speed_kmh=45.0, throttle=1.0,
                 name="EgoMinSpeedForcer"):
        super().__init__(name)
        self._ego = ego
        self._min_mps = float(min_speed_kmh) / 3.6
        self._throttle = float(throttle)

    def update(self):
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.RUNNING
        speed = _speed_mps(self._ego)
        if speed < self._min_mps:
            ctrl = self._ego.get_control()
            ctrl.brake = 0.0
            ctrl.throttle = self._throttle
            self._ego.apply_control(ctrl)
        return py_trees.common.Status.RUNNING


class _ProximityPurge(py_trees.behaviour.Behaviour):
    def __init__(self, ego, protected_actors, radius=45.0, interval=0.3,
                 name="ProximityPurge"):
        super().__init__(name)
        self._ego = ego
        self._protected = set(a.id for a in protected_actors if a is not None)
        self._radius = float(radius)
        self._interval = float(interval)
        self._last = None

    def update(self):
        if os.environ.get("CUSTOM_DISABLE_BACKGROUND_NPCS", "").lower() in ("1", "true", "yes", "on"):
            return py_trees.common.Status.RUNNING

        now = GameTime.get_time()
        if self._last is not None and (now - self._last) < self._interval:
            return py_trees.common.Status.RUNNING
        self._last = now
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.RUNNING
        world = CarlaDataProvider.get_world()
        if world is None:
            return py_trees.common.Status.RUNNING
        ego_loc = self._ego.get_location()
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.id in self._protected:
                continue
            if math.hypot(actor.get_location().x - ego_loc.x,
                          actor.get_location().y - ego_loc.y) > self._radius:
                continue
            try:
                actor.destroy()
            except RuntimeError:
                pass
        return py_trees.common.Status.RUNNING


class _WaitUntilAlongside(py_trees.behaviour.Behaviour):
    """SUCCESS when ego is alongside motorcycle (same longitudinal position).

    Uses road-aligned coordinates. Triggers when the ego is within
    [lon_behind, lon_ahead] of the motorcycle longitudinally.
    """

    def __init__(self, ego, motorcycle, carla_map,
                 lon_behind=-3.0, lon_ahead=5.0,
                 name="WaitUntilAlongside"):
        super().__init__(name)
        self._ego = ego
        self._moto = motorcycle
        self._map = carla_map
        self._lon_behind = float(lon_behind)
        self._lon_ahead = float(lon_ahead)
        self._tick = 0

    def update(self):
        if not (self._ego and self._moto
                and self._ego.is_alive and self._moto.is_alive):
            return py_trees.common.Status.RUNNING

        ego_loc = self._ego.get_location()
        moto_loc = self._moto.get_location()
        moto_wp = self._map.get_waypoint(moto_loc)
        if moto_wp is None:
            return py_trees.common.Status.RUNNING
        road_fwd = moto_wp.transform.get_forward_vector()

        dx = ego_loc.x - moto_loc.x
        dy = ego_loc.y - moto_loc.y
        lon_off = dx * road_fwd.x + dy * road_fwd.y
        dist = math.hypot(dx, dy)

        self._tick += 1
        if self._tick % 40 == 0:
            print(f"[MotoFilter] APPROACH  dist={dist:.1f}m  "
                  f"lon={lon_off:.1f}m  "
                  f"ego={_speed_mps(self._ego)*3.6:.0f}km/h  "
                  f"moto={_speed_mps(self._moto)*3.6:.0f}km/h",
                  flush=True)

        if self._lon_behind <= lon_off <= self._lon_ahead:
            print(f"[MotoFilter] TRIGGER alongside  lon={lon_off:.1f}m  "
                  f"dist={dist:.1f}m", flush=True)
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


# ── main cut-in behaviour ────────────────────────────────────────────


class _SideCutIn(py_trees.behaviour.Behaviour):
    """Drive motorcycle from left lane into ego's lane.

    Uses road-aligned velocity: forward = ego_speed, lateral = fixed cut speed.
    Anti-tip via set_target_angular_velocity.
    """

    def __init__(self, scenario_ref, motorcycle, ego, carla_map,
                 lat_cut_speed=6.0,
                 fwd_offset=0.0,
                 flip_launch_z=8.0,
                 name="SideCutIn"):
        super().__init__(name)
        self._scenario = scenario_ref
        self._moto = motorcycle
        self._ego = ego
        self._map = carla_map
        self._lat_cut = float(lat_cut_speed)
        self._fwd_off = float(fwd_offset)
        self._flip_z = float(flip_launch_z)
        self._flip_applied = False
        self._tick = 0
        self._cut_start_time = None
        self._lane_entered_time = None

    def initialise(self):
        self._cut_start_time = GameTime.get_time()
        print("[MotoFilter] CUT-IN phase started", flush=True)

    def update(self):
        if not (self._moto and self._ego
                and self._moto.is_alive and self._ego.is_alive):
            return py_trees.common.Status.SUCCESS

        # ── collision → flip & exit ──
        if self._scenario._impact_detected:
            if not self._flip_applied:
                v = self._moto.get_velocity()
                self._moto.set_target_velocity(
                    carla.Vector3D(v.x * 0.3, v.y * 0.3, self._flip_z))
                self._flip_applied = True
                print("[MotoFilter] *** IMPACT — flip ***", flush=True)
            return py_trees.common.Status.SUCCESS

        ego_loc = self._ego.get_location()
        moto_loc = self._moto.get_location()

        # road-aligned basis
        moto_wp = self._map.get_waypoint(moto_loc)
        if moto_wp is None:
            return py_trees.common.Status.RUNNING
        road_fwd = moto_wp.transform.get_forward_vector()
        road_right = moto_wp.transform.get_right_vector()

        dx = ego_loc.x - moto_loc.x
        dy = ego_loc.y - moto_loc.y
        lat_off = dx * road_right.x + dy * road_right.y
        lon_off = dx * road_fwd.x + dy * road_fwd.y

        ego_speed = _speed_mps(self._ego)

        # Phase A: sharp cross-lane move while the motorcycle is still offset.
        # Phase B: once it has entered the ego lane, slow down so the ego reaches it.
        if abs(lat_off) > 1.2:
            fwd_speed = max(4.0, ego_speed - 1.0 + self._fwd_off)
            lat_v = self._lat_cut if lat_off > 0 else -self._lat_cut
            self._lane_entered_time = None
        else:
            if self._lane_entered_time is None:
                self._lane_entered_time = GameTime.get_time()
                print(f"[MotoFilter] lane entered  lon={lon_off:.1f}m", flush=True)
            lat_v = 0.0
            in_lane_time = GameTime.get_time() - self._lane_entered_time
            if in_lane_time < 0.8:
                fwd_speed = max(1.5, ego_speed - 8.0)
            else:
                fwd_speed = max(2.0, ego_speed - 5.0)

        vx = fwd_speed * road_fwd.x + lat_v * road_right.x
        vy = fwd_speed * road_fwd.y + lat_v * road_right.y
        self._moto.set_target_velocity(carla.Vector3D(vx, vy, 0.0))

        # prevent tipping
        self._moto.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        self._moto.apply_control(
            carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))

        self._tick += 1
        if self._tick % 10 == 0:
            dist = math.sqrt(dx ** 2 + dy ** 2)
            elapsed = GameTime.get_time() - self._cut_start_time if self._cut_start_time else 0
            print(f"[MotoFilter] CUT-IN  t={elapsed:.1f}s  dist={dist:.1f}m  "
                  f"lat={lat_off:.2f}  lon={lon_off:.1f}  "
                  f"fwd_v={fwd_speed:.1f}  lat_v={lat_v:.1f}  "
                  f"ego={ego_speed * 3.6:.0f}km/h",
                  flush=True)

        return py_trees.common.Status.RUNNING


class _PostImpactHold(py_trees.behaviour.Behaviour):
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


# ── main scenario class ──────────────────────────────────────────────


class MotorcycleCutsInAndSlowsTown15Variant1(BasicScenario):
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
        self._moto_spawn_ahead = self._get_param(op, "moto_spawn_ahead_ego", 40.0)
        self._moto_lateral_offset = self._get_param(op, "moto_lateral_offset_m", 3.4)
        self._moto_cruise_speed_kmh = self._get_param(op, "moto_cruise_speed_kmh", 40.0)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 50.0)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 45.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.12)
        self._flip_launch_z = self._get_param(op, "flip_launch_z_mps", 8.0)
        self._aftermath_duration = self._get_param(op, "aftermath_duration_s", 6.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._motorcycle = None
        self._moto_plan = []
        self._collision_sensor = None
        self._impact_detected = False

        super().__init__(
            name="MotorcycleCutsInAndSlowsTown15Variant1",
            ego_vehicles=ego_vehicles, config=config, world=world,
            debug_mode=debug_mode, terminate_on_failure=True,
            criteria_enable=criteria_enable)

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
        from agents.navigation.local_planner import RoadOption
        wps, cursor, d = [], start_wp, 0.0
        while cursor is not None and d < distance:
            candidates = cursor.next(step)
            if not candidates:
                break
            cursor = candidates[0]
            wps.append((cursor, RoadOption.LANEFOLLOW))
            d += step
        return wps

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving)
        adjacent_wp = ego_wp.get_left_lane()
        lane_side = "left"
        virtual_adjacent = False
        if adjacent_wp is None or adjacent_wp.lane_type != carla.LaneType.Driving:
            adjacent_wp = ego_wp.get_right_lane()
            lane_side = "right"
        if adjacent_wp is None or adjacent_wp.lane_type != carla.LaneType.Driving:
            adjacent_wp = ego_wp
            lane_side = "virtual-right"
            virtual_adjacent = True

        moto_spawn_wp = (self._walk_waypoint(
            adjacent_wp, self._moto_spawn_ahead, forward=True) or adjacent_wp)
        self._moto_plan = self._build_plan(moto_spawn_wp)

        spawn_loc = carla.Location(moto_spawn_wp.transform.location)
        if virtual_adjacent:
            road_right = moto_spawn_wp.transform.get_right_vector()
            spawn_loc.x += road_right.x * self._moto_lateral_offset
            spawn_loc.y += road_right.y * self._moto_lateral_offset
            spawn_loc.z += road_right.z * self._moto_lateral_offset

        moto_tf = carla.Transform(
            carla.Location(spawn_loc.x,
                           spawn_loc.y,
                           spawn_loc.z + 0.5),
            moto_spawn_wp.transform.rotation)

        for model in ["vehicle.kawasaki.ninja", "vehicle.yamaha.yzf",
                       "vehicle.harley-davidson.low_rider"]:
            self._motorcycle = CarlaDataProvider.request_new_actor(
                model, moto_tf, rolename="motorcycle_filter")
            if self._motorcycle is not None:
                break
        if self._motorcycle is None:
            raise RuntimeError("[MotoFilter] Motorcycle spawn failed")
        self._motorcycle.set_simulate_physics(True)
        self._motorcycle.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.other_actors.append(self._motorcycle)

        col_bp = self._world.get_blueprint_library().find(
            "sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self._ego)
        self._collision_sensor.listen(lambda e: self._on_collision(e))

        # purge both lanes
        try:
            prot = {self._ego.id, self._motorcycle.id}
            el = self._ego.get_location()
            for a in self._world.get_actors().filter("vehicle.*"):
                if a.id in prot:
                    continue
                if math.hypot(a.get_location().x - el.x,
                              a.get_location().y - el.y) > 300:
                    continue
                wp = self._map.get_waypoint(a.get_location())
                if wp and (
                    (wp.road_id == ego_wp.road_id
                     and wp.lane_id == ego_wp.lane_id)
                    or (wp.road_id == adjacent_wp.road_id
                        and wp.lane_id == adjacent_wp.lane_id)):
                    try:
                        a.destroy()
                    except RuntimeError:
                        pass
        except Exception as exc:
            print(f"[MotoFilter] purge warning: {exc}", flush=True)

        print(f"\n[MotoFilter] Spawn\n"
              f"  Ego lane={ego_wp.lane_id}  "
              f"Moto lane={moto_spawn_wp.lane_id} ({lane_side})\n"
              f"  ahead={self._moto_spawn_ahead:.0f}m  "
              f"cruise={self._moto_cruise_speed_kmh:.0f}km/h\n",
              flush=True)

    def _on_collision(self, event):
        if self._motorcycle is None:
            return
        if event.other_actor is None:
            return
        if event.other_actor.id != self._motorcycle.id:
            return
        if not self._impact_detected:
            self._impact_detected = True
            print("[MotoFilter] *** MOTORCYCLE COLLISION ***", flush=True)

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego, self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateMotorcycleLaneFilter")

    def _create_behavior(self):
        root = py_trees.composites.Sequence("MotoFilterSeq")

        # Phase 0: sync
        root.add_child(HoldUntilEgoMoves(self._motorcycle, self._ego))

        # Phase 1: approach — moto cruises left lane at 40km/h, ego catches up
        approach = py_trees.composites.Parallel(
            "Phase1_Approach",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        approach.add_child(WaypointFollower(
            self._motorcycle,
            target_speed=self._moto_cruise_speed_kmh / 3.6,
            plan=self._moto_plan, avoid_collision=False,
            name="MotoCruise"))
        approach.add_child(_WaitUntilAlongside(
            self._ego, self._motorcycle, self._map,
            lon_behind=-8.0, lon_ahead=-2.0))
        approach.add_child(TimeOut(30.0))
        approach.add_child(EgoSpeedGovernor(
            self._ego, speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            max_brake=self._ego_cap_brake))
        approach.add_child(_EgoMinSpeedForcer(
            self._ego, min_speed_kmh=self._ego_min_speed_kmh))
        root.add_child(approach)

        # Phase 2: side cut-in — motorcycle moves right at ego speed
        cut_in = py_trees.composites.Parallel(
            "Phase2_CutIn",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        cut_in.add_child(_SideCutIn(
            self, self._motorcycle, self._ego, self._map,
            lat_cut_speed=7.5,
            fwd_offset=0.0,
            flip_launch_z=self._flip_launch_z))
        cut_in.add_child(EgoSpeedGovernor(
            self._ego, speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            max_brake=self._ego_cap_brake))
        cut_in.add_child(_EgoMinSpeedForcer(
            self._ego, min_speed_kmh=self._ego_min_speed_kmh))
        cut_in.add_child(TimeOut(5.0))
        root.add_child(cut_in)

        # Phase 3: aftermath
        root.add_child(_PostImpactHold(self._aftermath_duration))
        root.add_child(ActorDestroy(self._motorcycle, name="DestroyMoto"))

        # Outer: lane cleaner + proximity purge
        lanes = [self._trigger_wp]
        try:
            lw = self._trigger_wp.get_left_lane()
            if lw and lw.lane_type == carla.LaneType.Driving:
                lanes.append(lw)
            rw = self._trigger_wp.get_right_lane()
            if rw and rw.lane_type == carla.LaneType.Driving:
                lanes.append(rw)
        except Exception:
            pass

        outer = py_trees.composites.Parallel(
            "MotoFilter_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego, self._motorcycle],
            lane_waypoints=lanes,
            center_location=self._trigger_wp.transform.location,
            radius=300.0, interval=0.5))
        outer.add_child(_ProximityPurge(
            self._ego, protected_actors=[self._ego, self._motorcycle],
            radius=45.0, interval=0.25))
        return outer

    def _create_test_criteria(self):
        c = py_trees.composites.Parallel(
            "MotoFilterCriteria",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c.add_child(CollisionTest(self._ego))
        return c

    def remove_all_actors(self):
        if self._collision_sensor is not None and self._collision_sensor.is_alive:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()
            self._collision_sensor = None
        super().remove_all_actors()
