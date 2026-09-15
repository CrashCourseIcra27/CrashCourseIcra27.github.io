"""Cyclist topples into lane (Town11).

A cyclist riding ahead topples sideways into the ego's lane.
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
from srunner.scenariomanager.scenarioatomics.custom_atomics_cyclist_topples_into_lane_town11_variant1 import (
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


def _planar_distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class _CyclistFallManager(py_trees.behaviour.Behaviour):
    """
    Manages the cyclist: rides ahead in a bike/parking lane, then drifts
    sharply into ego's lane when ego is close behind.
    Keeps running (logs status) until collision or timeout.
    """

    def __init__(self, scenario, name="CyclistFallManager"):
        super().__init__(name)
        self._scenario = scenario
        self._start_time = None
        self._last_log = -1.0
        self._fall_triggered = False
        self._fall_time = None
        self._fall_pose_applied = False

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        scenario = self._scenario
        ego = scenario._ego
        cyclist = scenario._cyclist

        if ego is None or not ego.is_alive:
            return py_trees.common.Status.FAILURE
        if cyclist is None or not cyclist.is_alive:
            return py_trees.common.Status.FAILURE

        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0

        ego_t = ego.get_transform()
        ego_loc = ego_t.location
        cyclist_loc = cyclist.get_location()

        # Compute forward offset (positive = cyclist ahead)
        dx = cyclist_loc.x - ego_loc.x
        dy = cyclist_loc.y - ego_loc.y
        fwd = ego_t.get_forward_vector()
        forward_offset = dx * fwd.x + dy * fwd.y
        dist = _planar_distance(ego_loc, cyclist_loc)

        # Log every second
        if elapsed - self._last_log >= 1.0:
            self._last_log = elapsed
            ego_spd = _speed_mps(ego) * 3.6
            cyc_spd = _speed_mps(cyclist) * 3.6
            print(
                f"[CyclistFall] t={elapsed:.1f}s  ego={ego_spd:.0f}km/h  "
                f"cyclist={cyc_spd:.0f}km/h  fwd_offset={forward_offset:.1f}m  "
                f"dist={dist:.1f}m  fall={self._fall_triggered}  "
                f"impact={scenario._impact_detected}",
                flush=True,
            )

        # Check for collision -> end scenario
        if scenario._impact_detected:
            return py_trees.common.Status.SUCCESS

        # Trigger the fall: when ego is within trigger_dist behind the cyclist.
        if not self._fall_triggered and 0 < forward_offset <= scenario._fall_trigger_dist:
            self._fall_triggered = True
            self._fall_time = elapsed
            self._fall_pose_applied = False
            print(
                f"[CyclistFall] *** CYCLIST LOSES BALANCE and veers into lane! ***",
                flush=True,
            )

        if self._fall_triggered:
            fall_elapsed = elapsed - self._fall_time if self._fall_time else 0.0
            cyclist_t = cyclist.get_transform()
            if not self._fall_pose_applied:
                self._fall_pose_applied = True
                tipped = carla.Transform(
                    cyclist_t.location,
                    carla.Rotation(
                        pitch=cyclist_t.rotation.pitch,
                        yaw=cyclist_t.rotation.yaw,
                        roll=72.0 * scenario._fall_lateral_sign,
                    ),
                )
                try:
                    cyclist.set_transform(tipped)
                    cyclist_t = tipped
                    print("[CyclistFall] Bicycle visibly tipped onto its side", flush=True)
                except RuntimeError:
                    pass
            ego_fwd = ego_t.get_forward_vector()
            target = carla.Location(
                ego_loc.x + ego_fwd.x * scenario._fall_target_ahead_m,
                ego_loc.y + ego_fwd.y * scenario._fall_target_ahead_m,
                cyclist_loc.z,
            )
            dx = target.x - cyclist_loc.x
            dy = target.y - cyclist_loc.y
            mag = max(0.01, math.sqrt(dx * dx + dy * dy))
            slide_speed = scenario._fall_slide_speed
            if fall_elapsed > scenario._fall_lateral_duration + 0.5:
                # Fall motion complete — freeze bicycle to prevent endless road-contact spam
                try:
                    cyclist.set_simulate_physics(False)
                    cyclist.set_target_velocity(carla.Vector3D(0, 0, 0))
                except RuntimeError:
                    pass
            elif fall_elapsed > scenario._fall_lateral_duration:
                slide_speed *= 0.35
            cyclist.set_target_velocity(carla.Vector3D(
                dx / mag * slide_speed,
                dy / mag * slide_speed,
                0.0,
            ))

        # Timeout
        if elapsed >= scenario._max_run_time:
            return py_trees.common.Status.FAILURE

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


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    """Override autopilot braking -- force ego to maintain minimum speed."""

    def __init__(self, ego, min_speed_kmh=35.0, throttle=0.8,
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
        if speed > 5.0 and speed < self._min_mps:
            ctrl = self._ego.get_control()
            ctrl.brake = 0.0
            ctrl.throttle = self._throttle
            self._ego.apply_control(ctrl)
        return py_trees.common.Status.RUNNING


class CyclistTopplesIntoLaneTown11Variant1(BasicScenario):
    """
    A cyclist riding ahead of the ego suddenly loses balance and
    falls into ego's lane, causing a collision.
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

        op = config.other_parameters if hasattr(config, "other_parameters") else {}
        self._activation_distance = self._get_param(op, "activation_distance_m", 80.0)
        self._cyclist_spawn_ahead = self._get_param(op, "cyclist_spawn_ahead_m", 40.0)
        self._cyclist_lateral_offset = self._get_param(op, "cyclist_lateral_offset_m", 2.0)
        self._cyclist_speed = self._get_param(op, "cyclist_speed_kmh", 20.0) / 3.6
        self._fall_trigger_dist = self._get_param(op, "fall_trigger_dist_m", 24.0)
        self._fall_target_ahead_m = self._get_param(op, "fall_target_ahead_m", 2.5)
        self._fall_slide_speed = self._get_param(op, "fall_slide_speed_mps", 8.0)
        self._fall_lateral_speed = self._get_param(op, "fall_lateral_speed_mps", 4.8)
        self._fall_lateral_sign = self._get_param(op, "fall_lateral_sign", 1.0)
        self._fall_lateral_duration = self._get_param(op, "fall_lateral_duration_s", 2.6)
        self._fall_min_forward_speed = self._get_param(op, "fall_min_forward_speed_mps", 0.2)
        self._fall_forward_decay = self._get_param(op, "fall_forward_decay_mps2", 5.0)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 40.0)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 35.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.12)
        self._max_run_time = self._get_param(op, "max_run_time", 30.0)
        self._aftermath_duration = self._get_param(op, "aftermath_duration_s", 4.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._cyclist = None
        self._cyclist_plan = []
        self._collision_sensor = None
        self._impact_detected = False

        super().__init__(
            name="CyclistTopplesIntoLaneTown11Variant1",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            terminate_on_failure=True,
            criteria_enable=criteria_enable,
        )

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

    def _build_plan(self, start_wp, distance=300.0, step=2.0):
        from agents.navigation.local_planner import RoadOption
        waypoints = []
        cursor = start_wp
        travelled = 0.0
        while cursor is not None and travelled < distance:
            candidates = cursor.next(step)
            if not candidates:
                break
            cursor = candidates[0]
            waypoints.append((cursor, RoadOption.LANEFOLLOW))
            travelled += step
        return waypoints

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        # Cyclist ahead, offset laterally (simulating bike lane)
        cyclist_wp = self._walk_waypoint(
            ego_wp, self._cyclist_spawn_ahead, forward=True
        ) or ego_wp
        cyclist_t = cyclist_wp.transform
        right = cyclist_t.get_right_vector()
        cyclist_loc = carla.Location(
            x=cyclist_t.location.x + right.x * self._cyclist_lateral_offset,
            y=cyclist_t.location.y + right.y * self._cyclist_lateral_offset,
            z=cyclist_t.location.z + 0.5,
        )
        cyclist_transform = carla.Transform(cyclist_loc, cyclist_t.rotation)

        # Build a waypoint plan from the cyclist spawn forward
        self._cyclist_plan = self._build_plan(cyclist_wp)

        bicycle_models = [
            "vehicle.diamondback.century",
            "vehicle.bh.crossbike",
            "vehicle.gazelle.omafiets",
        ]
        self._cyclist = None
        for model in bicycle_models:
            self._cyclist = CarlaDataProvider.request_new_actor(
                model, cyclist_transform, rolename="falling_cyclist",
            )
            if self._cyclist is not None:
                break
        if self._cyclist is None:
            raise RuntimeError("[CyclistFall] Bicycle spawn failed")

        self._cyclist.set_simulate_physics(True)
        self.other_actors.append(self._cyclist)

        # Collision sensor
        collision_bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self._ego,
        )
        self._collision_sensor.listen(lambda event: self._on_collision(event))

        print(
            f"\n[CyclistFall] Spawn\n"
            f"  Ego road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  Cyclist at ({cyclist_loc.x:.1f}, {cyclist_loc.y:.1f}) "
            f"ahead={self._cyclist_spawn_ahead:.1f}m "
            f"lat={self._cyclist_lateral_offset:.1f}m\n"
            f"  speed={self._cyclist_speed * 3.6:.0f}km/h  "
            f"fall_trigger={self._fall_trigger_dist:.0f}m  "
            f"fall_lateral={self._fall_lateral_speed:.1f}m/s\n",
            flush=True,
        )

    def _on_collision(self, event):
        other = event.other_actor
        if other is not None and other.type_id.startswith("static."):
            # Silently ignore road/terrain contacts from fallen bicycle physics
            return
        if not self._impact_detected:
            self._impact_detected = True
            print(f"[CyclistFall] *** COLLISION: {other.type_id} ***", flush=True)
            # Stop bicycle physics immediately to prevent road-contact event spam
            if self._cyclist is not None and self._cyclist.is_alive:
                try:
                    self._cyclist.set_simulate_physics(False)
                    self._cyclist.set_target_velocity(carla.Vector3D(0, 0, 0))
                except RuntimeError:
                    pass

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateCyclistFallCollision",
        )

    def _create_behavior(self):
        root = py_trees.composites.Sequence("CyclistFallSequence")

        # Phase 0: wait for ego
        sync = py_trees.composites.Parallel(
            "Phase0_Sync",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
        )
        sync.add_child(HoldUntilEgoMoves(self._cyclist, self._ego))
        root.add_child(sync)

        main_loop = py_trees.composites.Parallel(
            "Phase1_MainLoop",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        main_loop.add_child(WaypointFollower(
            self._cyclist,
            target_speed=self._cyclist_speed,
            plan=self._cyclist_plan,
            avoid_collision=False,
            name="CyclistRide",
        ))
        main_loop.add_child(_CyclistFallManager(self))
        main_loop.add_child(TimeOut(self._max_run_time))
        root.add_child(main_loop)

        # Phase 2: aftermath hold
        root.add_child(_PostImpactHold(self._aftermath_duration))

        # Cleanup
        root.add_child(ActorDestroy(self._cyclist, name="DestroyCyclist"))

        # Outer wrapper: LaneCleaner + speed control run throughout ALL phases
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
            "CyclistFall_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego, self._cyclist],
            lane_waypoints=lane_wps,
            center_location=self._trigger_wp.transform.location,
            radius=500.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        outer.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
        ))
        outer.add_child(_EgoMinSpeedForcer(
            self._ego, min_speed_kmh=self._ego_min_speed_kmh))
        return outer

    def _create_test_criteria(self):
        criteria = py_trees.composites.Parallel(
            "CyclistFallCriteria",
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
