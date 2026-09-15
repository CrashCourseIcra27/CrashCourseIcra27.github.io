"""Pedestrian darts from behind bus (Town03).

A child pedestrian darts into the road from behind a stopped school bus.
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
from srunner.scenariomanager.scenarioatomics.custom_atomics_pedestrian_darts_from_behind_bus_town03 import (
    EgoSpeedGovernor,
    HoldUntilEgoMoves,
    LaneCleaner,
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


def _pick_driving_lane(*candidates):
    for candidate in candidates:
        if candidate and candidate.lane_type == carla.LaneType.Driving:
            return candidate
    return None


class _RunSchoolChildCrossing(py_trees.behaviour.Behaviour):
    def __init__(self, scenario, name="RunSchoolChildCrossing"):
        super().__init__(name)
        self._scenario = scenario
        self._start_time = None
        self._dash_started = False
        self._dash_start_time = None
        self._last_move_time = None
        self._target_location = None
        self._last_log = -999.0
        self._child_dash_started = False
        self._impact_hold_start = None

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        scenario = self._scenario
        if scenario._child_collision_seen:
            if self._impact_hold_start is None:
                self._impact_hold_start = scenario._child_collision_time or GameTime.get_time()
                print("[SchoolChild] *** CHILD IMPACT — holding aftermath ***", flush=True)
            child = scenario._child
            if child is not None and child.is_alive:
                child.apply_control(carla.WalkerControl(
                    direction=carla.Vector3D(0.0, 0.0, 0.0),
                    speed=0.0,
                ))
            if GameTime.get_time() - self._impact_hold_start >= scenario._post_event_hold:
                scenario._impact_detected = True
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING

        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0
        ego_loc = scenario._ego.get_location()
        occluder_loc = scenario._occluder.get_location() if scenario._occluder else ego_loc
        dist_to_occluder = _planar_distance(ego_loc, occluder_loc)

        if (not self._dash_started and
            (dist_to_occluder <= scenario._trigger_distance_to_occluder or
             elapsed >= scenario._child_dash_delay)):
            self._dash_started = True
            self._dash_start_time = elapsed
            self._last_move_time = elapsed
            self._target_location = scenario._child_target_location
            print(
                f"[SchoolChild] Ego reached school-zone occlusion at {dist_to_occluder:.1f}m — child crossing!",
                flush=True,
            )

        child = scenario._child
        if child is not None and child.is_alive:
            if self._dash_started:
                child_loc = child.get_location()
                if scenario._child_track_ego:
                    ego_tf = scenario._ego.get_transform()
                    fwd = ego_tf.get_forward_vector()
                    target = carla.Location(
                        x=ego_tf.location.x + fwd.x * scenario._child_intercept_lead,
                        y=ego_tf.location.y + fwd.y * scenario._child_intercept_lead,
                        z=child_loc.z,
                    )
                else:
                    target = self._target_location or child_loc
                dx = target.x - child_loc.x
                dy = target.y - child_loc.y
                magnitude = math.hypot(dx, dy)
                dt = max(0.03, min(0.1, elapsed - (self._last_move_time or elapsed)))
                self._last_move_time = elapsed
                if magnitude > 0.25:
                    step = min(magnitude, scenario._child_dash_speed * dt)
                    child.set_location(carla.Location(
                        child_loc.x + dx / magnitude * step,
                        child_loc.y + dy / magnitude * step,
                        child_loc.z,
                    ))
                    child.apply_control(carla.WalkerControl(
                        direction=carla.Vector3D(dx / magnitude, dy / magnitude, 0.0),
                        speed=scenario._child_dash_speed,
                    ))
                else:
                    child.apply_control(carla.WalkerControl(
                        direction=carla.Vector3D(0.0, 0.0, 0.0),
                        speed=0.0,
                    ))
            else:
                child.apply_control(carla.WalkerControl(
                    direction=carla.Vector3D(0.0, 0.0, 0.0),
                    speed=0.0,
                ))

        ego_speed = _speed_mps(scenario._ego) * 3.6
        if ego_speed < scenario._ego_min_speed_kmh and not scenario._impact_detected:
            control = scenario._ego.get_control()
            control.brake = 0.0
            control.throttle = max(control.throttle, scenario._ego_force_throttle)
            control.steer = 0.0
            scenario._ego.apply_control(control)

        if self._dash_start_time is not None:
            yield_elapsed = elapsed - self._dash_start_time
            if yield_elapsed <= scenario._ego_yield_duration:
                control = scenario._ego.get_control()
                control.throttle = 0.0
                control.brake = max(control.brake, scenario._ego_yield_brake)
                scenario._ego.apply_control(control)

        if elapsed - self._last_log >= 1.0:
            self._last_log = elapsed
            child_speed = _speed_mps(child) * 3.6 if child else 0.0
            print(
                f"[SchoolChild] t={elapsed:.1f}s ego={ego_speed:.0f}km/h child={child_speed:.0f}km/h "
                f"dist_occ={dist_to_occluder:.1f}m dash={self._dash_started}",
                flush=True,
            )

        if self._dash_started and child is not None and child.is_alive:
            if _planar_distance(child.get_location(), scenario._child_target_location) <= 0.6:
                if elapsed >= self._dash_start_time + scenario._clear_hold_duration:
                    return py_trees.common.Status.SUCCESS

        if elapsed >= scenario._max_run_time:
            if self._dash_started:
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING
class PedestrianDartsFromBehindBusTown03(BasicScenario):
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
        if isinstance(op, dict) and isinstance(op.get("other_parameters"), dict):
            op = op["other_parameters"]
        self._activation_distance = self._get_param(op, "activation_distance_m", 80.0)
        self._occluder_ahead_distance = self._get_param(op, "occluder_ahead_distance_m", 32.0)
        self._dropoff_ahead_distance = self._get_param(op, "dropoff_ahead_distance_m", 18.0)
        self._dropoff_rear_distance = self._get_param(op, "dropoff_rear_distance_m", 6.0)
        self._curb_offset = self._get_param(op, "curb_offset_m", 2.8)
        self._child_back_offset = self._get_param(op, "child_back_offset_m", 1.4)
        self._child_front_offset = self._get_param(op, "child_front_offset_m", 4.8)
        self._child_side_offset = self._get_param(op, "child_side_offset_m", 3.0)
        self._child_dash_speed = self._get_param(op, "child_dash_speed_mps", 5.2)
        self._child_dash_delay = self._get_param(op, "child_dash_delay_s", 2.2)
        self._trigger_distance_to_occluder = self._get_param(op, "trigger_distance_to_occluder_m", 14.0)
        self._crossing_distance = self._get_param(op, "crossing_distance_m", 8.5)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 22.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.24)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 18.0)
        self._ego_force_throttle = self._get_param(op, "ego_force_throttle", 0.46)
        self._ego_yield_duration = self._get_param(op, "ego_yield_duration_s", 1.7)
        self._ego_yield_brake = self._get_param(op, "ego_yield_brake", 0.45)
        self._clear_hold_duration = self._get_param(op, "clear_hold_duration_s", 2.0)
        self._max_run_time = self._get_param(op, "max_run_time", 16.0)
        self._post_event_hold = self._get_param(op, "post_event_hold_s", 1.8)
        self._occluder_model = str(self._get_param(
            op, "occluder_model", "vehicle.mitsubishi.fusorosa", cast=str))
        self._force_collision_target = bool(self._get_param(
            op, "force_child_collision_target", 0, cast=int))
        self._child_collision_ahead = self._get_param(
            op, "child_collision_ahead_m", 10.0)
        self._child_track_ego = bool(self._get_param(
            op, "child_track_ego_target", 0, cast=int))
        self._child_intercept_lead = self._get_param(
            op, "child_intercept_lead_m", 3.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._occluder = None
        self._dropoff_front = None
        self._dropoff_rear = None
        self._child = None
        self._child_target_location = None
        self._collision_sensor = None
        self._impact_detected = False
        self._child_collision_seen = False
        self._child_collision_time = None

        super().__init__(
            name="PedestrianDartsFromBehindBusTown03",
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

    def _spawn_stopped_vehicle(self, transform, rolename, models):
        actor = None
        for model in models:
            actor = CarlaDataProvider.request_new_actor(model, transform, rolename=rolename)
            if actor is not None:
                break
        if actor is None:
            raise RuntimeError(f"[SchoolChild] Failed to spawn {rolename}")
        actor.set_simulate_physics(True)
        actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.other_actors.append(actor)
        return actor

    def _spawn_stopped_vehicle_near(self, base_wp, right_vec, offset, rolename, models):
        for candidate_offset in (offset, offset - 0.8, offset + 0.8, offset - 1.6, offset + 1.6):
            loc = carla.Location(
                x=base_wp.transform.location.x + right_vec.x * candidate_offset,
                y=base_wp.transform.location.y + right_vec.y * candidate_offset,
                z=base_wp.transform.location.z + 0.3,
            )
            try:
                return self._spawn_stopped_vehicle(
                    carla.Transform(loc, base_wp.transform.rotation), rolename, models)
            except RuntimeError:
                continue
        raise RuntimeError(f"[SchoolChild] Failed to spawn {rolename}")

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving,
        )
        front_wp = self._walk_waypoint(ego_wp, self._dropoff_ahead_distance, forward=True) or ego_wp
        rear_wp = self._walk_waypoint(ego_wp, self._dropoff_rear_distance, forward=True) or ego_wp
        occluder_wp = self._walk_waypoint(ego_wp, self._occluder_ahead_distance, forward=True) or ego_wp

        ego_right = ego_wp.transform.get_right_vector()
        self._dropoff_front = self._spawn_stopped_vehicle_near(
            front_wp, ego_right, self._curb_offset,
            "school_dropoff_front",
            [
                "vehicle.mitsubishi.fusorosa",
                "vehicle.ford.ambulance",
                "vehicle.volkswagen.t2",
                "vehicle.mercedes.sprinter",
                "vehicle.dodge.charger_2020",
                "vehicle.audi.etron",
                "vehicle.tesla.model3",
            ],
        )
        self._dropoff_rear = self._spawn_stopped_vehicle_near(
            rear_wp, ego_right, self._curb_offset,
            "school_dropoff_rear",
            ["vehicle.nissan.micra", "vehicle.audi.a2", "vehicle.lincoln.mkz_2020"],
        )

        occluder_right = occluder_wp.transform.get_right_vector()
        occluder_forward = occluder_wp.transform.get_forward_vector()
        try:
            self._occluder = self._spawn_stopped_vehicle_near(
                occluder_wp, occluder_right, self._curb_offset,
                "school_occluder_vehicle",
                [
                    self._occluder_model,
                    "vehicle.mitsubishi.fusorosa",
                    "vehicle.mercedes.sprinter",
                    "vehicle.volkswagen.t2",
                    "vehicle.ford.ambulance",
                    "vehicle.dodge.charger_2020",
                    "vehicle.audi.etron",
                    "vehicle.tesla.model3",
                ],
            )
        except RuntimeError:
            self._occluder = self._dropoff_front
            print("[SchoolChild] Reusing front dropoff bus/van as occluder", flush=True)
        occluder_transform = self._occluder.get_transform()

        child_models = [
            "walker.pedestrian.0013",
            "walker.pedestrian.0014",
            "walker.pedestrian.0001",
            "walker.pedestrian.0005",
            "walker.pedestrian.0010",
        ]
        self._child = None
        child_loc = None
        child_offsets = [
            (self._child_front_offset, self._child_side_offset),
            (self._child_front_offset + 1.0, self._child_side_offset),
            (self._child_front_offset + 2.0, self._child_side_offset + 0.5),
            (self._child_front_offset - 0.8, self._child_side_offset + 1.0),
            (self._child_front_offset + 1.5, self._child_side_offset + 1.4),
        ]
        for front_offset, side_offset in child_offsets:
            candidate_loc = carla.Location(
                x=occluder_transform.location.x + occluder_forward.x * front_offset + occluder_right.x * side_offset,
                y=occluder_transform.location.y + occluder_forward.y * front_offset + occluder_right.y * side_offset,
                z=occluder_transform.location.z + 0.5,
            )
            child_transform = carla.Transform(candidate_loc, occluder_wp.transform.rotation)
            for model in child_models:
                self._child = CarlaDataProvider.request_new_actor(
                    model, child_transform, rolename="school_child_pedestrian",
                )
                if self._child is not None:
                    child_loc = candidate_loc
                    break
            if self._child is not None:
                break
        if self._child is None:
            raise RuntimeError("[SchoolChild] Failed to spawn child pedestrian")
        self.other_actors.append(self._child)

        crossing_distance_ahead = (
            self._child_collision_ahead if self._force_collision_target else
            max(4.0, self._occluder_ahead_distance + self._child_front_offset - 0.8)
        )
        crossing_wp = self._walk_waypoint(ego_wp, crossing_distance_ahead, forward=True) or ego_wp
        crossing_right = crossing_wp.transform.get_right_vector()
        if self._force_collision_target:
            self._child_target_location = carla.Location(
                x=crossing_wp.transform.location.x - crossing_right.x * 0.4,
                y=crossing_wp.transform.location.y - crossing_right.y * 0.4,
                z=child_loc.z,
            )
        else:
            self._child_target_location = carla.Location(
                x=child_loc.x - crossing_right.x * self._crossing_distance,
                y=child_loc.y - crossing_right.y * self._crossing_distance,
                z=child_loc.z,
            )

        collision_bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self._ego,
        )
        self._collision_sensor.listen(lambda event: self._on_collision(event))

        print(
            f"\n[SchoolChild] Spawn\n"
            f"  ego road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  dropoff front=({self._dropoff_front.get_location().x:.1f}, {self._dropoff_front.get_location().y:.1f}) "
            f"rear=({self._dropoff_rear.get_location().x:.1f}, {self._dropoff_rear.get_location().y:.1f})\n"
            f"  occluder=({occluder_transform.location.x:.1f}, {occluder_transform.location.y:.1f})\n"
            f"  child=({child_loc.x:.1f}, {child_loc.y:.1f}) target=({self._child_target_location.x:.1f}, {self._child_target_location.y:.1f})\n",
            flush=True,
        )

    def _on_collision(self, event):
        other = event.other_actor
        if self._child is not None and other.id != self._child.id:
            print(f"[SchoolChild] Ignoring non-child collision: {other.type_id}", flush=True)
            return
        if not self._child_collision_seen:
            self._child_collision_seen = True
            self._child_collision_time = GameTime.get_time()
            print(f"[SchoolChild] *** COLLISION: {other.type_id} ***", flush=True)

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateSchoolDropoffChildMidblock",
        )

    def _create_behavior(self):
        root = py_trees.composites.Sequence("SchoolChildSequence")

        sync = py_trees.composites.Parallel(
            "Phase0_Sync",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
        )
        sync.add_child(HoldUntilEgoMoves(self._occluder, self._ego))
        root.add_child(sync)

        main_loop = py_trees.composites.Parallel(
            "Phase1_MainLoop",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        main_loop.add_child(_RunSchoolChildCrossing(self))
        main_loop.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            max_brake=self._ego_cap_brake,
        ))
        main_loop.add_child(TimeOut(self._max_run_time))
        root.add_child(main_loop)

        root.add_child(TimeOut(self._post_event_hold, name="PostEventHold"))
        root.add_child(ActorDestroy(self._occluder, name="DestroyOccluder"))
        root.add_child(ActorDestroy(self._dropoff_front, name="DestroyDropoffFront"))
        root.add_child(ActorDestroy(self._dropoff_rear, name="DestroyDropoffRear"))
        root.add_child(ActorDestroy(self._child, name="DestroyChild"))

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
            "SchoolChild_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego, self._occluder, self._dropoff_front, self._dropoff_rear, self._child],
            lane_waypoints=lane_wps,
            center_location=self._trigger_wp.transform.location,
            radius=180.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        return outer

    def _create_test_criteria(self):
        criteria = py_trees.composites.Parallel(
            "SchoolChildCriteria",
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
