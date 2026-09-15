"""Parked car door opens into lane (Town03).

A parked car's door swings open into the ego's path.
"""

import math

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
)
from srunner.scenariomanager.scenarioatomics.custom_atomics_parked_car_door_opens_into_lane_town03 import (
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


def _planar_distance(loc_a, loc_b):
    return math.hypot(loc_a.x - loc_b.x, loc_a.y - loc_b.y)


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    """Override autopilot braking — force ego to maintain minimum speed."""

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


class _RunDoorOpenManeuver(py_trees.behaviour.Behaviour):
    """
    When ego is near the parked car, open the door facing the traffic lane
    while the parked vehicle remains stationary.
    """

    def __init__(self, scenario, name="RunDoorOpenManeuver"):
        super().__init__(name)
        self._scenario = scenario
        self._start_time = None
        self._door_opened = False
        self._last_log = -999.0

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        scenario = self._scenario
        if scenario._impact_detected:
            print(f"[DoorOpen] *** DOOR COLLISION ***", flush=True)
            return py_trees.common.Status.SUCCESS

        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0

        ego_loc = scenario._ego.get_location()
        parked_loc = scenario._parked_car.get_location() if scenario._parked_car else ego_loc
        dist = _planar_distance(ego_loc, parked_loc)

        if not self._door_opened and dist <= scenario._door_trigger_distance:
            self._door_opened = True
            scenario._show_open_door_panel()
            print(f"[DoorOpen] Ego within {dist:.1f}m — DOOR OPENS!", flush=True)

        parked = scenario._parked_car
        if parked is not None and parked.is_alive:
            if self._door_opened:
                parked.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                parked.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=1.0, hand_brake=True,
                ))
                try:
                    parked.open_door(carla.VehicleDoor.FL)
                except RuntimeError:
                    pass
                try:
                    parked.open_door(carla.VehicleDoor.RL)
                except RuntimeError:
                    pass
            else:
                parked.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                parked.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=1.0, hand_brake=True,
                ))
                # Force doors closed each tick until the trigger fires so the
                # car is visibly intact during the approach.
                try:
                    parked.close_door(carla.VehicleDoor.All)
                except RuntimeError:
                    pass

        if elapsed - self._last_log >= 1.0:
            self._last_log = elapsed
            ego_spd = _speed_mps(scenario._ego) * 3.6
            print(f"[DoorOpen] t={elapsed:.1f}s  ego={ego_spd:.0f}km/h  "
                  f"dist={dist:.1f}m  door={self._door_opened}  "
                  f"impact={scenario._impact_detected}", flush=True)

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


class _DestroyDoorOpenActors(py_trees.behaviour.Behaviour):
    def __init__(self, scenario, name="DestroyDoorOpenActors"):
        super().__init__(name)
        self._scenario = scenario

    def update(self):
        for attr_name in ("_parked_car", "_door_proxy"):
            actor = getattr(self._scenario, attr_name, None)
            if actor is not None and actor.is_alive:
                try:
                    CarlaDataProvider.remove_actor_by_id(actor.id)
                except RuntimeError:
                    try:
                        actor.destroy()
                    except RuntimeError:
                        pass
            setattr(self._scenario, attr_name, None)
        return py_trees.common.Status.SUCCESS


class ParkedCarDoorOpensIntoLaneTown03(BasicScenario):
    """
    A parked car's door suddenly opens into the lane as the ego vehicle passes.

    XML other_parameters:
      activation_distance_m       - trigger proximity
      parked_ahead_distance_m     - parked car offset ahead of ego
      parked_lateral_offset_m     - lateral offset (roadside parking)
      door_trigger_distance_m     - ego distance to parked car that triggers door
      ego_speed_cap_kmh           - ego speed governor cap
      ego_cap_brake               - brake force when above cap
      max_run_time                - scenario timeout
      post_impact_hold            - hold duration after collision
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

        op = dict(config.other_parameters) if hasattr(config, "other_parameters") else {}
        if hasattr(config, "scenario") and hasattr(config.scenario, "other_parameters"):
            op.update(config.scenario.other_parameters)
        self._activation_distance = self._get_param(op, "activation_distance_m", 90.0)
        self._parked_ahead_distance = self._get_param(op, "parked_ahead_distance_m", 70.0)
        self._parked_lateral_offset = self._get_param(op, "parked_lateral_offset_m", 2.4)
        self._door_trigger_distance = self._get_param(op, "door_trigger_distance_m", 10.0)
        self._door_proxy_lateral_offset = self._get_param(
            op, "door_proxy_lateral_offset_m", 1.35)
        self._door_proxy_forward_offset = self._get_param(
            op, "door_proxy_forward_offset_m", -0.35)
        self._door_proxy_yaw = self._get_param(
            op, "door_proxy_yaw_deg", 70.0)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 35.0)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 30.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.12)
        self._max_run_time = self._get_param(op, "max_run_time", 20.0)
        self._post_impact_hold = self._get_param(op, "post_impact_hold", 4.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._parked_car = None
        self._door_proxy = None
        self._collision_sensor = None
        self._impact_detected = False

        super().__init__(
            name="ParkedCarDoorOpensIntoLaneTown03",
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

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        parked_wp = self._walk_waypoint(
            ego_wp, self._parked_ahead_distance, forward=True
        ) or ego_wp
        parked_t = parked_wp.transform
        right = parked_t.get_right_vector()
        parked_loc = carla.Location(
            x=parked_t.location.x + right.x * self._parked_lateral_offset,
            y=parked_t.location.y + right.y * self._parked_lateral_offset,
            z=parked_t.location.z + 0.3,
        )
        parked_transform = carla.Transform(parked_loc, parked_t.rotation)

        car_models = [
            "vehicle.lincoln.mkz_2020",
            "vehicle.lincoln.mkz_2017",
            "vehicle.toyota.prius",
            "vehicle.nissan.micra",
            "vehicle.tesla.model3",
            "vehicle.mini.cooper_s_2021",
            "vehicle.mini.cooper_s",
        ]
        self._parked_car = None
        for model in car_models:
            self._parked_car = CarlaDataProvider.request_new_actor(
                model, parked_transform, rolename="parked_car_door",
            )
            if self._parked_car is not None:
                break
        if self._parked_car is None:
            raise RuntimeError("[DoorOpen] Parked car spawn failed")

        self._parked_car.set_simulate_physics(True)
        self._parked_car.apply_control(carla.VehicleControl(
            throttle=0.0, brake=1.0, hand_brake=True,
        ))
        try:
            self._parked_car.close_door(carla.VehicleDoor.All)
        except RuntimeError:
            pass
        self._parked_car.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self._parked_car.set_collisions(False)
        self.other_actors.append(self._parked_car)

        # Collision sensor
        collision_bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self._ego,
        )
        self._collision_sensor.listen(lambda event: self._on_collision(event))

        print(
            f"\n[DoorOpen] Spawn\n"
            f"  Ego road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  Parked car at ({parked_loc.x:.1f}, {parked_loc.y:.1f}) "
            f"ahead={self._parked_ahead_distance:.1f}m lat={self._parked_lateral_offset:.2f}m\n"
            f"  door_trigger={self._door_trigger_distance:.0f}m  "
            f"opens=real FL+RL doors; collision=independent RL door mesh\n",
            flush=True,
        )

    def _spawn_door_proxy(self):
        if self._door_proxy is not None and self._door_proxy.is_alive:
            return

        car_transform = self._parked_car.get_transform()
        right = car_transform.get_right_vector()
        left = carla.Vector3D(-right.x, -right.y, -right.z)
        forward = car_transform.get_forward_vector()
        proxy_location = carla.Location(
            x=car_transform.location.x
            + left.x * self._door_proxy_lateral_offset
            + forward.x * self._door_proxy_forward_offset,
            y=car_transform.location.y
            + left.y * self._door_proxy_lateral_offset
            + forward.y * self._door_proxy_forward_offset,
            z=car_transform.location.z + 0.55,
        )
        proxy_rotation = carla.Rotation(
            pitch=0.0,
            yaw=car_transform.rotation.yaw + self._door_proxy_yaw,
            roll=0.0,
        )
        mesh_bp = self._world.get_blueprint_library().find("static.prop.mesh")
        mesh_bp.set_attribute(
            "mesh_path",
            "/Game/Carla/Static/Car/4Wheeled/LincolnMKZ2020/"
            "SM_LincolnDoor_Back_L.SM_LincolnDoor_Back_L",
        )
        mesh_bp.set_attribute("scale", "1.0")
        self._door_proxy = self._world.spawn_actor(
            mesh_bp, carla.Transform(proxy_location, proxy_rotation),
        )
        if self._door_proxy is None:
            raise RuntimeError("[DoorOpen] Door proxy spawn failed")
        self.other_actors.append(self._door_proxy)
        print(
            f"[DoorOpen] Independent door proxy spawned at "
            f"({proxy_location.x:.1f}, {proxy_location.y:.1f}) "
            f"type={self._door_proxy.type_id}",
            flush=True,
        )

    def _show_open_door_panel(self):
        self._spawn_door_proxy()
        for door in (carla.VehicleDoor.FL, carla.VehicleDoor.RL):
            try:
                self._parked_car.open_door(door)
            except RuntimeError as exc:
                print(f"[DoorOpen] Door open failed ({door}): {exc}", flush=True)
        print("[DoorOpen] Parked-car left-side doors opened toward ego lane", flush=True)

    def _on_collision(self, event):
        other = event.other_actor
        valid_ids = set()
        if self._door_proxy is not None:
            valid_ids.add(self._door_proxy.id)
        if other.id not in valid_ids:
            print(f"[DoorOpen] Ignoring non-door collision: {other.type_id}", flush=True)
            return
        if not self._impact_detected:
            self._impact_detected = True
            if self._door_proxy is not None and self._door_proxy.is_alive:
                try:
                    self._door_proxy.set_collisions(False)
                except RuntimeError:
                    pass
            print(
                f"[DoorOpen] *** OPEN DOOR COLLISION: "
                f"{event.other_actor.type_id} id={event.other_actor.id} ***",
                flush=True,
            )

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateDoorOpenCollision",
        )

    def _create_behavior(self):
        root = py_trees.composites.Sequence("DoorOpenSequence")

        sync = py_trees.composites.Parallel(
            "Phase0_Sync",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
        )
        sync.add_child(HoldUntilEgoMoves(self._parked_car, self._ego))
        root.add_child(sync)

        main_loop = py_trees.composites.Parallel(
            "Phase1_MainLoop",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        main_loop.add_child(_RunDoorOpenManeuver(self))
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
        root.add_child(_DestroyDoorOpenActors(self))

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
            "DoorOpen_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego, self._parked_car],
            lane_waypoints=lane_wps,
            center_location=self._trigger_wp.transform.location,
            radius=180.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        return outer

    def _create_test_criteria(self):
        criteria = py_trees.composites.Parallel(
            "DoorOpenCriteria",
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
