"""Parked car reverses into lane (Town03).

A parked car reverses out into the ego's lane.
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
from srunner.scenariomanager.scenarioatomics.custom_atomics_parked_car_reverses_into_lane_town03 import (
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


class _RunReverseManeuver(py_trees.behaviour.Behaviour):
    """
    Wait for ego to approach, then reverse the parked vehicle into ego's path.
    """

    def __init__(self, scenario, name="RunReverseManeuver"):
        super().__init__(name)
        self._scenario = scenario
        self._start_time = None
        self._reverse_started = False
        self._last_log = -999.0

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        scenario = self._scenario
        if scenario._impact_detected:
            print(f"[VehicleReverse] *** COLLISION ***", flush=True)
            return py_trees.common.Status.SUCCESS

        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0

        ego_loc = scenario._ego.get_location()
        parked_loc = scenario._parked_vehicle.get_location() if scenario._parked_vehicle else ego_loc
        dist = _planar_distance(ego_loc, parked_loc)

        # Trigger reverse when ego is close
        if not self._reverse_started and dist <= scenario._trigger_distance:
            self._reverse_started = True
            print(f"[VehicleReverse] Ego within {dist:.1f}m — parked car reverses!", flush=True)

        # Drive the parked vehicle
        parked = scenario._parked_vehicle
        if parked is not None and parked.is_alive:
            if self._reverse_started:
                parked_t = parked.get_transform()
                right = parked_t.get_right_vector()
                fwd = parked_t.get_forward_vector()
                back_speed = scenario._reverse_back_speed
                lat_speed = scenario._reverse_lateral_speed
                vx = -right.x * lat_speed - fwd.x * back_speed
                vy = -right.y * lat_speed - fwd.y * back_speed
                parked.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
                parked.apply_control(carla.VehicleControl(
                    throttle=scenario._reverse_throttle,
                    steer=scenario._reverse_steer,
                    brake=0.0,
                    reverse=True,
                ))
            else:
                parked.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                parked.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=1.0, hand_brake=True,
                ))

        # Status log
        if elapsed - self._last_log >= 1.0:
            self._last_log = elapsed
            ego_spd = _speed_mps(scenario._ego) * 3.6
            parked_spd = _speed_mps(parked) * 3.6 if parked else 0
            print(f"[VehicleReverse] t={elapsed:.1f}s  ego={ego_spd:.0f}km/h  "
                  f"parked={parked_spd:.0f}km/h  dist={dist:.1f}m  "
                  f"reversing={self._reverse_started}  impact={scenario._impact_detected}",
                  flush=True)

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


class ParkedCarReversesIntoLaneTown03(BasicScenario):
    """
    A parked vehicle at roadside suddenly reverses into the ego vehicle's path.

    XML other_parameters:
      activation_distance_m       - trigger proximity
      parked_ahead_distance_m     - parked vehicle distance ahead of ego
      parked_lateral_offset_m     - lateral offset (roadside parking)
      reverse_throttle            - throttle for reverse maneuver (0-1)
      reverse_steer               - steer during reverse (toward lane center)
      trigger_distance_m          - ego distance to parked car to trigger reverse
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

        op = config.other_parameters if hasattr(config, "other_parameters") else {}
        self._activation_distance = self._get_param(op, "activation_distance_m", 80.0)
        self._parked_ahead_distance = self._get_param(op, "parked_ahead_distance_m", 38.0)
        self._parked_lateral_offset = self._get_param(op, "parked_lateral_offset_m", 3.1)
        self._reverse_throttle = self._get_param(op, "reverse_throttle", 0.7)
        self._reverse_steer = self._get_param(op, "reverse_steer", -0.3)
        self._reverse_back_speed = self._get_param(op, "reverse_back_speed_mps", 4.8)
        self._reverse_lateral_speed = self._get_param(op, "reverse_lateral_speed_mps", 1.4)
        self._trigger_distance = self._get_param(op, "trigger_distance_m", 24.0)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 35.0)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 30.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.12)
        self._max_run_time = self._get_param(op, "max_run_time", 20.0)
        self._post_impact_hold = self._get_param(op, "post_impact_hold", 4.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._parked_vehicle = None
        self._collision_sensor = None
        self._impact_detected = False

        super().__init__(
            name="ParkedCarReversesIntoLaneTown03",
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

        # Parked vehicle: ahead of ego, offset to the right (roadside)
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
        parked_rotation = carla.Rotation(
            pitch=parked_t.rotation.pitch,
            yaw=parked_t.rotation.yaw - 12.0,
            roll=parked_t.rotation.roll,
        )
        parked_transform = carla.Transform(parked_loc, parked_rotation)

        parked_models = [
            "vehicle.nissan.micra",
            "vehicle.mini.cooper_s",
            "vehicle.mini.cooper_s_2021",
            "vehicle.volkswagen.t2",
            "vehicle.tesla.model3",
        ]
        self._parked_vehicle = None
        for model in parked_models:
            self._parked_vehicle = CarlaDataProvider.request_new_actor(
                model, parked_transform, rolename="reversing_vehicle",
            )
            if self._parked_vehicle is not None:
                break
        if self._parked_vehicle is None:
            raise RuntimeError("[VehicleReverse] Parked vehicle spawn failed")

        self._parked_vehicle.set_simulate_physics(True)
        self._parked_vehicle.apply_control(carla.VehicleControl(
            throttle=0.0, brake=1.0, hand_brake=True,
        ))
        self._parked_vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.other_actors.append(self._parked_vehicle)

        # Collision sensor
        collision_bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self._ego,
        )
        self._collision_sensor.listen(lambda event: self._on_collision(event))

        print(
            f"\n[VehicleReverse] Spawn\n"
            f"  Ego road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  Parked at ({parked_loc.x:.1f}, {parked_loc.y:.1f}) "
            f"ahead={self._parked_ahead_distance:.1f}m lat={self._parked_lateral_offset:.1f}m\n"
            f"  trigger_dist={self._trigger_distance:.0f}m "
            f"reverse_thr={self._reverse_throttle:.2f} "
            f"reverse_steer={self._reverse_steer:.2f} "
            f"back={self._reverse_back_speed:.1f}m/s lat={self._reverse_lateral_speed:.1f}m/s\n",
            flush=True,
        )

    def _on_collision(self, event):
        other = event.other_actor
        if self._parked_vehicle is not None and other.id != self._parked_vehicle.id:
            print(f"[VehicleReverse] Ignoring non-parked collision: {other.type_id}", flush=True)
            return
        if not self._impact_detected:
            self._impact_detected = True
            print(f"[VehicleReverse] *** COLLISION: {event.other_actor.type_id} ***", flush=True)

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateVehicleReverseCollision",
        )

    def _create_behavior(self):
        root = py_trees.composites.Sequence("VehicleReverseSequence")

        sync = py_trees.composites.Parallel(
            "Phase0_Sync",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
        )
        sync.add_child(HoldUntilEgoMoves(self._parked_vehicle, self._ego))
        root.add_child(sync)

        main_loop = py_trees.composites.Parallel(
            "Phase1_MainLoop",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        main_loop.add_child(_RunReverseManeuver(self))
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
        root.add_child(ActorDestroy(self._parked_vehicle, name="DestroyParkedVehicle"))

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
            "VehicleReverse_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego, self._parked_vehicle],
            lane_waypoints=lane_wps,
            center_location=self._trigger_wp.transform.location,
            radius=180.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        return outer

    def _create_test_criteria(self):
        criteria = py_trees.composites.Parallel(
            "VehicleReverseCriteria",
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
