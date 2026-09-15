"""Pedestrian emerges from behind van (Town03).

A pedestrian steps out from behind a parked van into the ego's path.
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
from srunner.scenariomanager.scenarioatomics.custom_atomics_pedestrian_emerges_from_behind_van_town03 import (
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


class _RunPedestrianDash(py_trees.behaviour.Behaviour):
    """
    Wait for ego to be near the parked van, then command the pedestrian
    to sprint across the road.
    """

    def __init__(self, scenario, name="RunPedestrianDash"):
        super().__init__(name)
        self._scenario = scenario
        self._start_time = None
        self._dash_started = False
        self._dash_start_time = None
        self._last_move_time = None
        self._crossing_target = None
        self._crossing_waypoints = []
        self._crossing_waypoint_index = 0
        self._last_log = -999.0

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        scenario = self._scenario
        if scenario._impact_detected:
            print(f"[HiddenPed] *** PEDESTRIAN HIT ***", flush=True)
            return py_trees.common.Status.SUCCESS

        elapsed = GameTime.get_time() - self._start_time if self._start_time else 0.0

        ego_loc = scenario._ego.get_location()
        van_loc = scenario._van.get_location() if scenario._van else ego_loc
        dist_to_van = _planar_distance(ego_loc, van_loc)

        # Trigger pedestrian dash when ego is close to the van. The pedestrian
        # starts from the hidden spawn point behind the parked van.
        if (not self._dash_started
                and dist_to_van <= scenario._trigger_distance_to_van
                and elapsed >= scenario._dash_min_elapsed_s):
            self._dash_started = True
            self._dash_start_time = elapsed
            self._last_move_time = elapsed
            # Move around the van's front first, then cross into the ego lane.
            # The pedestrian reaches the lane center before the ego arrives and
            # waits there, making the intended vulnerable-road-user impact clear.
            van_tf = scenario._van.get_transform()
            van_fwd = van_tf.get_forward_vector()
            van_right = van_tf.get_right_vector()
            front_clearance = scenario._pedestrian_front_clearance_m
            side_front = carla.Location(
                van_loc.x + van_fwd.x * front_clearance +
                van_right.x * scenario._ped_lateral_offset,
                van_loc.y + van_fwd.y * front_clearance +
                van_right.y * scenario._ped_lateral_offset,
                van_loc.z + 0.5,
            )
            lane_center = carla.Location(
                van_loc.x - van_right.x * (
                    scenario._van_lateral_offset +
                    scenario._pedestrian_target_lateral_m
                ),
                van_loc.y - van_right.y * (
                    scenario._van_lateral_offset +
                    scenario._pedestrian_target_lateral_m
                ),
                van_loc.z + 0.5,
            )
            self._crossing_target = carla.Location(
                lane_center.x + van_fwd.x * front_clearance,
                lane_center.y + van_fwd.y * front_clearance,
                lane_center.z,
            )
            self._crossing_waypoints = [side_front, self._crossing_target]
            self._crossing_waypoint_index = 0
            print(f"[HiddenPed] Ego within {dist_to_van:.1f}m of van — "
                  f"pedestrian starts crossing!", flush=True)

        # Drive the pedestrian
        ped = scenario._pedestrian
        if ped is not None and ped.is_alive:
            if self._dash_started:
                ped_loc = ped.get_location()
                target = self._crossing_target or ped_loc
                if self._crossing_waypoint_index < len(self._crossing_waypoints):
                    waypoint = self._crossing_waypoints[self._crossing_waypoint_index]
                    if _planar_distance(ped_loc, waypoint) <= 0.3:
                        self._crossing_waypoint_index += 1
                    if self._crossing_waypoint_index < len(self._crossing_waypoints):
                        target = self._crossing_waypoints[self._crossing_waypoint_index]
                dx = target.x - ped_loc.x
                dy = target.y - ped_loc.y
                mag = math.sqrt(dx * dx + dy * dy)
                dt = max(0.03, min(0.1, elapsed - self._last_move_time))
                self._last_move_time = elapsed
                if mag > 0.3:
                    step = min(mag, scenario._pedestrian_dash_speed * dt)
                    ped.set_location(carla.Location(
                        ped_loc.x + dx / mag * step,
                        ped_loc.y + dy / mag * step,
                        ped_loc.z,
                    ))
                    ped.apply_control(carla.WalkerControl(
                        direction=carla.Vector3D(dx / mag, dy / mag, 0.0),
                        speed=scenario._pedestrian_dash_speed,
                    ))
                else:
                    ped.apply_control(carla.WalkerControl(
                        direction=carla.Vector3D(0.0, 0.0, 0.0),
                        speed=0.0,
                    ))
            else:
                # Pedestrian stands still behind van
                ped.apply_control(carla.WalkerControl(
                    direction=carla.Vector3D(0.0, 0.0, 0.0),
                    speed=0.0,
                ))

        # Status log
        if elapsed - self._last_log >= 1.0:
            self._last_log = elapsed
            ego_spd = _speed_mps(scenario._ego) * 3.6
            print(f"[HiddenPed] t={elapsed:.1f}s  ego={ego_spd:.0f}km/h  "
                  f"dist_van={dist_to_van:.1f}m  dash={self._dash_started}  "
                  f"impact={scenario._impact_detected}", flush=True)

        if elapsed >= scenario._max_run_time:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING


class _PostImpactHold(py_trees.behaviour.Behaviour):
    """Hold the crash scene for video."""

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
    """Override autopilot braking — force ego to maintain minimum speed."""

    def __init__(self, ego, min_speed_kmh=40.0, throttle=0.8,
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


class PedestrianEmergesFromBehindVanTown03(BasicScenario):
    """
    A pedestrian hidden behind a parked van suddenly dashes across the road
    in front of the ego vehicle on an urban street.

    XML other_parameters:
      activation_distance_m         - trigger proximity
      van_ahead_distance_m          - parked van offset ahead of ego
      van_lateral_offset_m          - van lateral offset from ego's lane center
      pedestrian_behind_van_offset_m - ped offset behind van (along road)
      pedestrian_lateral_offset_m   - ped offset from van toward sidewalk
      pedestrian_dash_speed_mps     - ped sprint speed (m/s)
      trigger_distance_to_van_m     - ego distance to van that triggers dash
      dash_direction_sign           - +1.0 or -1.0 for dash direction
      ego_speed_cap_kmh             - ego speed governor cap
      ego_cap_brake                 - brake force when above cap
      max_run_time                  - scenario timeout
      post_impact_hold              - hold duration after collision
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
        self._van_ahead_distance = self._get_param(op, "van_ahead_distance_m", 55.0)
        self._van_lateral_offset = self._get_param(op, "van_lateral_offset_m", 3.2)
        self._ped_behind_van_offset = self._get_param(op, "pedestrian_behind_van_offset_m", 2.0)
        self._ped_lateral_offset = self._get_param(op, "pedestrian_lateral_offset_m", 3.2)
        self._pedestrian_dash_speed = self._get_param(op, "pedestrian_dash_speed_mps", 6.5)
        self._pedestrian_target_ahead_m = self._get_param(op, "pedestrian_target_ahead_m", 15.0)
        self._pedestrian_target_lateral_m = self._get_param(op, "pedestrian_target_lateral_m", 0.0)
        self._pedestrian_intercept_ahead_m = self._get_param(op, "pedestrian_intercept_ahead_m", 12.0)
        self._pedestrian_front_clearance_m = self._get_param(op, "pedestrian_front_clearance_m", 4.5)
        self._dash_min_elapsed_s = self._get_param(op, "dash_min_elapsed_s", 0.0)
        self._trigger_distance_to_van = self._get_param(op, "trigger_distance_to_van_m", 22.0)
        self._dash_direction_sign = self._get_param(op, "dash_direction_sign", 1.0)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 45.0)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 40.0)
        self._ego_force_throttle = self._get_param(op, "ego_force_throttle", 0.8)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.15)
        self._max_run_time = self._get_param(op, "max_run_time", 20.0)
        self._post_impact_hold = self._get_param(op, "post_impact_hold", 4.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._van = None
        self._pedestrian = None
        self._collision_sensor = None
        self._impact_detected = False

        super().__init__(
            name="PedestrianEmergesFromBehindVanTown03",
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

        # Van location: ahead of ego, offset to the right (parked at roadside)
        van_wp = self._walk_waypoint(ego_wp, self._van_ahead_distance, forward=True) or ego_wp
        van_t = van_wp.transform
        right = van_t.get_right_vector()
        van_loc = carla.Location(
            x=van_t.location.x + right.x * self._van_lateral_offset,
            y=van_t.location.y + right.y * self._van_lateral_offset,
            z=van_t.location.z + 0.3,
        )
        van_transform = carla.Transform(van_loc, van_t.rotation)

        # Try van-like models
        van_models = [
            "vehicle.volkswagen.t2",
            "vehicle.volkswagen.t2_2021",
            "vehicle.carlamotors.carlacola",
            "vehicle.tesla.cybertruck",
        ]
        self._van = None
        for model in van_models:
            self._van = CarlaDataProvider.request_new_actor(
                model, van_transform, rolename="parked_van",
            )
            if self._van is not None:
                break
        if self._van is None:
            raise RuntimeError("[HiddenPed] Van spawn failed")

        # Park the van (brake on, no movement)
        self._van.set_simulate_physics(True)
        self._van.apply_control(carla.VehicleControl(
            throttle=0.0, brake=1.0, hand_brake=True,
        ))
        self._van.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self._van.set_collisions(False)
        self.other_actors.append(self._van)

        # Pedestrian: hidden behind the van (further from road center)
        fwd = van_t.get_forward_vector()
        ped_loc = carla.Location(
            x=van_loc.x - fwd.x * self._ped_behind_van_offset + right.x * self._ped_lateral_offset,
            y=van_loc.y - fwd.y * self._ped_behind_van_offset + right.y * self._ped_lateral_offset,
            z=van_loc.z + 0.5,
        )
        ped_transform = carla.Transform(ped_loc, van_t.rotation)

        ped_models = [
            "walker.pedestrian.0013",
            "walker.pedestrian.0014",
            "walker.pedestrian.0001",
        ]
        self._pedestrian = None
        for model in ped_models:
            self._pedestrian = CarlaDataProvider.request_new_actor(
                model, ped_transform, rolename="hidden_pedestrian",
            )
            if self._pedestrian is not None:
                break
        if self._pedestrian is None:
            raise RuntimeError("[HiddenPed] Pedestrian spawn failed")
        self.other_actors.append(self._pedestrian)

        # Attach collision sensor to ego
        collision_bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self._ego,
        )
        self._collision_sensor.listen(lambda event: self._on_collision(event))

        print(
            f"\n[HiddenPed] Spawn\n"
            f"  Ego road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  Van at ({van_loc.x:.1f}, {van_loc.y:.1f}) "
            f"ahead={self._van_ahead_distance:.1f}m lat_off={self._van_lateral_offset:.1f}m\n"
            f"  Pedestrian at ({ped_loc.x:.1f}, {ped_loc.y:.1f}) "
            f"dash_speed={self._pedestrian_dash_speed:.1f}m/s "
            f"trigger_dist={self._trigger_distance_to_van:.1f}m\n",
            flush=True,
        )

    def _on_collision(self, event):
        other = event.other_actor
        if self._pedestrian is not None and other.id != self._pedestrian.id:
            print(f"[HiddenPed] Ignoring non-pedestrian collision: {other.type_id}", flush=True)
            return
        if not self._impact_detected:
            self._impact_detected = True
            print(f"[HiddenPed] *** COLLISION: {event.other_actor.type_id} ***", flush=True)

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateHiddenPedestrianCrossing",
        )

    def _create_behavior(self):
        root = py_trees.composites.Sequence("HiddenPedestrianSequence")

        # Phase 0: wait for ego to start moving
        sync = py_trees.composites.Parallel(
            "Phase0_Sync",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
        )
        sync.add_child(HoldUntilEgoMoves(self._van, self._ego))
        root.add_child(sync)

        # Phase 1: main scenario (ego approaches, ped dashes)
        main_loop = py_trees.composites.Parallel(
            "Phase1_MainLoop",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        main_loop.add_child(_RunPedestrianDash(self))
        main_loop.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            max_brake=self._ego_cap_brake,
        ))
        main_loop.add_child(_EgoMinSpeedForcer(
            self._ego,
            min_speed_kmh=self._ego_min_speed_kmh,
            throttle=self._ego_force_throttle,
        ))
        main_loop.add_child(TimeOut(self._max_run_time))
        root.add_child(main_loop)

        # Phase 2: post-impact hold
        root.add_child(_PostImpactHold(self._post_impact_hold))

        # Cleanup
        root.add_child(ActorDestroy(self._van, name="DestroyVan"))
        root.add_child(ActorDestroy(self._pedestrian, name="DestroyPedestrian"))

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
            "HiddenPedestrian_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego, self._van, self._pedestrian],
            lane_waypoints=lane_wps,
            center_location=self._trigger_wp.transform.location,
            radius=180.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        return outer

    def _create_test_criteria(self):
        criteria = py_trees.composites.Parallel(
            "HiddenPedestrianCriteria",
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
