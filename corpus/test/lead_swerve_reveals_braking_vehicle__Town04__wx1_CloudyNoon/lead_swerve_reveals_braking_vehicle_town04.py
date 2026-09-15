"""Lead swerve reveals braking vehicle (Town04).

A lead vehicle changes lane to reveal a hard-braking vehicle stopped in the ego's lane.
"""

import math
import os
import xml.etree.ElementTree as ET

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
)
from srunner.scenariomanager.scenarioatomics.custom_atomics_lead_swerve_reveals_braking_vehicle_town04 import (
    EgoSpeedGovernor,
    LaneCleaner,
    StopActorsOnEgoCollision,
)
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenarios.basic_scenario import BasicScenario


def _speed_mps(actor):
    if actor is None or not actor.is_alive:
        return 0.0
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _planar_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    def __init__(self, ego, min_speed_kmh=0.0, throttle=0.75, scenario=None, name="EgoMinSpeedForcer"):
        super().__init__(name)
        self._ego = ego
        self._min_mps = float(min_speed_kmh) / 3.6
        self._throttle = float(throttle)
        self._scenario = scenario

    def update(self):
        if self._ego is None or not self._ego.is_alive:
            return py_trees.common.Status.RUNNING
        # Release the speed floor as soon as the hazard slams on its brakes so the
        # ego is free to brake straight to a near-miss instead of being forced
        # forward (which causes an off-road swerve or a rear-end collision).
        if self._scenario is not None and getattr(self._scenario, "_hazard_brake_triggered", False):
            return py_trees.common.Status.RUNNING
        if _speed_mps(self._ego) < self._min_mps:
            control = self._ego.get_control()
            control.brake = 0.0
            control.throttle = self._throttle
            self._ego.apply_control(control)
        return py_trees.common.Status.RUNNING


class _RunLeadSwerveReveal(py_trees.behaviour.Behaviour):
    def __init__(self, scenario, name="RunLeadSwerveReveal"):
        super().__init__(name)
        self._scenario = scenario
        self._start_time = None
        self._last_time = None
        self._last_log = -999.0

    def initialise(self):
        now = GameTime.get_time()
        self._start_time = now
        self._last_time = now

    def update(self):
        scenario = self._scenario
        now = GameTime.get_time()
        elapsed = 0.0 if self._start_time is None else max(0.0, now - self._start_time)
        dt = 0.05 if self._last_time is None else max(0.02, min(0.10, now - self._last_time))
        self._last_time = now

        scenario._step_scene(elapsed, dt)

        if elapsed - self._last_log >= 0.75:
            self._last_log = elapsed
            ego_loc = scenario._ego.get_location()
            lead_gap = 0.0 if scenario._lead is None else _planar_distance(ego_loc, scenario._lead["loc"])
            hazard_gap = 0.0 if scenario._hazard is None else _planar_distance(ego_loc, scenario._hazard["loc"])
            print(
                f"[LeadSwerveReveal] t={elapsed:.1f}s ego={_speed_mps(scenario._ego) * 3.6:.0f}km/h "
                f"lead_gap={lead_gap:.1f}m hazard_gap={hazard_gap:.1f}m "
                f"lead_lat={scenario._lead_lateral_offset:.1f} hazard_stop={scenario._hazard_stopped} "
                f"impact={scenario._impact_detected}",
                flush=True,
            )

        if scenario._impact_detected:
            print("[LeadSwerveReveal] *** IMPACT DETECTED ***", flush=True)
            return py_trees.common.Status.SUCCESS
        if elapsed >= scenario._max_run_time:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class _PostImpactHold(py_trees.behaviour.Behaviour):
    def __init__(self, scenario, duration, name="PostImpactHold"):
        super().__init__(name)
        self._scenario = scenario
        self._duration = float(duration)
        self._start_time = None

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        self._scenario._hold_scene()
        elapsed = 0.0 if self._start_time is None else GameTime.get_time() - self._start_time
        if elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class LeadSwerveRevealsBrakingVehicleTown04(BasicScenario):
    """Occluded braking-chain highway scenario with a lead-car swerve reveal."""

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

    @classmethod
    def _flatten_params(cls, value):
        params = {}
        if isinstance(value, dict):
            if "value" in value:
                return params
            for key, item in value.items():
                if isinstance(item, dict) and "value" in item:
                    params[key] = item
                elif hasattr(item, "value"):
                    params[key] = {"value": item.value}
                else:
                    params.update(cls._flatten_params(item))
        elif hasattr(value, "items"):
            params.update(cls._flatten_params(dict(value.items())))
        return params

    @staticmethod
    def _params_from_route_xml(route_name, scenario_type):
        if not route_name:
            return {}
        if isinstance(route_name, (list, tuple)):
            route_name = route_name[0] if route_name else ""
        if not isinstance(route_name, str):
            route_name = str(route_name)
        route_file = route_name if route_name.endswith(".xml") else f"{route_name}.xml"
        candidates = [
            os.environ.get("CUSTOM_SCENARIO_ROUTE_XML", ""),
            os.path.join(os.getcwd(), route_file),
            os.path.join(os.getcwd(), "leaderboard", "data", route_file),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                root = ET.parse(path).getroot()
            except ET.ParseError:
                continue
            for scenario in root.findall(".//scenario"):
                if scenario.attrib.get("type") != scenario_type:
                    continue
                params = {}
                other = scenario.find("other_parameters")
                if other is not None:
                    for child in other:
                        params[child.tag] = {"value": child.attrib.get("value", child.text)}
                return params
        return {}

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

    @staticmethod
    def _vector_add(location, vector, scale):
        return carla.Location(
            x=location.x + vector.x * scale,
            y=location.y + vector.y * scale,
            z=location.z + vector.z * scale,
        )

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._ego = ego_vehicles[0]
        self._actors = []
        self._lead = None
        self._hazard = None
        self._left_front = None
        self._left_rear = None
        self._right_front = None
        self._right_rear = None
        self._follower = None
        self._collision_sensor = None

        op = {}
        for source in (config, getattr(config, "scenario", None)):
            if source is not None and hasattr(source, "other_parameters"):
                op.update(self._flatten_params(source.other_parameters))
        if not op:
            op.update(self._params_from_route_xml(getattr(config, "route", ""), "LeadSwerveRevealsBrakingVehicleTown04"))

        self._activation_distance = self._get_param(op, "activation_distance_m", 130.0)
        self._lead_distance = self._get_param(op, "lead_vehicle_ahead_distance_m", 22.0)
        self._hazard_distance = self._get_param(op, "hazard_vehicle_ahead_distance_m", 43.0)
        self._follower_distance = self._get_param(op, "following_vehicle_behind_distance_m", 24.0)
        self._side_traffic_ahead = self._get_param(op, "side_traffic_ahead_m", 30.0)
        self._side_traffic_behind = self._get_param(op, "side_traffic_behind_m", 12.0)
        self._lead_speed = self._get_param(op, "lead_speed_kmh", 70.0) / 3.6
        self._hazard_initial_speed = self._get_param(op, "hazard_initial_speed_kmh", 48.0) / 3.6
        self._hazard_stop_speed = self._get_param(op, "hazard_stop_speed_kmh", 0.0) / 3.6
        self._side_speed = self._get_param(op, "side_traffic_speed_kmh", 66.0) / 3.6
        self._follower_speed = self._get_param(op, "following_vehicle_speed_kmh", 74.0) / 3.6
        self._hazard_brake_start_s = self._get_param(op, "hazard_brake_start_s", 0.25)
        self._hazard_brake_trigger_gap_m = self._get_param(op, "hazard_brake_trigger_gap_m", 60.0)
        self._hazard_brake_ramp_s = self._get_param(op, "hazard_brake_ramp_s", 0.9)
        self._hazard_brake_max_s = self._get_param(op, "hazard_brake_max_s", 6.5)
        self._lead_follow_gap_m = self._get_param(op, "lead_follow_gap_m", 18.0)
        self._lead_swerve_delay_after_brake_s = self._get_param(op, "lead_swerve_delay_after_brake_s", 0.6)
        self._lead_swerve_start_s = self._get_param(op, "lead_swerve_start_s", 2.2)
        self._lead_swerve_duration_s = self._get_param(op, "lead_swerve_duration_s", 0.85)
        self._lead_swerve_lateral_m = self._get_param(op, "lead_swerve_lateral_m", 3.2)
        self._lead_swerve_speed_kmh = self._get_param(op, "lead_swerve_speed_kmh", 76.0) / 3.6
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 72.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.06)
        self._ego_min_speed_kmh = self._get_param(op, "ego_min_speed_kmh", 64.0)
        self._ego_force_throttle = self._get_param(op, "ego_force_throttle", 0.95)
        self._max_run_time = self._get_param(op, "max_run_time", 12.0)
        self._post_impact_hold = self._get_param(op, "post_impact_hold", 5.0)
        self._road_friction_scale = self._get_param(op, "road_friction_scale", 1.0)

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._impact_detected = False
        self._lead_lateral_offset = 0.0
        self._hazard_stopped = False
        self._hazard_brake_triggered = False
        self._hazard_brake_time = None
        self._hazard_brake_from_speed = 0.0

        super().__init__(
            name="LeadSwerveRevealsBrakingVehicleTown04",
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
            world.spawn_actor(bp, carla.Transform(carla.Location(-10000.0, -10000.0, 0.0)))
        except RuntimeError:
            pass

    def _spawn_vehicle(self, models, transform, rolename, color=None, physics=False):
        for model in models:
            actor = CarlaDataProvider.request_new_actor(
                model, transform, rolename=rolename, autopilot=False, color=color,
            )
            if actor is not None:
                actor.set_simulate_physics(physics)
                if not physics:
                    actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                self.other_actors.append(actor)
                self._actors.append(actor)
                return actor
        return None

    def _make_pose(self, actor, waypoint, lateral_offset=0.0):
        loc = carla.Location(
            x=waypoint.transform.location.x,
            y=waypoint.transform.location.y,
            z=waypoint.transform.location.z + 0.35,
        )
        if abs(lateral_offset) > 0.01:
            loc = self._vector_add(loc, waypoint.transform.get_right_vector(), lateral_offset)
        return {
            "actor": actor,
            "loc": loc,
            "forward": waypoint.transform.get_forward_vector(),
            "right": waypoint.transform.get_right_vector(),
            "yaw": waypoint.transform.rotation.yaw,
        }

    def _spawn_pose_actor(self, waypoint, models, rolename, color=None, lateral_offset=0.0):
        loc = carla.Location(
            x=waypoint.transform.location.x,
            y=waypoint.transform.location.y,
            z=waypoint.transform.location.z + 0.35,
        )
        if abs(lateral_offset) > 0.01:
            loc = self._vector_add(loc, waypoint.transform.get_right_vector(), lateral_offset)
        actor = self._spawn_vehicle(models, carla.Transform(loc, waypoint.transform.rotation), rolename, color=color)
        if actor is None:
            raise RuntimeError(f"[LeadSwerveReveal] Failed to spawn {rolename}")
        return self._make_pose(actor, waypoint, lateral_offset=lateral_offset)

    def _attach_collision_sensor(self):
        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(bp, carla.Transform(), attach_to=self._ego)
        self.other_actors.append(self._collision_sensor)

        def _on_collision(event):
            if event.other_actor is not None and event.other_actor.type_id.startswith("vehicle."):
                self._impact_detected = True

        self._collision_sensor.listen(_on_collision)

    def _initialize_actors(self, config):
        # Anchor the convoy to the EGO's real spawn position (not the trigger,
        # which sits ~40m ahead). This lets the lead and hazard spawn at a tight
        # 1-2s following gap so they can ride alongside the ego, accelerate up to
        # highway speed together, and then have the hazard SLAM its brakes from
        # full speed right in front of the camera -- a visibly hard brake, not a
        # slow car rolling to a stop.
        ego_wp = self._map.get_waypoint(self._ego.get_location()) or self._trigger_wp
        left_wp = ego_wp.get_left_lane()
        right_wp = ego_wp.get_right_lane()
        if left_wp is not None and left_wp.lane_type != carla.LaneType.Driving:
            left_wp = None
        if right_wp is not None and right_wp.lane_type != carla.LaneType.Driving:
            right_wp = None

        lead_wp = self._walk_waypoint(ego_wp, self._lead_distance, forward=True) or ego_wp
        hazard_wp = self._walk_waypoint(ego_wp, self._hazard_distance, forward=True) or lead_wp
        follower_wp = None

        self._lead = self._spawn_pose_actor(
            lead_wp,
            ["vehicle.audi.tt", "vehicle.lincoln.mkz_2020", "vehicle.tesla.model3"],
            "lead_occluder_swerve",
            color="35,95,210",
        )
        self._hazard = self._spawn_pose_actor(
            hazard_wp,
            ["vehicle.nissan.patrol", "vehicle.tesla.model3", "vehicle.lincoln.mkz_2017"],
            "hard_braking_hidden_vehicle",
            color="245,245,245",
        )
        if self._follower_distance > 0.0 and follower_wp is not None:
            self._follower = self._spawn_pose_actor(
                follower_wp,
                ["vehicle.audi.tt", "vehicle.tesla.model3"],
                "rear_close_follower",
                color="210,40,40",
            )

        if left_wp is not None:
            self._left_front = self._spawn_pose_actor(
                self._walk_waypoint(left_wp, self._side_traffic_ahead, forward=True) or left_wp,
                ["vehicle.carlamotors.carlacola", "vehicle.tesla.cybertruck", "vehicle.nissan.patrol"],
                "left_lane_dense_front",
                color="190,190,190",
            )
            self._left_rear = self._spawn_pose_actor(
                self._walk_waypoint(left_wp, self._side_traffic_behind, forward=False) or left_wp,
                ["vehicle.tesla.model3", "vehicle.audi.tt"],
                "left_lane_dense_rear",
                color="80,80,95",
            )
        if right_wp is not None:
            self._right_front = self._spawn_pose_actor(
                self._walk_waypoint(right_wp, self._side_traffic_ahead + 6.0, forward=True) or right_wp,
                ["vehicle.carlamotors.european_hgv", "vehicle.carlamotors.carlacola", "vehicle.nissan.patrol"],
                "right_lane_dense_front",
                color="70,100,140",
            )
            self._right_rear = self._spawn_pose_actor(
                self._walk_waypoint(right_wp, self._side_traffic_behind + 4.0, forward=False) or right_wp,
                ["vehicle.lincoln.mkz_2020", "vehicle.tesla.model3"],
                "right_lane_dense_rear",
                color="120,90,70",
            )

        self._attach_collision_sensor()
        print(
            f"\n[LeadSwerveReveal] Spawn\n"
            f"  Ego road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  lead={self._lead_distance:.1f}m hazard={self._hazard_distance:.1f}m follower={self._follower_distance:.1f}m\n"
            f"  lead_swerve_start={self._lead_swerve_start_s:.1f}s lateral={self._lead_swerve_lateral_m:.1f}m\n"
            f"  side lanes={'L' if left_wp else '-'}{'R' if right_wp else '-'}\n",
            flush=True,
        )

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateLeadSwerveReveal",
        )

    def _advance_pose(self, pose, dt, forward_speed, lateral_speed=0.0):
        if pose is None:
            return
        pose["loc"] = self._vector_add(pose["loc"], pose["forward"], forward_speed * dt)
        pose["loc"] = self._vector_add(pose["loc"], pose["right"], lateral_speed * dt)

    def _set_pose(self, pose, yaw_offset=0.0, forward_speed=0.0, lateral_speed=0.0):
        if pose is None:
            return
        actor = pose["actor"]
        if actor is None or not actor.is_alive:
            return
        actor.set_transform(carla.Transform(
            pose["loc"],
            carla.Rotation(yaw=pose["yaw"] + yaw_offset),
        ))
        actor.set_target_velocity(carla.Vector3D(
            pose["forward"].x * forward_speed + pose["right"].x * lateral_speed,
            pose["forward"].y * forward_speed + pose["right"].y * lateral_speed,
            0.0,
        ))

    def _step_scene(self, elapsed, dt):
        # The lead and hazard ride at the EGO's own speed, holding the tight gap
        # they spawned with, so the whole convoy accelerates up to highway speed
        # together. Once they are all cruising fast, the hazard SLAMS its brakes
        # from full speed (a big, visible deceleration right in front of the
        # camera) and the lead darts aside a beat later to reveal it. Matching
        # the ego keeps the hard brake close to the camera instead of 70m ahead
        # where a stopped car just looks parked.
        ego_mps = _speed_mps(self._ego)
        ego_loc = self._ego.get_location()
        hazard_gap = _planar_distance(ego_loc, self._hazard["loc"]) if self._hazard else 999.0

        # --- Hazard: cruise with the ego, then SLAM the brakes from full speed. ---
        if not self._hazard_brake_triggered:
            if elapsed >= self._hazard_brake_start_s and hazard_gap <= self._hazard_brake_trigger_gap_m:
                self._hazard_brake_triggered = True
                self._hazard_brake_time = elapsed
                # Brake from whatever speed the convoy is actually doing (~highway).
                self._hazard_brake_from_speed = min(ego_mps, self._hazard_initial_speed)
        if self._hazard_brake_triggered:
            brake_ratio = min(1.0, (elapsed - self._hazard_brake_time) / max(0.1, self._hazard_brake_ramp_s))
            hazard_speed = self._hazard_brake_from_speed * (1.0 - brake_ratio) + self._hazard_stop_speed * brake_ratio
            if brake_ratio >= 1.0:
                self._hazard_stopped = True
        else:
            hazard_speed = min(ego_mps, self._hazard_initial_speed)
        self._advance_pose(self._hazard, dt, hazard_speed, 0.0)
        self._set_pose(self._hazard, 0.0, hazard_speed, 0.0)

        # --- Lead: rides just ahead of the ego (occluding the hazard) at the
        #     ego's speed, then swerves aside a beat after the hazard brakes,
        #     exposing the hard-braking vehicle to the ego behind. ---
        lead_speed = min(ego_mps, self._lead_speed)
        lead_lat_speed = 0.0
        lead_yaw = 0.0
        swerve_active = (
            self._hazard_brake_triggered
            and elapsed >= self._hazard_brake_time + self._lead_swerve_delay_after_brake_s
        )
        if swerve_active:
            swerve_t = elapsed - (self._hazard_brake_time + self._lead_swerve_delay_after_brake_s)
            ratio = min(1.0, swerve_t / max(0.1, self._lead_swerve_duration_s))
            target_lat = self._lead_swerve_lateral_m * ratio
            lead_lat_speed = (target_lat - self._lead_lateral_offset) / max(0.02, dt)
            self._lead_lateral_offset = target_lat
            lead_yaw = -16.0 * ratio
            lead_speed = self._lead_swerve_speed_kmh
        self._advance_pose(self._lead, dt, lead_speed, lead_lat_speed)
        self._set_pose(self._lead, lead_yaw, lead_speed, lead_lat_speed)

        for pose in [self._left_front, self._left_rear, self._right_front, self._right_rear]:
            # Side traffic rides alongside the ego at its own speed so it forms a
            # moving wall in the adjacent lanes. By tracking the ego (including
            # while it brakes) the wall stays beside the ego, boxing it in: when
            # the hazard is revealed too late to stop, the ego cannot swerve
            # around it and is forced into a straight rear-end collision.
            side_speed = min(ego_mps, self._side_speed)
            self._advance_pose(pose, dt, side_speed, 0.0)
            self._set_pose(pose, 0.0, side_speed, 0.0)

        if self._follower is not None:
            self._advance_pose(self._follower, dt, self._follower_speed, 0.0)
            self._set_pose(self._follower, 0.0, self._follower_speed, 0.0)

    def _hold_scene(self):
        stop = carla.Vector3D(0.0, 0.0, 0.0)
        for pose in [self._lead, self._hazard, self._left_front, self._left_rear, self._right_front, self._right_rear, self._follower]:
            if pose is not None and pose["actor"] is not None and pose["actor"].is_alive:
                pose["actor"].set_target_velocity(stop)

    def _create_behavior(self):
        sequence = py_trees.composites.Sequence("LeadSwerveReveal_Root")
        sequence.add_child(_RunLeadSwerveReveal(self))
        sequence.add_child(_PostImpactHold(self, self._post_impact_hold))
        cleanup = py_trees.composites.Sequence("Cleanup")
        for index, pose in enumerate([self._lead, self._hazard, self._left_front, self._left_rear, self._right_front, self._right_rear, self._follower]):
            if pose is not None:
                cleanup.add_child(ActorDestroy(pose["actor"], name=f"Destroy{index}"))
        sequence.add_child(cleanup)

        lane_waypoints = [self._trigger_wp]
        for lane in (self._trigger_wp.get_left_lane(), self._trigger_wp.get_right_lane()):
            if lane is not None and lane.lane_type == carla.LaneType.Driving:
                lane_waypoints.append(lane)

        outer = py_trees.composites.Parallel(
            "LeadSwerveReveal_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(sequence)
        outer.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            max_brake=self._ego_cap_brake,
            name="EgoSpeedGovernor",
        ))
        outer.add_child(_EgoMinSpeedForcer(
            self._ego,
            min_speed_kmh=self._ego_min_speed_kmh,
            throttle=self._ego_force_throttle,
            scenario=self,
        ))
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego] + [p["actor"] for p in [self._lead, self._hazard, self._left_front, self._left_rear, self._right_front, self._right_rear, self._follower] if p is not None],
            lane_waypoints=lane_waypoints,
            center_location=self._trigger_wp.transform.location,
            radius=420.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        outer.add_child(StopActorsOnEgoCollision(
            self._ego,
            actors_to_stop=[p["actor"] for p in [self._lead, self._hazard, self._left_front, self._left_rear, self._right_front, self._right_rear, self._follower] if p is not None],
            name="StopOnCollision",
        ))
        return outer

    def _create_test_criteria(self):
        return [CollisionTest(self._ego)]

    def __del__(self):
        try:
            if self._collision_sensor is not None and self._collision_sensor.is_alive:
                self._collision_sensor.stop()
                self._collision_sensor.destroy()
        except RuntimeError:
            pass
        self.remove_all_actors()
