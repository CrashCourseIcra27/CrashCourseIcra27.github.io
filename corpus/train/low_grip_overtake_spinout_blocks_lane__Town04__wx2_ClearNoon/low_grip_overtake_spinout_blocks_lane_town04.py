"""Low grip overtake spinout blocks lane (Town04).

An overtaking vehicle spins out on a reduced-grip surface and blocks the ego's lane.
"""

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
from srunner.scenariomanager.scenarioatomics.custom_atomics_low_grip_overtake_spinout_blocks_lane_town04 import (
    EgoSpeedGovernor,
    HoldUntilEgoMoves,
    LaneCleaner,
    PeriodicStatusLogger,
    SpinOutControl,
    WaitUntilAheadOfEgo,
)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario


# ---------------------------------------------------------------------------
# Scenario class
# ---------------------------------------------------------------------------


class LowGripOvertakeSpinoutBlocksLaneTown04(BasicScenario):
    """Ego vehicle drives into a sudden pile-up triggered by an adjacent-lane spin."""

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

        op = dict(config.other_parameters) if hasattr(config, "other_parameters") else {}
        if hasattr(config, "scenario") and hasattr(config.scenario, "other_parameters"):
            op.update(config.scenario.other_parameters)

        self._activation_distance = self._get_param(op, "activation_distance_m", 140.0)
        self._trig_spawn_behind = self._get_param(op, "trig_spawn_behind_ego", 55.0)
        self._approach_speed = self._get_param(op, "approach_speed_kmh", 104.0) / 3.6
        # Fire SpinOut while trigger is 5 m BEHIND the ego so the drift/block
        # phase forms RIGHT AT the ego rather than racing 20 m ahead first.
        self._ahead_trigger_dist = self._get_param(op, "ahead_trigger_distance_m", -5.0)
        self._road_friction_scale = self._get_param(op, "road_friction_scale", 0.78)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 88.0)
        self._ego_cap_brake = self._get_param(op, "ego_cap_brake", 0.12)
        self._aftermath_duration = self._get_param(op, "aftermath_duration_s", 6.0)
        # Set to 0 to disable pre-staged lead: the spinning trigger is the blockage.
        # A non-zero value spawns a stopped vehicle that distance ahead, but on this
        # short route it causes the autopilot to pre-brake, preventing the collision.
        self._lead_distance = self._get_param(op, "lead_vehicle_ahead_m", 0.0)
        self._ego_lane_shift = self._get_param(op, "ego_lane_shift", 0, int)
        self._fallback_adjacent_offset = self._get_param(op, "fallback_adjacent_offset_m", 3.6)

        # SpinOutControl parameters (proven values from adjacent_lane_spin)
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

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._trigger_vehicle = None
        self._lead_vehicle = None
        self._trig_lane_wps = []
        self._trigger_side = "left"

        super().__init__(
            name="LowGripOvertakeSpinoutBlocksLaneTown04",
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
            world.spawn_actor(
                bp, carla.Transform(carla.Location(-10000.0, -10000.0, 0.0))
            )
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
            self._ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        for _ in range(abs(self._ego_lane_shift)):
            next_wp = ego_wp.get_left_lane() if self._ego_lane_shift > 0 else ego_wp.get_right_lane()
            if next_wp is None or next_wp.lane_type != carla.LaneType.Driving:
                break
            ego_wp = next_wp

        # Require an adjacent driving lane for the trigger vehicle.
        left_wp = ego_wp.get_left_lane()
        right_wp = ego_wp.get_right_lane()
        fallback_offset = 0.0
        if left_wp is not None and left_wp.lane_type == carla.LaneType.Driving:
            adjacent_wp = left_wp
            self._trigger_side = "left"
        elif right_wp is not None and right_wp.lane_type == carla.LaneType.Driving:
            adjacent_wp = right_wp
            self._trigger_side = "right"
        else:
            adjacent_wp = ego_wp
            self._trigger_side = "left"
            fallback_offset = self._fallback_adjacent_offset

        # Trigger vehicle: spawns BEHIND ego in the adjacent lane.
        trig_spawn_wp = (
            self._walk_waypoint(adjacent_wp, self._trig_spawn_behind, forward=False)
            or adjacent_wp
        )
        self._trig_lane_wps = self._build_plan(trig_spawn_wp)

        trig_location = carla.Location(
            trig_spawn_wp.transform.location.x,
            trig_spawn_wp.transform.location.y,
            trig_spawn_wp.transform.location.z + 0.5,
        )
        if fallback_offset:
            right_vec = trig_spawn_wp.transform.get_right_vector()
            trig_location.x -= right_vec.x * fallback_offset
            trig_location.y -= right_vec.y * fallback_offset

        trig_transform = carla.Transform(
            trig_location,
            trig_spawn_wp.transform.rotation,
        )
        self._trigger_vehicle = CarlaDataProvider.request_new_actor(
            "vehicle.tesla.model3",
            trig_transform,
            rolename="pile_up_trigger",
            color="220,40,40",
        )
        if self._trigger_vehicle is None:
            raise RuntimeError("[FollowingPileup] Trigger vehicle spawn failed")
        self._trigger_vehicle.set_simulate_physics(True)
        self._trigger_vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.other_actors.append(self._trigger_vehicle)

        # Lead vehicle: stopped ahead in the ego's lane (pile-up scene context).
        if self._lead_distance > 0.0:
            lead_wp = (
                self._walk_waypoint(ego_wp, self._lead_distance, forward=True)
                or ego_wp
            )
            lead_transform = carla.Transform(
                carla.Location(
                    lead_wp.transform.location.x,
                    lead_wp.transform.location.y,
                    lead_wp.transform.location.z + 0.5,
                ),
                lead_wp.transform.rotation,
            )
            self._lead_vehicle = CarlaDataProvider.request_new_actor(
                "vehicle.audi.tt",
                lead_transform,
                rolename="pile_up_lead",
                color="40,80,200",
            )
            if self._lead_vehicle is not None:
                self._lead_vehicle.set_simulate_physics(True)
                self._lead_vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                self._lead_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
                )
                self.other_actors.append(self._lead_vehicle)

        print(
            f"\n[FollowingPileup] Spawn\n"
            f"  Ego    road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  TRIG   road={trig_spawn_wp.road_id} lane={trig_spawn_wp.lane_id} "
            f"side={self._trigger_side}  "
            f"fallback_offset={fallback_offset:.1f}m  "
            f"behind={self._trig_spawn_behind:.1f}m  "
            f"approach={self._approach_speed * 3.6:.1f} km/h\n"
            f"  LEAD   {self._lead_distance:.0f}m ahead (stopped)\n"
            f"  friction={self._road_friction_scale:.2f}  "
            f"ego_cap={self._ego_speed_cap_kmh:.0f} km/h\n",
            flush=True,
        )

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateFollowingPileupTrigger",
        )

    def _create_behavior(self):
        from agents.navigation.local_planner import RoadOption

        P = py_trees.common.ParallelPolicy

        plan = [(wp, RoadOption.LANEFOLLOW) for wp in self._trig_lane_wps]

        root = py_trees.composites.Sequence("PileupRoot")

        # Phase 0 – hold trigger at rest until ego starts moving.
        phase0 = py_trees.composites.Parallel(
            "Phase0_Hold", policy=P.SUCCESS_ON_ONE
        )
        phase0.add_child(TimeOut(8.0, name="Phase0_Timeout"))
        phase0.add_child(
            HoldUntilEgoMoves(self._trigger_vehicle, self._ego, name="HoldTrigger")
        )
        root.add_child(phase0)

        # Phase 1 – trigger approaches at high speed in the adjacent lane.
        phase1 = py_trees.composites.Parallel(
            "Phase1_Approach", policy=P.SUCCESS_ON_ONE
        )
        phase1.add_child(TimeOut(25.0, name="Phase1_Timeout"))
        phase1.add_child(
            WaitUntilAheadOfEgo(
                self._trigger_vehicle,
                self._ego,
                ahead_distance=self._ahead_trigger_dist,
                name="WaitTrigAhead",
            )
        )
        phase1.add_child(
            WaypointFollower(
                self._trigger_vehicle,
                target_speed=self._approach_speed,
                plan=plan,
                name="Trig_Approach",
            )
        )
        root.add_child(phase1)

        # Phase 2 – trigger cuts into the ego's lane and spins out.
        phase2_timeout = (
            self._cut_in_duration
            + self._drift_duration
            + self._spin_duration
            + self._block_duration
            + 2.0
        )
        phase2 = py_trees.composites.Parallel(
            "Phase2_SpinIntoEgoLane", policy=P.SUCCESS_ON_ONE
        )
        phase2.add_child(
            SpinOutControl(
                self._trigger_vehicle,
                steer_direction=1.0 if self._trigger_side == "left" else -1.0,
                cut_in_steer=self._cut_in_steer,
                cut_in_throttle=self._cut_in_throttle,
                cut_in_duration=self._cut_in_duration,
                drift_steer=self._drift_steer,
                drift_steer_sign=self._drift_steer_sign,
                drift_throttle=self._drift_throttle,
                drift_brake=self._drift_brake,
                drift_duration=self._drift_duration,
                spin_throttle=self._spin_throttle,
                spin_duration=self._spin_duration,
                block_duration=self._block_duration,
                name="Trig_SpinOut",
            )
        )
        phase2.add_child(TimeOut(phase2_timeout, name="Phase2_Timeout"))
        root.add_child(phase2)

        # Phase 3 – aftermath: hold the crash scene for the video.
        root.add_child(TimeOut(self._aftermath_duration, name="Phase3_Aftermath"))

        # Cleanup.
        cleanup = py_trees.composites.Sequence("Cleanup")
        cleanup.add_child(ActorDestroy(self._trigger_vehicle, name="DestroyTrig"))
        if self._lead_vehicle is not None:
            cleanup.add_child(ActorDestroy(self._lead_vehicle, name="DestroyLead"))
        root.add_child(cleanup)

        # Outer parallel: EgoSpeedGovernor + LaneCleaner + StatusLogger.
        lanes = [self._trigger_wp]
        left_lane = self._trigger_wp.get_left_lane()
        if left_lane is not None and left_lane.lane_type == carla.LaneType.Driving:
            lanes.append(left_lane)

        protected = [
            a for a in [self._ego, self._trigger_vehicle, self._lead_vehicle]
            if a is not None
        ]

        outer = py_trees.composites.Parallel(
            "PileupOuter", policy=P.SUCCESS_ON_ONE
        )
        outer.add_child(root)
        outer.add_child(
            LaneCleaner(
                protected_actors=protected,
                lane_waypoints=lanes,
                center_location=self._trigger_wp.transform.location,
                radius=500.0,
                interval=1.0,
                name="ScenarioLaneCleaner",
            )
        )
        outer.add_child(
            EgoSpeedGovernor(
                self._ego,
                speed_cap_kmh=self._ego_speed_cap_kmh,
                brake_force=self._ego_cap_brake,
                name="EgoSpeedGovernor",
            )
        )
        outer.add_child(
            PeriodicStatusLogger(
                self._ego,
                self._trigger_vehicle,
                self._lead_vehicle,
                interval=1.0,
                name="StatusLogger",
            )
        )
        return outer

    def _create_test_criteria(self):
        return [CollisionTest(self._ego)]

    def __del__(self):
        self.remove_all_actors()
