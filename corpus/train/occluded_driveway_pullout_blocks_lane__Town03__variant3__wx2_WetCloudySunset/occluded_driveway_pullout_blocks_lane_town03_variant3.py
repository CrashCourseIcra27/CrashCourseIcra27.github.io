"""Occluded driveway pullout blocks lane (Town03).

A vehicle pulls out from a wall-occluded driveway and blocks the ego's lane.
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
from srunner.scenariomanager.scenarioatomics.custom_atomics_occluded_driveway_pullout_blocks_lane_town03_variant3 import (
    EgoSpeedGovernor,
    HoldUntilEgoMoves,
    LaneCleaner,
)
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenarios.basic_scenario import BasicScenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _speed_mps(actor):
    if actor is None or not actor.is_alive:
        return 0.0
    v = actor.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def _planar_distance(loc_a, loc_b):
    return math.hypot(loc_a.x - loc_b.x, loc_a.y - loc_b.y)


# ---------------------------------------------------------------------------
# Behaviour: force ego to maintain a minimum speed
# ---------------------------------------------------------------------------

class _EgoMinSpeedForcer(py_trees.behaviour.Behaviour):
    """Override autopilot braking — force ego to maintain a minimum speed."""

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
        if speed > 5.0 and speed < self._min_mps:
            ctrl = self._ego.get_control()
            ctrl.brake = 0.0
            ctrl.throttle = self._throttle
            self._ego.apply_control(ctrl)
        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# Behaviour: driveway pull-out maneuver
# ---------------------------------------------------------------------------

class _RunDrivewayPullout(py_trees.behaviour.Behaviour):
    """
    Wait until the ego is close, then drive the adversary out of the hidden
    driveway and into the ego lane.

    States
    ------
    WAITING     — adversary sits motionless behind the wall (hand-brake on)
    PULLING_OUT — adversary drives perpendicularly toward the road
    IN_ROAD     — adversary has entered the lane and holds position
    """

    WAITING     = "waiting"
    PULLING_OUT = "pulling_out"
    IN_ROAD     = "in_road"

    def __init__(self, scenario, name="RunDrivewayPullout"):
        super().__init__(name)
        self._scenario = scenario
        self._start_time = None
        self._state = self.WAITING
        self._last_log = -999.0

    def initialise(self):
        self._start_time = GameTime.get_time()
        self._state = self.WAITING

    # ------------------------------------------------------------------
    def _d_right(self, loc):
        """
        Signed distance from `loc` to the road centre-line measured along
        the road's right vector.  Positive = still on the driveway side.
        """
        ref = self._scenario._driveway_wp.transform.location
        rv  = self._scenario._right_vec
        return (loc.x - ref.x) * rv.x + (loc.y - ref.y) * rv.y

    # ------------------------------------------------------------------
    def update(self):
        scenario = self._scenario

        if scenario._impact_detected:
            print("[WallOcclusion] *** COLLISION ***", flush=True)
            return py_trees.common.Status.SUCCESS

        elapsed = (
            GameTime.get_time() - self._start_time if self._start_time else 0.0
        )

        adv = scenario._adversary
        if adv is None or not adv.is_alive:
            return py_trees.common.Status.FAILURE

        ego_loc  = scenario._ego.get_location()
        adv_loc  = adv.get_location()
        d_spawn  = _planar_distance(ego_loc, scenario._adversary_spawn_loc)
        d_right  = self._d_right(adv_loc)

        # ------------------------------------------------------------------
        if self._state == self.WAITING:
            # Keep adversary motionless (hidden behind wall)
            adv.set_target_velocity(carla.Vector3D(0, 0, 0))
            adv.apply_control(carla.VehicleControl(
                throttle=0.0, brake=1.0, hand_brake=True,
            ))

            if d_spawn <= scenario._trigger_distance_to_driveway:
                self._state = self.PULLING_OUT
                adv.set_simulate_physics(True)
                print(
                    f"[WallOcclusion] Ego {d_spawn:.1f}m from driveway — "
                    "PULL-OUT TRIGGERED!",
                    flush=True,
                )

        # ------------------------------------------------------------------
        elif self._state == self.PULLING_OUT:
            rv = scenario._right_vec
            spd = scenario._pull_out_speed_mps

            if d_right > scenario._in_road_brake_threshold:
                # Still on the driveway side — drive decisively toward road
                adv.set_target_velocity(carla.Vector3D(
                    -rv.x * spd, -rv.y * spd, 0.0,
                ))
                adv.apply_control(carla.VehicleControl(
                    throttle=0.85, brake=0.0, steer=0.0, hand_brake=False,
                ))
            else:
                # Entered the ego lane — hard brake and stop
                self._state = self.IN_ROAD
                adv.set_target_velocity(carla.Vector3D(0, 0, 0))
                adv.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=1.0, hand_brake=True,
                ))
                print(
                    f"[WallOcclusion] Adversary in ego lane "
                    f"(d_right={d_right:.2f}m) — holding position",
                    flush=True,
                )

        # ------------------------------------------------------------------
        elif self._state == self.IN_ROAD:
            # Keep adversary stationary in the conflict zone
            adv.set_target_velocity(carla.Vector3D(0, 0, 0))
            adv.apply_control(carla.VehicleControl(
                throttle=0.0, brake=1.0, hand_brake=True,
            ))

        # ------------------------------------------------------------------
        # Periodic status log
        if elapsed - self._last_log >= 1.0:
            self._last_log = elapsed
            ego_spd = _speed_mps(scenario._ego) * 3.6
            adv_spd = _speed_mps(adv) * 3.6
            print(
                f"[WallOcclusion] t={elapsed:.1f}s  state={self._state}  "
                f"ego={ego_spd:.0f}km/h  d_spawn={d_spawn:.1f}m  "
                f"adv={adv_spd:.1f}km/h  d_right={d_right:.2f}m  "
                f"impact={scenario._impact_detected}",
                flush=True,
            )

        if elapsed >= scenario._max_run_time:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# Behaviour: hold the crash scene for video recording
# ---------------------------------------------------------------------------

class _PostImpactHold(py_trees.behaviour.Behaviour):
    def __init__(self, duration, name="PostImpactHold"):
        super().__init__(name)
        self._duration = float(duration)
        self._start_time = None

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        elapsed = (
            GameTime.get_time() - self._start_time if self._start_time else 0.0
        )
        if elapsed >= self._duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# Main scenario class
# ---------------------------------------------------------------------------

class OccludedDrivewayPulloutBlocksLaneTown03Variant3(BasicScenario):
    """
    A vehicle exits a right-side residential driveway through wall occlusion
    and crosses into the ego's lane — Town03 urban.

    The adversary is hidden behind a roadside building / wall on the ego's
    right side.  When the ego is within `trigger_distance_to_driveway_m`
    planar metres of the adversary's spawn location, the adversary accelerates
    perpendicularly out of the driveway and stops in the ego lane.  The ego
    cannot see the adversary until it clears the occluding wall, leaving very
    little time to react.

    XML other_parameters
    --------------------
    activation_distance_m           scenario activates when ego is this close
                                    to the trigger_point
    driveway_ahead_from_trigger_m   road distance from trigger to adversary x
    driveway_right_offset_m         lateral offset (right of road centre)
    pull_out_speed_kmh              adversary pull-out speed
    in_road_brake_threshold_m       brake when d_right drops below this value
    trigger_distance_to_driveway_m  planar distance that fires the pull-out
    ego_speed_cap_kmh               ego speed governor cap
    ego_min_speed_kmh               minimum ego speed enforced
    ego_cap_brake                   brake force applied when above cap
    max_run_time                    scenario timeout (s)
    post_impact_hold                hold duration after collision (s)
    """

    timeout = 120

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=120):
        self._world = world
        self._map   = CarlaDataProvider.get_map()
        self._ego   = ego_vehicles[0]

        op = dict(config.other_parameters) if hasattr(config, "other_parameters") else {}
        if hasattr(config, "scenario") and hasattr(config.scenario, "other_parameters"):
            op.update(config.scenario.other_parameters)
        if isinstance(op, dict) and isinstance(op.get("other_parameters"), dict):
            op = op["other_parameters"]
        self._activation_distance         = self._get_param(op, "activation_distance_m",             80.0)
        self._driveway_ahead_from_trigger  = self._get_param(op, "driveway_ahead_from_trigger_m",     30.0)
        self._driveway_right_offset        = self._get_param(op, "driveway_right_offset_m",            7.5)
        self._pull_out_speed_mps           = self._get_param(op, "pull_out_speed_kmh",                20.0) / 3.6
        self._in_road_brake_threshold      = self._get_param(op, "in_road_brake_threshold_m",          1.5)
        self._trigger_distance_to_driveway = self._get_param(op, "trigger_distance_to_driveway_m",    18.0)
        self._ego_speed_cap_kmh            = self._get_param(op, "ego_speed_cap_kmh",                 45.0)
        self._ego_min_speed_kmh            = self._get_param(op, "ego_min_speed_kmh",                 40.0)
        self._ego_cap_brake                = self._get_param(op, "ego_cap_brake",                      0.12)
        self._max_run_time                 = self._get_param(op, "max_run_time",                       25.0)
        self._post_impact_hold             = self._get_param(op, "post_impact_hold",                    4.5)

        self._trigger_wp          = self._map.get_waypoint(config.trigger_points[0].location)
        self._driveway_wp         = None
        self._right_vec           = None
        self._adversary           = None
        self._adversary_spawn_loc = None
        self._collision_sensor    = None
        self._impact_detected     = False

        super().__init__(
            name="OccludedDrivewayPulloutBlocksLaneTown03Variant3",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            terminate_on_failure=True,
            criteria_enable=criteria_enable,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _walk_waypoint(waypoint, distance, forward=True):
        remaining = abs(float(distance))
        cursor = waypoint
        while remaining > 0.2 and cursor is not None:
            step       = min(8.0, remaining)
            candidates = cursor.next(step) if forward else cursor.previous(step)
            if not candidates:
                break
            cursor     = candidates[0]
            remaining -= step
        return cursor

    # ------------------------------------------------------------------
    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        # Walk ahead along the road to find the driveway position
        self._driveway_wp = (
            self._walk_waypoint(
                self._trigger_wp,
                self._driveway_ahead_from_trigger,
                forward=True,
            ) or self._trigger_wp
        )
        road_t = self._driveway_wp.transform
        self._right_vec = road_t.get_right_vector()

        # Adversary spawn: right of road (behind building), facing toward road
        # Yaw perpendicular to road, pointing toward road (opposite of right vector)
        adv_yaw = math.degrees(
            math.atan2(-self._right_vec.y, -self._right_vec.x)
        )
        adv_rotation = carla.Rotation(
            pitch=road_t.rotation.pitch,
            yaw=adv_yaw,
            roll=0.0,
        )

        # Try several compact passenger-car models
        adv_models = [
            "vehicle.audi.a2",
            "vehicle.mini.cooper_s",
            "vehicle.nissan.micra",
            "vehicle.citroen.c3",
            "vehicle.tesla.model3",
        ]
        self._adversary = None
        adv_loc = None
        offsets = [
            self._driveway_right_offset,
            max(2.5, self._driveway_right_offset - 1.5),
            max(2.0, self._driveway_right_offset - 3.0),
            self._driveway_right_offset + 1.5,
        ]
        for offset in offsets:
            candidate_loc = carla.Location(
                x=road_t.location.x + self._right_vec.x * offset,
                y=road_t.location.y + self._right_vec.y * offset,
                z=road_t.location.z + 0.3,
            )
            adv_transform = carla.Transform(candidate_loc, adv_rotation)
            for model in adv_models:
                self._adversary = CarlaDataProvider.request_new_actor(
                    model, adv_transform, rolename="driveway_vehicle",
                )
                if self._adversary is not None:
                    adv_loc = candidate_loc
                    self._driveway_right_offset = offset
                    break
            if self._adversary is not None:
                break
        if self._adversary is None:
            raise RuntimeError("[WallOcclusion] Adversary spawn failed")
        self._adversary_spawn_loc = adv_loc

        self._adversary.set_simulate_physics(True)
        self._adversary.apply_control(carla.VehicleControl(
            throttle=0.0, brake=1.0, hand_brake=True,
        ))
        self._adversary.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.other_actors.append(self._adversary)

        # Collision sensor on the ego
        collision_bp = self._world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        self._collision_sensor = self._world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self._ego,
        )
        self._collision_sensor.listen(lambda event: self._on_collision(event))

        print(
            f"\n[WallOcclusion] Spawn summary\n"
            f"  Ego road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  Trigger wp  ({road_t.location.x - self._driveway_ahead_from_trigger:.1f}, "
            f"{self._trigger_wp.transform.location.y:.1f})\n"
            f"  Driveway wp ({road_t.location.x:.1f}, {road_t.location.y:.1f})\n"
            f"  Adversary   ({adv_loc.x:.1f}, {adv_loc.y:.1f})  yaw={adv_yaw:.1f}°\n"
            f"  right_vec   ({self._right_vec.x:.3f}, {self._right_vec.y:.3f})\n"
            f"  trigger_dist={self._trigger_distance_to_driveway:.1f}m  "
            f"pull_out={self._pull_out_speed_mps * 3.6:.1f}km/h  "
            f"brake_thresh={self._in_road_brake_threshold:.1f}m\n",
            flush=True,
        )

    # ------------------------------------------------------------------
    def _on_collision(self, event):
        other = event.other_actor
        if self._adversary is not None and other.id != self._adversary.id:
            print(
                f"[WallOcclusion] Ignoring non-adversary collision: "
                f"{other.type_id}",
                flush=True,
            )
            return
        if not self._impact_detected:
            self._impact_detected = True
            print(
                f"[WallOcclusion] *** COLLISION: {event.other_actor.type_id} ***",
                flush=True,
            )

    # ------------------------------------------------------------------
    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateSideStreetWallOcclusion",
        )

    # ------------------------------------------------------------------
    def _create_behavior(self):
        root = py_trees.composites.Sequence("WallOcclusionSequence")

        # Phase 0: wait for ego to start moving before engaging adversary
        sync = py_trees.composites.Parallel(
            "Phase0_Sync",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
        )
        sync.add_child(HoldUntilEgoMoves(self._adversary, self._ego))
        root.add_child(sync)

        # Phase 1: main loop — ego approaches, adversary pulls out, collision
        main_loop = py_trees.composites.Parallel(
            "Phase1_MainLoop",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        main_loop.add_child(_RunDrivewayPullout(self))
        main_loop.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=self._ego_cap_brake,
            max_brake=self._ego_cap_brake,
        ))
        main_loop.add_child(_EgoMinSpeedForcer(
            self._ego, min_speed_kmh=self._ego_min_speed_kmh,
        ))
        main_loop.add_child(TimeOut(self._max_run_time))
        root.add_child(main_loop)

        # Phase 2: hold crash scene for video
        root.add_child(_PostImpactHold(self._post_impact_hold))

        # Cleanup
        root.add_child(ActorDestroy(self._adversary, name="DestroyAdversary"))

        # Lane cleaner — remove background NPCs near the conflict zone
        ego_wp = self._map.get_waypoint(
            self._trigger_wp.transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        lane_wps = [ego_wp]
        left = ego_wp.get_left_lane()
        if left and left.lane_type == carla.LaneType.Driving:
            lane_wps.append(left)
        right = ego_wp.get_right_lane()
        if right and right.lane_type == carla.LaneType.Driving:
            lane_wps.append(right)

        outer = py_trees.composites.Parallel(
            "WallOcclusion_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=[self._ego, self._adversary],
            lane_waypoints=lane_wps,
            center_location=self._trigger_wp.transform.location,
            radius=180.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        return outer

    # ------------------------------------------------------------------
    def _create_test_criteria(self):
        criteria = py_trees.composites.Parallel(
            "WallOcclusionCriteria",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        criteria.add_child(CollisionTest(self._ego))
        return criteria

    # ------------------------------------------------------------------
    def remove_all_actors(self):
        if self._collision_sensor is not None and self._collision_sensor.is_alive:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()
            self._collision_sensor = None
        super().remove_all_actors()
