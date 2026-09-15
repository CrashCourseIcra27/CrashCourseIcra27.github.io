"""Lead swerve reveals braking vehicle (Town12).

A lead vehicle changes lane to reveal a hard-braking vehicle stopped in the ego's lane.
"""

import math
import os

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation,
)
from srunner.scenariomanager.scenarioatomics.custom_atomics_lead_swerve_reveals_braking_vehicle_town12 import (
    EgoSpeedGovernor,
    LaneCleaner,
    PeriodicStatusLogger,
    actor_speed,
    lane_follow_control,
    planar_distance,
)
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.basic_scenario import BasicScenario


class _RunNaturalLeadSwerve(py_trees.behaviour.Behaviour):
    """Deterministic physics state machine for lead + hazard."""

    def __init__(self, scenario, name="RunNaturalLeadSwerve"):
        super().__init__(name)
        self._s = scenario
        self._state = "cruise"
        self._t0 = None
        self._ego_settled_since = None
        self._last_log = -99.0

    def initialise(self):
        self._t0 = GameTime.get_time()

    def _enter(self, state):
        print(f"[LeadSwerve] state -> {state}  t={GameTime.get_time():.1f}", flush=True)
        self._state = state
        self._t0 = GameTime.get_time()

    def _pace_lead(self, world_map):
        """Lead holds a natural following gap AHEAD of the ego via engine/brakes."""
        s = self._s
        lead, ego = s._lead, s._ego
        if lead is None or not lead.is_alive:
            return
        gap = planar_distance(lead.get_location(), ego.get_location())
        ego_speed = actor_speed(ego)
        # P-control on the gap: lead too far ahead -> ease off, too close -> pull away.
        target = ego_speed + 0.45 * (s._lead_gap - gap)
        target = max(0.0, min(s._lead_max_speed, target))
        if ego_speed < 0.3 and gap > s._lead_gap * 0.6:
            target = 0.0  # ego parked and lead already has its gap: wait like traffic
        lead.apply_control(lane_follow_control(world_map, lead, target))

    def update(self):
        s = self._s
        lead, hazard, ego = s._lead, s._hazard, s._ego
        if hazard is None:
            return py_trees.common.Status.FAILURE
        now = GameTime.get_time()
        elapsed = now - self._t0 if self._t0 else 0.0
        world_map = CarlaDataProvider.get_map()
        alive_l = lead is not None and lead.is_alive
        alive_h = hazard.is_alive

        if now - self._last_log > 1.5:
            self._last_log = now
            hz_gap = planar_distance(ego.get_location(), hazard.get_location()) if alive_h else -1
            print(f"[LeadSwerve] st={self._state} ego={actor_speed(ego)*3.6:.0f}km/h "
                  f"hazard_gap={hz_gap:.0f}m", flush=True)

        if self._state == "cruise":
            self._pace_lead(world_map)
            ego_gap = planar_distance(ego.get_location(), hazard.get_location()) if alive_h else 999.0
            if alive_h:
                # Hazard behaves like traffic ahead: negative feedback anchored to the
                # ego's speed holds ~hold_gap in front of it (too far ahead -> slow
                # down/wait, too close -> pull away), so even a slow ego always closes
                # in and the scene can never dead-end.
                hz_target = max(0.0, min(
                    s._hazard_speed,
                    actor_speed(ego) + 0.5 * (s._hazard_hold_gap - ego_gap)))
                hazard.apply_control(lane_follow_control(world_map, hazard, hz_target))
            if actor_speed(ego) > 2.5 and ego_gap < s._brake_trigger_gap:
                self._enter("hazard_brakes")

        elif self._state == "hazard_brakes":
            # Hazard slams the brakes from speed — a real, visible hard stop.
            if alive_h:
                hazard.apply_control(carla.VehicleControl(brake=1.0))
                try:
                    hazard.set_light_state(carla.VehicleLightState.Brake)
                except Exception:
                    pass
            self._pace_lead(world_map)
            # Lead keeps pacing until it actually closes on the stopped hazard,
            # then dodges at the last moment — a natural late reveal.
            lead_gap = (planar_distance(lead.get_location(), hazard.get_location())
                        if (alive_l and alive_h) else 0.0)
            # 12s fallback: if the ego (and thus the pacing lead) parked early and
            # the lead never closes on the hazard, the lead driver still decides
            # to pull around — the scene can never deadlock here.
            if elapsed > s._lead_react_delay and (lead_gap < s._swerve_gap or elapsed > 12.0):
                self._enter("lead_swerve")

        elif self._state == "lead_swerve":
            if alive_h:
                hazard.apply_control(carla.VehicleControl(brake=1.0, hand_brake=actor_speed(hazard) < 0.3))
            if alive_l:
                # Committed steer pulse -> counter-steer -> settle in adjacent lane.
                ctrl = carla.VehicleControl()
                t = elapsed
                pulse = s._swerve_pulse_s
                if t < pulse:
                    ctrl.steer = s._swerve_steer
                    ctrl.throttle = 0.55
                elif t < 2.0 * pulse:
                    ctrl.steer = -s._swerve_steer * 0.85
                    ctrl.throttle = 0.55
                else:
                    lead.apply_control(lane_follow_control(world_map, lead, s._lead_escape_speed))
                    ctrl = None
                if ctrl is not None:
                    lead.apply_control(ctrl)
            if elapsed > 2.0 * s._swerve_pulse_s + 0.8:
                self._enter("block")

        elif self._state == "block":
            # Stopped hazard holds the ego lane until the ego has dealt with it.
            if alive_h:
                hazard.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            if alive_l:
                lead.apply_control(lane_follow_control(world_map, lead, s._lead_escape_speed))
                if planar_distance(lead.get_location(), ego.get_location()) > s._vanish_distance:
                    try:
                        lead.destroy()
                    except RuntimeError:
                        pass
            ego_gap = planar_distance(ego.get_location(), hazard.get_location()) if alive_h else 999.0
            ego_arrived = ego_gap < s._ego_arrive_gap and actor_speed(ego) < 1.0
            ego_passed = False
            if alive_h:
                fwd = hazard.get_transform().get_forward_vector()
                delta = ego.get_location() - hazard.get_location()
                ego_passed = (delta.x * fwd.x + delta.y * fwd.y) > 4.0
            if ego_arrived:
                if self._ego_settled_since is None:
                    self._ego_settled_since = now
            else:
                self._ego_settled_since = None
            settled = (self._ego_settled_since is not None
                       and now - self._ego_settled_since > s._block_hold_after_stop)
            if settled or ego_passed or elapsed > s._block_max_s:
                self._enter("clear")

        elif self._state == "clear":
            done_h = True
            if alive_h:
                hazard.apply_control(lane_follow_control(world_map, hazard, s._clear_speed))
                far = planar_distance(hazard.get_location(), ego.get_location()) > s._vanish_distance
                done_h = far or elapsed > s._clear_max_s
                if done_h:
                    try:
                        hazard.destroy()
                    except RuntimeError:
                        pass
            done_l = True
            if alive_l:
                lead.apply_control(lane_follow_control(world_map, lead, s._lead_escape_speed))
                far_l = planar_distance(lead.get_location(), ego.get_location()) > s._vanish_distance
                done_l = far_l or elapsed > s._clear_max_s
                if done_l:
                    try:
                        lead.destroy()
                    except RuntimeError:
                        pass
            if done_h and done_l:
                return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class LeadSwerveRevealsBrakingVehicleTown12(BasicScenario):
    """Natural-physics occlusion-reveal: lead swerves, hazard is stopped in-lane."""

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

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=180):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._ego = ego_vehicles[0]
        self._lead = None
        self._hazard = None

        op = {}
        for source in (config, getattr(config, "scenario", None)):
            if source is not None and hasattr(source, "other_parameters"):
                try:
                    op.update(dict(source.other_parameters))
                except Exception:
                    pass

        self._activation_distance = self._get_param(op, "activation_distance_m", 130.0)
        self._lead_spawn_ahead = self._get_param(op, "lead_vehicle_ahead_distance_m", 22.0)
        self._hazard_spawn_ahead = self._get_param(op, "hazard_vehicle_ahead_distance_m", 58.0)
        self._lead_gap = self._get_param(op, "lead_follow_gap_m", 20.0)
        self._lead_max_speed = self._get_param(op, "lead_max_speed_kmh", 70.0) / 3.6
        self._hazard_speed = self._get_param(op, "hazard_speed_kmh", 42.0) / 3.6
        self._hazard_hold_gap = self._get_param(op, "hazard_hold_gap_m", 42.0)
        self._brake_trigger_gap = self._get_param(op, "hazard_brake_trigger_gap_m", 55.0)
        self._lead_react_delay = self._get_param(op, "lead_react_delay_s", 0.9)
        self._swerve_gap = self._get_param(op, "swerve_gap_m", 15.0)
        self._swerve_pulse_s = self._get_param(op, "swerve_pulse_s", 0.55)
        self._swerve_steer_mag = self._get_param(op, "swerve_steer", 0.32)
        self._lead_escape_speed = self._get_param(op, "lead_escape_speed_kmh", 62.0) / 3.6
        self._block_hold_after_stop = self._get_param(op, "block_hold_after_stop_s", 4.0)
        self._block_max_s = self._get_param(op, "block_max_s", 30.0)
        self._ego_arrive_gap = self._get_param(op, "ego_arrive_gap_m", 24.0)
        self._clear_speed = self._get_param(op, "clear_speed_kmh", 45.0) / 3.6
        self._clear_max_s = self._get_param(op, "clear_max_s", 18.0)
        self._vanish_distance = self._get_param(op, "vanish_distance_m", 70.0)
        self._ego_speed_cap_kmh = self._get_param(op, "ego_speed_cap_kmh", 80.0)
        self._swerve_steer = -abs(self._swerve_steer_mag)  # sign set at spawn

        self._trigger_wp = self._map.get_waypoint(config.trigger_points[0].location)

        super().__init__(
            name="LeadSwerveRevealsBrakingVehicleTown12",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            terminate_on_failure=True,
            criteria_enable=criteria_enable,
        )

    def _spawn(self, models, waypoint, rolename, color=None):
        transform = carla.Transform(
            carla.Location(
                waypoint.transform.location.x,
                waypoint.transform.location.y,
                waypoint.transform.location.z + 0.3,
            ),
            waypoint.transform.rotation,
        )
        for model in models:
            actor = CarlaDataProvider.request_new_actor(
                model, transform, rolename=rolename, color=color)
            if actor is not None:
                actor.set_simulate_physics(True)
                actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                self.other_actors.append(actor)
                return actor
        return None

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(
            self._ego.get_location(), project_to_road=True,
            lane_type=carla.LaneType.Driving) or self._trigger_wp

        lead_wp = self._walk_waypoint(ego_wp, self._lead_spawn_ahead, True) or ego_wp
        hazard_wp = self._walk_waypoint(ego_wp, self._hazard_spawn_ahead, True) or lead_wp

        # Swerve toward whichever adjacent driving lane exists (prefer left).
        left_wp = ego_wp.get_left_lane()
        right_wp = ego_wp.get_right_lane()
        if left_wp is not None and left_wp.lane_type == carla.LaneType.Driving:
            self._swerve_steer = -abs(self._swerve_steer_mag)
        elif right_wp is not None and right_wp.lane_type == carla.LaneType.Driving:
            self._swerve_steer = abs(self._swerve_steer_mag)

        self._lead = self._spawn(
            ["vehicle.audi.tt", "vehicle.lincoln.mkz_2020", "vehicle.tesla.model3"],
            lead_wp, "lead_occluder_swerve", color="35,95,210")
        if self._lead is None:
            raise RuntimeError("[LeadSwerve] lead spawn failed")
        self._hazard = self._spawn(
            ["vehicle.nissan.patrol", "vehicle.tesla.model3", "vehicle.lincoln.mkz_2017"],
            hazard_wp, "hard_braking_hidden_vehicle", color="245,245,245")
        if self._hazard is None:
            raise RuntimeError("[LeadSwerve] hazard spawn failed")

        print(
            f"\n[LeadSwerve/NATURAL] Spawn\n"
            f"  Ego road={ego_wp.road_id} lane={ego_wp.lane_id}\n"
            f"  Lead ahead={self._lead_spawn_ahead:.0f}m gap_hold={self._lead_gap:.0f}m\n"
            f"  Hazard ahead={self._hazard_spawn_ahead:.0f}m @ {self._hazard_speed*3.6:.0f}km/h "
            f"brake_gap={self._brake_trigger_gap:.0f}m swerve_steer={self._swerve_steer:+.2f}\n",
            flush=True,
        )

    def _setup_scenario_trigger(self, config):
        return InTriggerDistanceToLocation(
            self._ego,
            self._trigger_wp.transform.location,
            self._activation_distance,
            name="ActivateLeadSwerveReveal",
        )

    def _create_behavior(self):
        root = py_trees.composites.Sequence("LeadSwerveNatural_Root")
        root.add_child(_RunNaturalLeadSwerve(self))

        lane_waypoints = [self._trigger_wp]
        for lane in (self._trigger_wp.get_left_lane(), self._trigger_wp.get_right_lane()):
            if lane is not None and lane.lane_type == carla.LaneType.Driving:
                lane_waypoints.append(lane)

        outer = py_trees.composites.Parallel(
            "LeadSwerveNatural_Outer",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
        )
        outer.add_child(root)
        outer.add_child(LaneCleaner(
            protected_actors=[a for a in (self._ego, self._lead, self._hazard) if a is not None],
            lane_waypoints=lane_waypoints,
            center_location=self._trigger_wp.transform.location,
            radius=420.0,
            interval=1.0,
            name="ScenarioLaneCleaner",
        ))
        outer.add_child(PeriodicStatusLogger(
            self._ego, self._lead, self._hazard,
            interval=2.0, name="StatusLogger",
        ))
        # Top-speed cap only — NO brake clamp, NO forced throttle (fair to both).
        outer.add_child(EgoSpeedGovernor(
            self._ego,
            speed_cap_kmh=self._ego_speed_cap_kmh,
            brake_force=0.3,
            name="EgoSpeedGovernor",
        ))
        return outer

    def _create_test_criteria(self):
        return [CollisionTest(self._ego)]

    def __del__(self):
        self.remove_all_actors()
