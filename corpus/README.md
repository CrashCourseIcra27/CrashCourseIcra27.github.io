# CARLA Evaluation Corpus

164 self-contained CARLA scenario bundles: 82 distinct routes, each
instantiated under 2 different weather presets.

## Layout

```
train/<scenario_name>__<Town>[__variantN]__wx<1|2>_<WeatherPreset>/
    route.xml                        route definition
    <scenario_name>_<town>[_variantN].py    the scenario implementation
    custom_atomics_<same_stem>.py    extra behaviours (only where required)
test/ ...
manifest.csv
```

Every folder is standalone: it contains everything specific to that route.
Files are duplicated across folders on purpose — there are no cross-folder
dependencies.

* `<scenario_name>` — describes what the scenario actually does.
* `<Town>` — the stock CARLA map the route runs on.
* `__variantN` — present only when the same scenario family is sited more than
  once in the same town.
* `__wx1` / `__wx2` — the two weather instantiations of the same route.

`manifest.csv` lists every instance with its split, town, weather preset,
route count, scenario class, and scenario module filename.

## Splits

82 train folders and 82 test folders. The split is per route, so both weather
variants of a route always land in the same split.

## Running a route

The leaderboard/ScenarioRunner harness discovers scenario classes by globbing
`${SCENARIO_RUNNER_ROOT}/srunner/scenarios/*.py` and importing each module to
find `BasicScenario` subclasses. A route's `<scenario type="...">` attribute is
matched against those discovered class names. So, for an instance folder:

1. Copy the instance's scenario module into `srunner/scenarios/`:
   `cp <instance>/<stem>.py $SCENARIO_RUNNER_ROOT/srunner/scenarios/`
2. If the folder contains a `custom_atomics_*.py`, copy it into
   `srunner/scenariomanager/scenarioatomics/` (the module imports it from
   there by that exact name).
3. Point the leaderboard at the instance's `route.xml`.

Every scenario class name and every module filename is unique across all 164
folders, and each module imports its own uniquely-named `custom_atomics_*`
module. You can therefore install the entire corpus at once without name
collisions — two routes never resolve to the same class.

This matters: several scenario families are sited in more than one town, and
the per-town implementations differ (road/lane guards, and in some cases
hazard geometry). Giving every instance its own class name prevents a route
from silently loading a sibling town's implementation.

## Dependencies

Beyond a standard CARLA 0.9.15 + ScenarioRunner/leaderboard install, the
modules import only: `carla`, `py_trees`, `numpy`, `cv2`, the stock
`srunner.scenariomanager.*` / `srunner.tools.*` / `srunner.scenarios.basic_scenario`
modules, the `agents.navigation.local_planner` helper, and the bundled
`custom_atomics_*` module. No other project-specific code is required.

## Weather

Each `route.xml` carries a `<weathers>` block with keyframes at
`route_percentage` 0 and 100 holding identical values, i.e. constant weather
for the whole route. The nine parameters (`cloudiness`, `precipitation`,
`precipitation_deposits`, `wind_intensity`, `sun_azimuth_angle`,
`sun_altitude_angle`, `fog_density`, `fog_distance`, `wetness`) are the exact
values of the named `carla.WeatherParameters` preset.

Note that in CARLA 0.9.15 route mode, `precipitation_deposits` and `wetness`
are visual only — they do not change tyre grip. Scenarios that genuinely
reduce grip do so in code, via a friction trigger or a tyre-friction write.

## Ego control

Some scenarios actively govern the ego vehicle's speed (a speed cap and/or a
minimum-speed floor applied through `apply_control`) so that the hazard is
encountered at a repeatable closing speed. This is a property of the scenario
harness rather than of the hazard, and it affects comparisons between routes
that use it and routes that do not.
