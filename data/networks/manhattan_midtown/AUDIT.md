# Network audit: manhattan_midtown

Per [docs/design/08_data_strategy.md §8.1](../../../docs/design/08_data_strategy.md) --
this is the record of the Week 2 network-audit step: how the district was selected,
acquired, built, and checked before anything is built on top of it.

## Selection

Criteria applied, in order (per §8.1): OSM signal-tag completeness, lane-count tag
coverage, presence of a real bus-route grid, ~25 signalized intersections.

**Chosen: Midtown Manhattan, W 42nd-48th St between 6th and 9th Ave.** Reasoning:
Manhattan's avenue/street grid has some of the most complete `highway=traffic_signals`
and `lanes=*` tagging on OSM of any real district, and this particular sub-area is a pure
orthogonal grid (no diagonal streets like Broadway cutting through, which would have
added irregular 5-way/6-way junctions and complicated the "well-tagged, low-complexity"
goal this session picked over the "personally meaningful" alternative).

## Acquisition

- Source: OpenStreetMap via the Overpass API (`overpass.kumi.systems` mirror), bounding
  box `(40.7545, -73.9895, 40.7595, -73.9810)` (south, west, north, east).
- Data snapshot date: `2026-06-01` (from the Overpass response's `osm_base` metadata).
- Query: all `way[highway]` of type primary/secondary/tertiary/\*_link/residential in the
  bbox, plus every node they reference (so tagged `traffic_signals` nodes come along).
- The raw extract is committed at `raw.osm.xml` (~115 KB) rather than DVC-tracked, since
  DVC isn't wired up yet (deferred, see Week 1 milestone notes) and re-fetching later
  would silently drift as OSM is edited -- a dated, versioned snapshot is the honest
  choice until DVC lands.
- Regenerate the derived network with `python scripts/build_district_network.py`
  (`--refetch` to pull a fresh extract instead of using the committed one).

## Build

`netconvert` with SUMO's own OSM typemap (`data/typemap/osmNetconvert.typ.xml`, not the
binary's built-in fallback -- see script comments for why that distinction matters) and
standard cleanup flags: geometry simplification, roundabout/ramp guessing, junction
joining, and signal-guessing/joining for jointly-controlled junction clusters. Full flag
list: `scripts/build_district_network.py::NETCONVERT_FLAGS`.

**Sizing iteration:** the first extract (8 streets x 4 avenues) produced 40 signalized
junctions, well above the ~25-intersection design target (docs/design/03_system_
architecture.md). Trimmed to 6 streets x 3 avenues and rebuilt -> **22 signalized
junctions**, landing in spec.

## Audit results

| Check | Result |
|---|---|
| Total junctions | 58 |
| Signalized (`type="traffic_light"`) | 22 |
| Priority (unsignalized) | 8 |
| Dead-end (edges clipped at the bbox boundary) | 17 -- expected for any bounding-box extract, not a defect |
| Connected components (`netcheck.py`) | **1** (100% edge coverage) -- no islands, no unreachable sub-networks |
| Cycleway compound types | Discarded during import (7 warnings) -- cycle lanes aren't modeled; harmless, ATLAS does not simulate bicycles |
| TLS nodes dropped as non-controlling | 12 -- mid-block pedestrian-crossing signal tags that don't control a vehicle junction after import; normal `netconvert` cleanup, not a data defect |
| Loads and steps cleanly via `SumoTwinEnv` | Confirmed (50-step run, junction state reads correctly) |

**Known caveat:** `trafficlight.getIDList()` returns `GS_42439981` (a SUMO-assigned
grouped-signal id) rather than the raw OSM junction id `42439981` for one jointly-
controlled cluster -- `configs/network/manhattan_midtown.yaml`'s `traffic_light_ids` uses
the runtime ids, not the static XML ids; the two differ for exactly this one entry.

## Demand generation & plausibility invariants (part 2)

Four time-of-day periods generated via `scripts/generate_demand.py` from
`configs/demand/manhattan_midtown.yaml` (night/midday/am_peak/pm_peak, each 1800s,
`randomTrips.py --fringe-factor 5 --validate`, per-period seed). Turn-ratio "calibration"
is two mechanisms, not one: `--fringe-factor` biases generation toward through-traffic
(the shaping half); `atlas.twin.demand.turn_direction_counts` audits the result against
SUMO's own computed connection directions (the checking half) -- there is no real turn-
count ground truth to calibrate against, so this is a sanity check, not a fit-to-data
claim (see docs/design/08_data_strategy.md section 8.2's own "plausibility invariants,
not ground truth" framing).

| Check | Result |
|---|---|
| Vehicle count ordering | night (200) < midday (600) < am_peak (1385) < pm_peak (1637) |
| Turn direction mix (pm_peak) | straight 66%, left 20%, right 13%, U-turn 1.4% -- plausible, not degenerate |
| Congestion proxy (mean speed / allowed speed, sampled every 60 steps) | night 0.381 > midday 0.332 > am_peak 0.277 > pm_peak 0.190 -- monotonic with demand |

**Important calibration note:** the congestion proxy does **not** approach 1.0 (free
flow) even at night. This is a real property of a dense signalized grid (a traffic light
roughly every block), not a demand-generation defect -- most of a vehicle's average delay
here is waiting at red lights, not queueing behind other traffic, so the absolute scale
stays well below 1.0 regardless of volume. The invariant actually validated in tests is
therefore *relative* (monotonic ordering with demand, never near-total gridlock), not an
absolute "close to free-flow" target -- see `tests/unit/test_demand.py`'s
`TestCongestionProxy` docstring for the full reasoning. This is exactly the kind of wrong
initial assumption the plausibility-invariant approach is meant to catch before it
becomes a silently-miscalibrated training signal three weeks from now.

## Not yet done (next Week 2 sub-steps)

- Fleet mix (buses on real GTFS routes, trucks, emergency vehicles) -- currently
  passenger cars only.
- Scenario engine v1 (closure, accident, rush-hour disruption classes).

See [docs/design/13_roadmap.md](../../../docs/design/13_roadmap.md), Week 2.
