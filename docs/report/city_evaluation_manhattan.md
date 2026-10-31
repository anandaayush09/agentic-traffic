# Manhattan controller comparison

Network: `manhattan`, 2670 signals controlled by each controller
Demand: synthetic `pm_peak`, scaled x0.25
Episode: 180 decisions = 900 simulated seconds, starting from an empty network
Seeds per controller: {'fixed_time': 5, 'actuated': 5, 'max_pressure': 5}
Teleport after: 300 s

## How to read this

- Demand is synthetic, not calibrated to real counts.
- The network starts empty, so a short episode measures how each controller copes while the network fills, not steady state.
- Waiting and travel time are averaged over every trip that departed, including vehicles still en route at the end (their time so far counts). Throughput counts finished trips only; queue length is sampled every 30 s on every approach lane.
- **Seeds only change SUMO's random choices (driver behaviour, insertion order); every run replays the same generated trips.** Variation between seeds is therefore far smaller than real day-to-day variation, so the confidence intervals, p-values and Cohen's d below are over-confident. Read them as "consistent on this demand", not as proof it generalises to other demand patterns. Independent demand draws would be needed for that.
- p-values are Holm-Bonferroni corrected across the comparisons below.

## Means per controller

| Controller | n | Waiting (s) | Travel (s) | Throughput (veh/h) | Mean queue (veh/lane) |
|---|---|---|---|---|---|
| fixed_time | 5 | 197.6 | 523.4 | 2015 | 0.030 |
| actuated | 5 | 107.4 | 501.0 | 2490 | 0.021 |
| max_pressure | 5 | 106.0 | 492.4 | 2708 | 0.015 |

## Comparison against fixed_time

| Metric | Controller | Mean | Baseline mean | 95% CI | p-value | Significant (Holm)? | Cohen's d |
|---|---|---|---|---|---|---|---|
| mean_waiting_time_s | actuated | 107.36 | 197.60 | [106.43, 108.29] | 0.0000 | YES | 149.41 |
| mean_waiting_time_s | max_pressure | 106.01 | 197.60 | [103.54, 108.49] | 0.0000 | YES | 63.63 |
| throughput_veh_per_hour | actuated | 2490.00 | 2015.40 | [2470.64, 2509.36] | 0.0000 | YES | -26.15 |
| throughput_veh_per_hour | max_pressure | 2707.80 | 2015.40 | [2686.96, 2728.64] | 0.0000 | YES | -37.08 |

## Reproduce

```bash
python scripts/run_city_evaluation.py --city manhattan --scale 0.25 --seeds 3 --decisions 180
```

Per-cell raw metrics are in `data/networks/<city>/runs/evaluation_cells.jsonl`.
