# Week 11 Evaluation Pilot Report

Git SHA: `f4a4e0f41029c844bd75b6cba81060ba4c40e902`
Network: manhattan_midtown corridor (42430333, 42430338, 42430342, 42430344, 42430347)
Seeds per cell: 10 (seeds 1-10)
Decisions per episode: 360 (1800s simulated)
Baseline controller: fixed_time

**Scope**: a real, honest pilot subset, not the full ~1,400-run suite docs/design/11_evaluation.md section 11.4 specifies -- see this script's own module docstring for what scaling up means and why this size was chosen.

## Claims table

| Condition | Controller vs. baseline | Mean waiting time (s) | Baseline mean (s) | 95% CI (treatment) | Welch's t | p-value | Holm-Bonferroni significant? | Cohen's d |
|---|---|---|---|---|---|---|---|---|
| no_disruption | max_pressure vs fixed_time | 160.78 | 173.76 | [145.01, 176.54] | 1.723 | 0.1107 | no | 0.770 |
| no_disruption | actuated vs fixed_time | 151.27 | 173.76 | [142.80, 159.73] | 4.766 | 0.0002 | YES | 2.131 |
| accident | max_pressure vs fixed_time | 162.25 | 177.18 | [141.29, 183.21] | 1.495 | 0.1611 | no | 0.668 |
| accident | actuated vs fixed_time | 150.09 | 177.18 | [139.75, 160.44] | 4.586 | 0.0003 | YES | 2.051 |
| closure | max_pressure vs fixed_time | 157.47 | 167.26 | [142.38, 172.57] | 1.290 | 0.2182 | no | 0.577 |
| closure | actuated vs fixed_time | 145.45 | 167.26 | [140.04, 150.85] | 5.047 | 0.0001 | YES | 2.257 |
| construction | max_pressure vs fixed_time | 160.47 | 174.22 | [144.28, 176.65] | 1.784 | 0.1001 | no | 0.798 |
| construction | actuated vs fixed_time | 151.88 | 174.22 | [143.60, 160.15] | 4.806 | 0.0002 | YES | 2.149 |
| weather | max_pressure vs fixed_time | 160.82 | 173.73 | [145.07, 176.58] | 1.713 | 0.1125 | no | 0.766 |
| weather | actuated vs fixed_time | 151.20 | 173.73 | [142.77, 159.63] | 4.788 | 0.0002 | YES | 2.141 |

## How to reproduce

```bash
python scripts/run_evaluation_pilot.py
```

Deterministic given the same git SHA and seeds -- rerunning regenerates an identical claims table (docs/design/13_roadmap.md's Week 11 test: "report-generation reproducibility").
