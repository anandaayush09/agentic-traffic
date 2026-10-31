# Training the MAPPO policy on Kaggle (or Colab)

The from-scratch MAPPO trainer is built and verified on short runs, but convergence needs
roughly 5-15M environment steps -- an overnight-to-weekend GPU/CPU session the dev machine
doesn't have ([ADR-0009](../adr/0009-mappo-from-scratch-not-rllib.md),
[ADR-0013](../adr/0013-r1-r2-r4-descope-outcomes.md)). This is how to run it in the cloud and
bring the result back. **Everything here is prepared but unverified on Kaggle itself** -- the
same scripts run locally on Windows; the Linux/notebook path is untested until you run it.

## What you get

A `mappo_district.pt` file (plain `state_dict`) and a resumable checkpoint folder. The policy is
a shared, decentralized actor, so one district-trained policy can drive any set of junctions.

## Steps

1. **Create a Kaggle notebook**, enable Internet and a GPU/CPU accelerator, and add this repo
   (clone it, or upload it as a dataset):
   ```python
   !git clone <your fork or the repo URL> atlas && cd atlas
   ```
2. **(Resuming only)** put the previous session's `checkpoints/` folder at
   `/kaggle/working/atlas_out/checkpoints/` (e.g. add last session's output as a dataset and
   copy it). Kaggle wipes the disk between sessions; without this the run starts from zero.
3. **Run the one-shot script** from the repo root:
   ```python
   !cd atlas && MAX_HOURS=8.5 bash scripts/kaggle_train.sh
   ```
   It installs the minimal training dependencies, rebuilds the road network and demand from
   the committed OSM extract, and trains. The run stops cleanly after the iteration that
   crosses `MAX_HOURS`, saving a checkpoint and `mappo_district.pt`.
4. **Download** `/kaggle/working/atlas_out/` (the notebook's *Output* tab), or "Save Version"
   so the checkpoints persist as a dataset for step 2 next time.
5. **Repeat steps 2-4** until the reward curve flattens (the `train.iteration` log lines show
   `mean_reward`, `entropy` and `approx_kl`).

## Using the result

```powershell
# watch it control the live map (adds an "RL" option to each junction's dropdown)
$env:ATLAS_RL_CHECKPOINT = "C:\path\to\mappo_district.pt"
python scripts/run_demo.py

# measure it against the classical controllers (same statistics, Holm-Bonferroni corrected)
python scripts/run_evaluation_pilot.py --rl-checkpoint C:\path\to\mappo_district.pt
```

## Reading the evaluation honestly

- The pilot compares `rl` to fixed-time on the 5-junction corridor with the same seeds,
  scenarios and tests as the classical controllers. A converged policy should beat fixed-time;
  whether it beats **actuated** (which already beats fixed-time by ~13%) is the real question.
  Max-pressure/actuated are strong baselines; "RL beats fixed-time" alone proves little.
- A checkpoint from a few iterations loads and runs identically to a converged one -- it will
  just lose. The report records the checkpoint name and iteration so that is visible.
- Every RL action goes through the safety shield, so even a bad policy cannot violate
  min/max green or skip a yellow; it can only waste green time.
