# DR02 temporal / velocity parameter analysis (matched rerun)

Run date: 2026-09-01.  This revision replaces the earlier historical-B0 and
prefix-grouped tracking analysis.  Every candidate was rerun on the same 25
full, untrimmed MotionDecode actions, using frozen PKLs and the current code.

## Decision

**Conditional software-test choice: B2 (`cost=10`, shoulder `8 rad/s`, elbow
`8 rad/s`).**  It is the best four-end-effector and upper-limb tracking choice
among candidates whose elbow cap is no greater than 8 rad/s.  B3 is the more
conservative elbow alternative; B4 has marginally better tracking but permits
10 rad/s at the elbow.  This is not a hardware-safety recommendation: all
candidates still exhibit long hard-limit runs and B2--B5 retain a 1920 rad/s²
finite-difference peak acceleration.

## What B0 means now

B0 is a **matched unconstrained baseline**, not the old historical QA CSV.
For each of the 25 frozen PKLs it uses the same DR02 model, task weights,
contact handling, FPS (120 Hz), solver and source action as B1--B5, with:

```yaml
two_stage_optimization: {enabled: false}
temporal_continuity_regularization: {enabled: false}
inter_frame_velocity_limit: {enabled: false}
```

Thus B0-to-Bx differences are controlled comparisons of the tested temporal
and velocity settings.  `edge %` is intentionally N/A for B0 because it has
no configured speed budget.

## Candidates and reproducible artifacts

| candidate | temporal cost | shoulder cap | elbow cap | completed |
|---|---:|---:|---:|---:|
| B0 | off | off | off | 25/25 |
| B1 | 6.5 | 4 | 4 | 25/25 |
| B2 | 10 | 8 | 8 | 25/25 |
| B3 | 10 | 8 | 6 | 25/25 |
| B4 | 10 | 8 | 10 | 25/25 |
| B5 | 12 | 8 | 8 | 25/25 |

The frozen source/PKL hashes are in
`docs/data/dr02_temporal_velocity_matched_v2_manifest.json`; the 150-row result
table is `docs/data/dr02_temporal_velocity_matched_v2_results.csv`.  The former
historical result table is preserved, clearly separated, in
`docs/data/historical/dr02_temporal_velocity_historical_results.csv`.

## Corrected tracking metrics

The previous report incorrectly used a name-prefix group: every `left_*` body
was included in `left_arm`, and similarly for the right side.  This mixed legs
and arms and counted the hand again.  It affected both the old `track P95` and
`rot P95`; neither value is used below.

Every frame is now evaluated against an explicit one-to-one PKL-to-DR02 FK
pair.  Each per-motion group score is the arithmetic mean of its individual
pair P95s; the tables then average the 25 motion scores.  “P95” means the 95th
percentile of that pair's framewise errors: about 95% of frames have an error
at or below it, and about 5% are worse.  The final group value is therefore a
mean of P95s, not a pooled P95.

| group | explicit pairs |
|---|---|
| four end effectors | `left_fore_arm→left_hand_link`, `right_fore_arm→right_hand_link`, `left_foot_end_link→left_foot_end_link`, `right_foot_end_link→right_foot_end_link` |
| upper limb | `left/right_shoulder→left/right_shoulder_x_link`, `left/right_arm→left/right_elbow_link`, `left/right_fore_arm→left/right_hand_link` |

Position is Euclidean FK-to-PKL distance (cm); rotation is the shortest
quaternion angular distance (degrees).  The four-end-effector group checks
hand/foot placement; the upper-limb group makes a strange shoulder/elbow IK
branch visible even if a hand happens to reach its target.

## Limit, motion and aggregation metrics

`near` means at least one relevant joint is within 5° of either effective
joint limit; `hard` means within 0.001 rad (about 0.057°).  A `near/hard frame
%` is the fraction of frames for which at least one shoulder or elbow joint,
respectively, meets that condition.  `hard run` is the longest contiguous
hard-frame interval, in frames.  Both near and hard are retained below so the
5° warning region is not silently discarded.

`qdot P95` and `qddot P95` are P95 absolute shoulder/elbow finite-difference
velocity (rad/s) and acceleration (rad/s²).  `edge %` is the fraction of
shoulder/elbow velocity samples at least 99% of their configured cap.  `arc /
B0` is four-end-effector FK trajectory length relative to B0; `range / B0` is
the mean shoulder/elbow/wrist joint range relative to B0.  Ratios are best
near 1, not unboundedly high.  ↓ means lower is normally preferable; ↑ means
closer to 1 is preferable.  `qdot` has no arrow because both freezing and
spikes are undesirable.

Action-equal is the primary aggregation: compute a metric separately for each
motion, then take the ordinary mean of 25 values.  Hence a short sports clip
and the 14,146-frame running clip each have one vote.

## Action-equal results (primary)

| candidate | 4EE pos P95 cm ↓ | 4EE rot P95° ↓ | UL pos P95 cm ↓ | UL rot P95° ↓ | shoulder near/hard % ↓ | shoulder hard run ↓ | elbow near/hard % ↓ | elbow hard run ↓ | qdot P95 | edge % ↓ | qddot P95 ↓ | arc/B0 ↑ | range/B0 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 6.74 | 24.62 | 7.90 | 54.32 | 31.78 / 30.80 | 623 | 25.37 / 20.12 | 275 | 7.59 | N/A | 304.9 | 1.000 | 1.000 |
| B1 | 11.98 | 38.49 | 14.32 | 70.33 | **17.53 / 15.99** | **276** | 13.62 / 10.13 | 158 | 4.00 | 19.80 | 69.9 | 0.981 | **0.877** |
| **B2** | **9.71** | 38.58 | **11.38** | 68.87 | 20.92 / 19.08 | 309 | 15.71 / 10.86 | 166 | 5.73 | 2.49 | 67.5 | **0.990** | 0.797 |
| B3 | 9.90 | 38.69 | 11.54 | 69.28 | 20.88 / 19.02 | 308 | 15.24 / 10.52 | 165 | **5.45** | 3.39 | 66.4 | 0.989 | **0.800** |
| B4 | **9.64** | **38.56** | **11.32** | **68.50** | 20.79 / 18.94 | 309 | 15.88 / 10.98 | 166 | 5.70 | 2.14 | 68.1 | **0.990** | 0.795 |
| B5 | 10.43 | 40.42 | 12.26 | 70.92 | 19.43 / 17.58 | 301 | 14.21 / **9.53** | **157** | 5.00 | **1.44** | **56.8** | 0.986 | 0.762 |

The tracking cost of temporal/velocity constraints is now clear in a matched
comparison: B0 has the lowest tracking error but poor limit/acceleration
behavior.  B1 suppresses motion most strongly; B5 most strongly regularizes
the 8 rad/s family.  B2 is the practical compromise for an 8 rad/s elbow
budget.  B4's small tracking improvement is not enough to justify its higher
elbow cap on a safety-first software test.

## Verification and limitations

An independent FK audit recomputed all four-end-effector and upper-limb,
limit, velocity, acceleration, edge, arc and range values directly from every
saved qpos CSV and frozen PKL.  It has its own literal pairs, pickle/CSV
reader, MuJoCo FK and rotation-matrix angle calculation; it does not call the
production evaluator or `RobotRetarget`. All 150 candidate-motion rows pass;
the maximum reported-vs-audited difference is `3.2836e-4` degrees, the normal
near-zero `acos` roundoff between matrix and quaternion implementations
(tolerance `5e-4`). The row-level audit and summary are in
`docs/data/dr02_temporal_velocity_matched_v2_independent_audit.csv` and
`docs/data/dr02_temporal_velocity_matched_v2_independent_audit_summary.json`.

## Reproducibility details

The compact archive includes all 150 metrics, sample counts (`T*4` for 4EE and
`T*6` for UL), 25-motion manifest, six resolved YAML files, trajectory/status
hashes and the generation SHA (`43c4527`) separately from the current report
SHA. This distinction is deliberate: evaluation/audit validate a historical
generation snapshot rather than relabel it as newly generated. See
`docs/data/dr02_temporal_velocity_matched_v2_provenance.json`.

`dr02_temporal_velocity_matched_v2_action_equal.csv` contains the primary
motion-equal aggregates. `..._frame_weighted.csv` weights each per-motion
metric by its number of frames; it is explicitly a weighted mean of per-motion
P95 values, not a pooled P95. `..._worst_cases.csv` identifies, separately,
the worst 4EE/UL tracking, shoulder/elbow hard frame/run, and qddot P95/max
motions for each candidate. The table above additionally omits no `near`
data: both joint/frame/run near and hard columns are retained in the archive.

These are kinematic IK results only.  They do not validate collision,
torque/current, controller tracking, acceleration/jerk shaping, thermal
limits, payload, or hardware safety.  Long hard-limit runs and the residual
acceleration peaks require further feasibility work before deployment.
