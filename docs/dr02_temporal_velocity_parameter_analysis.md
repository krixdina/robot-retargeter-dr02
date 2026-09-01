# DR02 temporal / velocity parameter full-QA analysis

Run date: 2026-08-31. This is a full rerun analysis of the 25 visibly abnormal motions listed in `command.md`; the user-edited YAML and all functional code remain unchanged.

## Decision

**Conditional software-test recommendation: `cost=10`, shoulder `8 rad/s`, elbow `8 rad/s` (B2).**

This is the lowest-error member among the tested, velocity-bounded non-dominated candidates.  It should not be copied to a robot safety configuration without hardware validation.  If elbow conservatism is more important than the small tracking difference, use **B3 (`c=10,s=8,e=6`)**; if an elbow limit above 8 rad/s is acceptable, **B4 (`c=10,s=8,e=10`)** has the slightly best tracking figures but is not preferable on a safety-first criterion.  `c=6.5,s=4,e=4` is a deliberately more conservative fallback, at a substantial tracking/motion penalty.

No tested setting cures the underlying reachability/range issue: shoulder hard-limit contact remains concentrated in the tennis serve and elbow contact remains concentrated in particular violent arm actions.  This is why the recommendation is conditional, rather than a claim of final tuning.

## Scope, inputs, and important comparability qualification

`command.md` contains 25 numbered QA actions: 24 are shoulder-led cases and only #25 (`MA_Long_Fist_6`) is elbow-led.  Consequently the set is deliberately useful for shoulder regression testing but is statistically unbalanced for selecting an elbow limit.  The source action, QA focus frame, and original symptom are defined in `command.md`; all 25 rows were used in each completed candidate.

| ID | candidate | temporal cost | shoulder limit | elbow limit | rows |
|---|---|---:|---:|---:|---:|
| B0 | historical baseline QA CSV | — | — | — | 25 |
| B1 | `c6p5_s4_e4` | 6.5 | 4 | 4 | 25 |
| B2 | `c10_s8_e8` | 10 | 8 | 8 | 25 |
| B3 | `c10_s8_e6` | 10 | 8 | 6 | 25 |
| B4 | `c10_s8_e10` | 10 | 8 | 10 | 25 |
| B5 | `c12_s8_e8` | 12 | 8 | 8 | 25 |

The B1–B5 files are full reruns under the respective settings. **B0 is not a same-version re-solve with constraints disabled**: it is the pre-existing QA target-CSV measurement.  Therefore it is a useful historical severity reference (and the denominator for arc/range ratios), but B0-to-Bx deltas must not be interpreted as a controlled causal estimate of temporal/velocity constraints alone.

Source results (each has one header plus 25 rows): `/tmp/dr02_tuning_full_B0_baseline.csv`, `/tmp/dr02_tuning_full_B1_c6p5_s4_e4.csv`, `/tmp/dr02_tuning_full_B2_c10_s8_e8.csv`, `/tmp/dr02_tuning_full_B3_c10_s8_e6.csv`, `/tmp/dr02_tuning_full_B4_c10_s8_e10.csv`, and `/tmp/dr02_tuning_full_B5_c12_s8_e8.csv`.  Their exact 150-row union (including all original columns plus `candidate`) is archived at `docs/data/dr02_temporal_velocity_full_results.csv`.

## Semantics and configuration audit

The active YAML is `config/robot/DR02_temporal_constraints_test.yaml`:

```yaml
temporal_continuity_regularization:
  enabled: true
  cost: 6.5
  joint_name_substrings: ["shoulder", "elbow", "wrist"]
inter_frame_velocity_limit:
  enabled: true
  joint_velocity_limits: {shoulder: 4.0, elbow: 4.0}
```

`cost=c` is Mink's residual scaling, not the direct least-squares coefficient: its squared-objective contribution is `||c e||²`, i.e. **c²**.  Thus 0.05 means 0.0025, 6.5 means 42.25, 10 means 100, and 12 means 144.  The early `0.05` value was an initial/test-YAML-scale value, while the early `10` was a coarse screening candidate; neither is evidence of a physically calibrated regularization coefficient.  In particular, 0.05 is effectively negligible relative to the task residual scales and must not be described as a meaningful continuity setting.

The current temporal selector includes **wrist** joints.  The reported `qdot`, speed-edge, and `qddot` diagnostics are nevertheless computed from `frame_velocity_limit.indices`, i.e. **shoulder/elbow (S/E) only**.  Wrist continuity can still change those S/E diagnostics indirectly through the coupled IK solution, and can change tracking, hand arc, and range.  Results therefore support this *combined temporal (shoulder/elbow/wrist) + shoulder/elbow velocity* configuration, not a result attributable to a pure S/E temporal regularizer.

The `4 rad/s` limit has no demonstrated DR02 hardware basis in this experiment.  It is a software test bound, as are 6/8/10 rad/s.  The available 19.38 rad/s actuator figure is only a supplied reference, not proof of certified joint, controller, thermal, collision, or payload capability.  Do not treat any of these as hardware-safe operating limits.

## Metrics and aggregation

Each per-motion CSV reports P95 Cartesian/orientation errors for both arms and hands; `near` and `hard` are joint-limit proximity/contact fractions; longest runs are frames; finite differences use 120 Hz.  “Track P95” below is the mean of the four per-motion P95 position values (two arms, two hands); “rot P95” is analogous.  It is a comparison statistic, not a pooled sample P95.

Two aggregations are reported:

* **Action-equal:** arithmetic mean of the 25 motion rows; every motion has equal influence.
* **Frame-equal:** each row weighted by its number of frames; long clips have more influence.

This distinction matters here because `DAN_running` has 14,146 frames whereas some sports clips have about 1,000.  Candidate ordering is stable enough for the conditional B2 recommendation, but all deployment decisions should retain the action-equal view.

## Aggregate results

### Action-equal (primary)

| candidate | track P95 cm ↓ | rot P95° ↓ | shoulder hard % ↓ | shoulder hard run ↓ | elbow hard % ↓ | elbow hard run ↓ | qdot P95 | edge % ↓ | qddot P95 ↓ | arc / B0 ↑ | range / B0 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 historical | 8.57 | 42.50 | 30.80 | 623 | 20.12 | 275 | 7.59 | 14.71 | 304.9 | 1.000 | 1.000 |
| B1 c6.5/s4/e4 | 16.05 | 61.90 | **15.99** | **276** | 10.13 | 158 | 4.00 | 19.80 | 69.9 | 0.950 | **0.877** |
| **B2 c10/s8/e8** | **12.92** | 62.75 | 19.08 | 309 | 10.86 | 166 | 5.73 | 2.49 | 67.5 | 0.973 | 0.797 |
| B3 c10/s8/e6 | 13.16 | 63.00 | 19.02 | 308 | 10.52 | 165 | **5.45** | 3.39 | **66.4** | 0.972 | **0.800** |
| B4 c10/s8/e10 | **12.84** | **62.60** | 18.94 | 309 | 10.98 | 166 | 5.70 | **2.14** | 68.1 | **0.973** | 0.795 |
| B5 c12/s8/e8 | 14.15 | 66.03 | 17.58 | 301 | **9.53** | **157** | 5.00 | **1.44** | **56.8** | 0.960 | 0.760 |

“edge %” is the runner's S/E speed-budget edge statistic, calculated from `frame_velocity_limit.indices`; `qdot` and `qddot` use the same S/E diagnostic set.  All B1–B5 runs report zero branch-jump frames/events; B0 reports 6.12/11.56 per action on average.  The maxima are constraint-active, not hardware validation.  The interpretation remains confounded by wrist temporal regularization because the IK solution is coupled.

### Frame-equal sensitivity check

| candidate | track P95 cm ↓ | rot P95° ↓ | shoulder hard % ↓ | elbow hard % ↓ | qdot P95 | qddot P95 ↓ | arc / B0 ↑ | range / B0 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 historical | 8.20 | 34.37 | 22.76 | 16.06 | 8.56 | 338.1 | 1.000 | 1.000 |
| B1 c6.5/s4/e4 | 16.77 | 61.82 | **11.63** | 8.51 | 4.00 | 79.1 | 0.939 | **0.908** |
| **B2 c10/s8/e8** | 13.44 | 63.36 | 14.46 | 8.72 | 6.18 | 74.1 | 0.965 | 0.814 |
| B3 c10/s8/e6 | 13.72 | 63.67 | 14.42 | **8.39** | **5.82** | **73.1** | 0.963 | **0.819** |
| B4 c10/s8/e10 | **13.36** | **63.18** | 14.30 | 8.86 | 6.15 | 74.7 | **0.965** | 0.813 |
| B5 c12/s8/e8 | 14.89 | 67.13 | 13.47 | **7.53** | 5.38 | **62.3** | 0.953 | 0.780 |

Increasing `c` from 10 to 12 improves limit/speed/acceleration measures but loses roughly 1.2 cm action-equal tracking and 4.3 percentage points of arm range relative to B2.  Reducing elbow limit 8→6 (B2→B3) has a small but directionally consistent tracking penalty and does not improve shoulder metrics; it only makes sense when the elbow-specific conservatism is required.  Raising elbow 8→10 (B2→B4) offers only marginal tracking gains while relaxing the requested elbow cap.

## Hard gates and non-dominated view

The numerical solver/output gates satisfied by all B1–B5 are: 25/25 completed rows, finite reported values, and zero branch-jump frames/events.  These are not robot-safety gates.  For the S/E speed gate, use an explicit numerical tolerance of **2e-4 rad/s**: observed maximum exceedances versus the applicable configured cap are B1 `+1.09828e-4`, B2 `+6.85967e-5`, B3 `+6.54092e-5`, B4 approximately `+0` at printed precision (10.000000000000), and B5 `+5.53038e-6 rad/s`.  Thus all candidates pass only the *tolerance-based* numerical speed check; they would not all pass a strict zero-exceedance check.  **Every candidate fails the limit-persistence gate**: its worst continuous hard-limit run is hundreds of frames (B2/B3/B4: shoulder 790 frames and elbow 481; B5: shoulder 788 and elbow 475).  No candidate passes a stronger feasibility gate such as “no hard-limit contact”: even B5 retains 17.58% shoulder and 9.53% elbow action-equal hard contact.

Likewise, B2–B5 retain a **1920 rad/s² peak finite-difference acceleration**.  In particular B4 and B5 have this peak, so neither can be described as smooth enough or safe based on this tuning.  The limit and acceleration gate failures rule out any claim that the anomaly has been solved; the parameter choice is only a bounded software-test trade-off.

Among B1–B5, no single candidate minimizes tracking error, both joint contact measures, motion preservation, and speed/acceleration simultaneously.  The practical non-dominated set is **B1, B2, B3, B4, B5** under those competing objectives:

* B1 wins conservatism and preserved range, but is clearly dominated in tracking by B2/B4 and is more likely to introduce lag/freeze.
* B2 is the balanced 8/8 choice and has the best tracking among candidates with elbow ≤8.
* B3 trades a small amount of tracking for elbow cap 6.
* B4 has the best tracking but relaxes elbow to 10.
* B5 wins several smoothness/contact metrics but over-regularizes motion and has the worst tracking among bounded candidates.

The “non-dominated” label depends on retaining physical-limit/contact and motion-preservation objectives.  If the only objective were track P95, B4 wins; if it were only smoothness/contact, B5/B1 would win.  This is precisely why the recommendation is conditional.

## Worst cases requiring review

The following table gives the per-candidate worst observed action for the requested failure modes (not a statement that every other action is acceptable).

| candidate | worst track P95 (cm) | worst shoulder hard % / run | worst elbow hard % / run | max acceleration case |
|---|---|---|---|---|
| B0 | #1 Eagle Claw 19.85 | #1 Eagle Claw 90.83% / 3614 | #1 Eagle Claw 42.93% / #14 Theatrical 596 | #4 Long Fist 3, 49,272 rad/s² |
| B1 | #1 Eagle Claw 25.97 | #5 Serve 00005 53.62% / 778 | #2 Serve 00155 38.45% / #22 Mantis 485 | #1 Eagle Claw, 960 rad/s² |
| B2 | #17 Long Fist 11 19.04 | #5 Serve 00005 54.45% / 790 | #2 Serve 00155 34.81% / #22 Mantis 481 | #1 Eagle Claw, 1920 rad/s² |
| B3 | #17 Long Fist 11 19.36 | #5 Serve 00005 54.45% / 790 | #2 Serve 00155 34.91% / #22 Mantis 481 | #1 Eagle Claw, 1920 rad/s² |
| B4 | #17 Long Fist 11 18.93 | #5 Serve 00005 54.45% / 790 | #2 Serve 00155 34.61% / #22 Mantis 481 | #1 Eagle Claw, 1920 rad/s² |
| B5 | #17 Long Fist 11 20.44 | #5 Serve 00005 54.31% / 788 | #10 Spike 00112 33.47% / #22 Mantis 475 | #3 Smash 00221, 1920 rad/s² |

For B2 specifically, review #17 for tracking, #5 for shoulder saturation, #2 and #22 for elbow saturation, #1 for acceleration and the original severe shoulder issue, plus #25 (`MA_Long_Fist_6`) because it is the only elbow-led entry.  The 25-action composition means the evidence for an elbow limit is weak beyond these few cases.

## Limitations and next validation

1. The run measures kinematic IK output only.  It does not test torque/current, closed-loop tracking latency, interpolation and controller rate, payload, collision, contacts, thermal limits, or emergency-stop behaviour.
2. Peak finite-difference acceleration is still 1920 rad/s² in B2–B5; a velocity cap alone does not constitute smooth command shaping.  Add/validate acceleration and jerk bounds before hardware use.
3. The baseline is historical QA output, not a matched re-solve.  Repeat a true same-code, same-target unconstrained baseline for a causal before/after comparison.
4. Temporal regularization currently includes wrists, and the candidate test varies cost and velocity together.  Run a factorial ablation (shoulder/elbow-only temporal vs. wrist included; cost fixed while speed varies) before attributing gains to a single joint group.
5. Add elbow-led and symmetric left/right elbow stress motions; 24 shoulder-led cases versus one elbow-led case cannot establish a general elbow budget.
6. Start hardware work below the chosen software limit, with measured joint-speed/torque/current logs and supervised stop criteria; promote a limit only after those measurements and visual review of the listed worst cases.

## Reproducibility

The candidate naming and all metrics are retained in the six `/tmp/dr02_tuning_full_B*.csv` files.  Each row includes source CSV, cached target keypoints path, full-frame window, fps, settings, errors, shoulder/elbow near/hard contact, speed/acceleration, branch jumps, IK statistics, and motion-retention ratios.  Re-run or archive those files before `/tmp` is cleared if a permanent raw artifact is required.
