# DR02 temporal/velocity parameter experiment

Run date: 2026-08-31. This report deliberately keeps the user-edited test YAML unchanged.
All 25 `command.md` QA rows were parsed: **25/25** source CSVs, baseline QA target CSVs,
and cached `output_data/keypoints/DR02/*_keypoints.pkl` were available. The experiment
therefore reused the cached targets and wrote only to `/tmp/dr02_temporal_parameter_experiment`.

## Method and interpretation

`temporal_continuity_regularization.cost=c` is Mink's residual scale, so its least-squares
coefficient is **c²** (0.05 -> 0.0025, 6.5 -> 42.25, 10 -> 100), not c. Each short-screen
candidate used six representative severe motions (01 Eagle Claw, 02 tennis serve, 03 badminton
smash, 07 Muay Thai, 10 volleyball spike, 24 running), a 480-frame window centred on the
annotated QA frame. Cost was screened with velocity off; velocity was then screened with cost 6.5.
This avoids an uninformative Cartesian sweep.

Position/orientation error is computed for the six arm mapping bodies against cached target
keypoints. `limit` counts values within 1e-4 rad of either URDF range. Velocity/acceleration use
120 Hz finite differences; the official supplied actuator speed reference is 19.38 rad/s. Motion
amplitude is the mean per-joint min--max range, guarding against selecting a result merely because
it stayed still. IK failures/nonfinite values were checked per run.

## Stage 1: representative screening

| candidate | arm pos P95 (m) | arm rot P95 (deg) | shoulder max speed | shoulder amplitude | elbow max speed | interpretation |
|---|---:|---:|---:|---:|---:|---|
| cost .05, velocity off | .127 | 103.9 | 182.30 | 2.268 | >10 | effectively negligible regularisation; large jumps |
| cost 3, velocity off | .133 | 105.4 | 116.80 | 2.225 | >10 | still impulsive |
| cost 5, velocity off | .135 | 105.8 | 84.20 | 2.164 | >10 | improvement, but large peaks |
| **cost 6.5, velocity off** | **.135** | **105.2** | **65.13** | **2.097** | >10 | elbow of cost trade-off |
| cost 8, velocity off | .142 | 105.8 | 51.56 | 2.038 | >10 | slightly less motion, worse position |
| cost 10, velocity off | .164 | 107.2 | 38.57 | 1.979 | over-regularized |
| c6.5, shoulder/elbow 4 | .179 | 67.3 | 4.00 | 1.745 | 4.00 | hard-safe but substantial lag/freeze risk |
| **c6.5, shoulder/elbow 10** | **.112** | **79.1** | **10.00** | **2.036** | **10.00** | best balanced tested limit |
| c6.5, shoulder/elbow 19.38 | .137 | 105.4 | 18.41 | 2.133 | 19.19 | close to unconstrained tracking |
| c6.5, shoulder10/elbow4 | .121 | 81.0 | 10.00 | 2.044 | 4.00 | elbow 4 costs motion |
| c6.5, shoulder4/elbow10 | .164 | 77.1 | 4.00 | 1.796 | 10.00 | shoulder 4 is the dominant lag source |

The selected normalization is a balanced qualitative score over normalized arm P95 error,
limit occupancy/longest streak, max jump/acceleration, speed-to-limit fraction, and amplitude
loss (no single hard-constraint pass can win). It selects **c=6.5, shoulder=10, elbow=10 rad/s**.
This is a software test limit (51.6% of the 19.38 rad/s reference), **not** a claim about the
hardware's certified operating limit.

## Stage 2: final candidate, complete 25-motion validation

Final c6.5/s10/e10 reran all **126,405 frames** in 222.24 s wall-clock IK time (single process;
no G1 replay). No nonfinite values or solver exceptions occurred. Aggregate arm errors were
mean/P95/max position **3.41/10.28/64.47 cm** and orientation **17.99/76.88/176.14 deg**.
Shoulder and elbow maximum finite-difference velocity were both 10.00 rad/s; max per-frame
jump was 0.08333 rad. Mean/max IK iterations were 3.51/51. These iteration-cap frames should
be reviewed as difficult fits, not called solver failures.

| # | motion | frames | pos P95 cm | rot P95 deg | shoulder limit/longest | s max | e max | IK mean/max |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|1|Eagle Claw Fist 2|3979|15.4|117.5|1869/337|10|10|3.85/51|
|2|Serve 00155|991|10.0|162.1|575/492|10|10|2.37/37|
|3|Smash 00221|920|5.1|14.7|0/0|10|10|2.28/51|
|4|Long Fist 3|5999|13.2|91.3|3161/472|10|10|4.38/51|
|5|Serve 00005|1451|10.7|164.1|911/795|10|10|2.22/36|
|6|havana|9637|11.8|59.6|3070/757|10|10|4.19/51|
|7|Muay Thai|4633|11.1|100.2|2054/1143|10|10|3.01/29|
|8|wubuquan 1|3584|10.2|29.6|488/164|10|10|3.53/51|
|9|Intense Joy 58|2711|10.4|84.0|974/412|10|10|4.00/34|
|10|Spike 00112|1189|8.6|159.5|643/532|10|10|2.02/51|
|11|bad|8550|10.7|69.3|2206/552|10|10|4.12/51|
|12|Long Fist 2|4970|10.4|34.9|974/385|10|10|3.74/51|
|13|Intense Joy 41|5870|8.8|32.2|1292/647|10|10|2.99/29|
|14|Theatrical Emotion 18|6194|9.2|71.5|835/662|10|10|2.68/51|
|15|cheating on you|8844|9.9|61.1|2849/519|10|10|3.83/50|
|16|Spike 00171|1115|11.0|165.8|1653/578|10|10|2.60/49|
|17|Long Fist 11|1702|13.8|64.5|436/170|10|10|6.06/51|
|18|the girls|6807|10.3|39.3|639/115|10|10|4.28/51|
|19|light rain|9504|9.6|53.0|3066/243|10|10|3.70/51|
|20|Spike 00087|1143|8.4|161.8|731/616|10|10|2.29/51|
|21|Tiger Fist 8|5044|10.8|49.8|1830/228|10|10|4.03/51|
|22|Mantis Fist 1|7213|11.6|45.9|1991/227|10|10|4.46/51|
|23|lips hips kiss|6776|8.4|29.4|533/138|10|10|3.91/51|
|24|running|14146|8.3|27.9|905/236|10|10|3.82/51|
|25|Long Fist 6|3433|9.5|33.1|246/54|10|10|3.29/50|

Limit contact persists because many target arm poses are beyond DR02 geometry/range; temporal and
velocity terms cannot make them feasible. Pure visual tuning also misses torque/current margins,
closed-loop tracking latency, interpolation/controller rates, collisions, thermal limits, and
contact dynamics. Hardware validation should begin below 10 rad/s and check those signals.

## Reproduction and checks

Temporary runner: `/tmp/dr02_temporal_experiment.py`; JSON/logs: `/tmp/dr02_temporal_parameter_experiment/`.
Representative command: `python /tmp/dr02_temporal_experiment.py --ids 1,2,3,7,10,24 --cost 6.5 --shoulder 10 --elbow 10 --crop 480 --out /tmp/.../velocity_10.json`.
Full command: omit `--crop`, use ids 1--25, and write `final_c6.5_s10_e10_all25.json`.

`python -m unittest tests/test_temporal_constraints.py` ran 8 tests: 7 pass, 1 fails because that
test still asserts eight selected DOFs while the user-edited YAML now includes `wrist`, selecting
14 DOFs. This is a stale expectation/config mismatch, not a numerical experiment failure.
