### 单文件测试模式

普通单文件脚本会依次导出规范化 CSV、DR02 关键点 PKL、DR02 动作 CSV 和 NPZ；
完成后默认启动 MuJoCo 对照查看器，同时显示原始 MotionDecode G1（+Y）和本次生成的
DR02（-Y）。

```bash
scripts/dr02_data_converter/convert_single_motiondecode_to_dr02.sh \
  --input dataset/MotionDecode/samples/4.Martial_Arts/Muay_Thai/MA_Muay_Thai.csv \
  --fps 120 \
  --overwrite
```

## DR02 两阶段 coarse-to-fine IK 测试

下面的独立测试流水线使用 `config/robot/DR02_two_stage_optimization_test.yaml`，不会修改或启用普通 `DR02.yaml`。默认在最后打开 MuJoCo，同时显示 G1（+Y）与重定向后的 DR02（-Y）。

```bash
PYTHON_BIN=/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  bash scripts/dr02_data_converter/convert_single_motiondecode_to_dr02_two_stage_optimization.sh \
  --input 'dataset/MotionDecode/samples/4.Martial_Arts/Eagle_Claw_Fist_2/MA_Eagle_Claw_Fist_2.csv' \
  --fps 120 \
  --output-name eagle_claw_fist_2_two_stage
```

- `--input PATH`：必填，MotionDecode 的 G1 CSV。
- `--fps FPS`：各阶段共同使用的源帧率，默认 `120`。
- `--output-name NAME`：可选的输出文件名 stem；缺省时采用输入 CSV 文件名。必须是单个文件名，不能包含 `/`。
- `--no-compare`：仅完成 adapter、关键点、两阶段 IK 与 NPZ 导出，不打开最终 G1/DR02 MuJoCo 对比窗口；`--no-viewer` 是兼容别名。
- `--overwrite`：允许替换该独立 two-stage 测试目录内同名的输出。
- `PYTHON_BIN`：可选环境变量，用于指定 Python 解释器。

二阶段开关与粗阶段收敛参数位于 `two_stage_optimization`：`enabled` 控制是否运行首帧初始化，`application: first_frame_only` 表示只有第 0 帧会先运行粗阶段；`initialize_root_from_target: true` 会在粗优化前把 floating root 的 xyz 与 `root_target_keypoint`（默认 `hips_mean`）的 yaw 对齐，并清除 roll/pitch。`coarse.max_additional_iterations` 是首次求解之后最多允许的额外求解次数（允许 `0`，`10` 忠实对应 GMR 的循环语义），`coarse.convergence_threshold` 控制停止条件。第 0 帧粗阶段只使用 YAML 明列的 `coarse.ik_match_table` FrameTask 和基础关节限位，随后在同一构型上进行细优化；其细优化保留普通 FrameTask/接触任务，但不使用 PostureTask 或 FrameVelocityLimit。第 1 帧及以后完全使用项目原有 `ik_match_table`、接触锁定、时间连续性正则及帧间速度硬约束。

25 动作批量 QA 可用 `scripts/dr02_data_converter/compare_dr02_retarget_quality.py` 比较同一目标 PKL 下的基线与候选 CSV，输出有限值、完整 fine-task、四末端位置/方向误差、关节限位和速度预算指标。例如：

```bash
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_dr02_retarget_quality.py \
  --keypoints-pkl output_data/keypoints/DR02_two_stage_optimization_test/MA_Eagle_Claw_Fist_2_from_g1_keypoints.pkl \
  --baseline-csv output_data/robot_motion/MA_Eagle_Claw_Fist_2_from_g1_DR02_temporal_constraints_test.csv \
  --candidate-csv output_data/robot_motion/MA_Eagle_Claw_Fist_2_from_g1_DR02_two_stage_optimization_test.csv \
  --output-json /tmp/eagle_quality.json --output-csv /tmp/eagle_quality.csv
```

对已生成的结果进行非 GUI 校验：

```bash
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/4.Martial_Arts/Eagle_Claw_Fist_2/MA_Eagle_Claw_Fist_2.csv' \
  --target-csv 'output_data/robot_motion/eagle_claw_fist_2_two_stage_from_g1_DR02_two_stage_optimization_test.csv' \
  --source-fps 120 --target-fps 120 --render-fps 120 --validate-only
```

关闭转换后的对照查看器（适合批处理或无图形界面环境）：

```bash
scripts/dr02_data_converter/convert_single_motiondecode_to_dr02.sh \
  --input dataset/MotionDecode/samples/4.Martial_Arts/Muay_Thai/MA_Muay_Thai.csv \
  --fps 120 \
  --no-compare \
  --overwrite
```

`--no-compare` 仅跳过最后的 G1/DR02 对照，不影响 CSV、PKL 或 NPZ 输出。
旧参数 `--no-viewer` 仍可用，但已弃用，等价于 `--no-compare`。

### 带时间连续性与帧间速度约束的单文件测试

该命令先使用 `config/robot/DR02.yaml` 生成 DR02 目标关键点，再使用
`config/robot/DR02_temporal_constraints_test.yaml` 执行带时间连续性正则和
帧间关节速度硬约束的 IK，最后导出测试版 CSV、NPZ，并默认打开 MuJoCo
对照查看器，同时显示原始 MotionDecode G1 和本次生成的受约束 DR02。

```bash
scripts/dr02_data_converter/convert_single_motiondecode_to_dr02_temporal_velocity_constraints.sh \
  --input dataset/MotionDecode/samples/4.Martial_Arts/Muay_Thai/MA_Muay_Thai.csv \
  --fps 120 \
  --overwrite
```

关闭对照查看器：

```bash
scripts/dr02_data_converter/convert_single_motiondecode_to_dr02_temporal_velocity_constraints.sh \
  --input dataset/MotionDecode/samples/4.Martial_Arts/Muay_Thai/MA_Muay_Thai.csv \
  --fps 120 \
  --no-compare \
  --overwrite
```

启动项说明：

| 启动项 | 是否必需 | 默认值 | 说明 |
|---|---|---|---|
| `--input PATH` | 是 | 无 | 输入的 MotionDecode G1 CSV。文件必须存在且扩展名为 `.csv`。 |
| `--fps FPS` | 否 | `120` | 输入动作的真实采样率。该值会传给接触检测、帧间速度预算和最终 NPZ，必须是正数。 |
| `--output-name NAME` | 否 | 输入 CSV 的文件名（不含扩展名） | 自定义本次测试输出的文件名前缀。只能是单个文件名，不能包含 `/`。适合对同一输入测试多组参数时避免重名。 |
| `--no-compare` | 否 | 未启用 | 跳过最后的 G1/DR02 MuJoCo 对照，适合服务器、自动测试和批处理。 |
| `--no-viewer` | 否 | 未启用 | 已弃用；等价于 `--no-compare`，仅为兼容旧调用保留。 |
| `--overwrite` | 否 | 未启用 | 允许替换同名测试产物。未启用且任一目标文件已存在时，脚本会停止，避免误覆盖。 |
| `-h`, `--help` | 否 | 未启用 | 显示脚本帮助并退出。 |

查看帮助：

```bash
scripts/dr02_data_converter/convert_single_motiondecode_to_dr02_temporal_velocity_constraints.sh --help
```

测试产物目录：

```text
output_data/dr02_data_converter/temporal_constraints_test/normalized/  # 规范化后的 G1 CSV
output_data/keypoints/DR02_temporal_constraints_test/                  # DR02 目标关键点 PKL
output_data/robot_motion/                                              # 带约束的 DR02 qpos CSV
output_data/dr02_data_converter/temporal_constraints_test/npz/         # 强化学习 NPZ
```

默认输出文件名会包含 `DR02_temporal_constraints_test`，因此不会与普通
`DR02.yaml` 流程产生的 `_DR02.csv` 混淆。若已经激活 `robot_retargeter`
Conda 环境，无需额外设置 `PYTHON_BIN`；如需指定解释器，可在命令前设置：

```bash
PYTHON_BIN=/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
scripts/dr02_data_converter/convert_single_motiondecode_to_dr02_temporal_velocity_constraints.sh \
  --input dataset/MotionDecode/samples/4.Martial_Arts/Muay_Thai/MA_Muay_Thai.csv \
  --fps 120 \
  --no-compare
```

### 批量数据转换测试
```
scripts/dr02_data_converter/run_dr02_batch.sh \
  --input-dir dataset/MotionDecode/samples/1.1.Basic_Movement_Category/1.1.1.High_Dynamic_Movement/1.1.1.1.Standing_High_Jump \
  --fps 120 \
  --jobs 4
```
### 重定向数据展示
```
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv \
  dataset/MotionDecode/samples/4.Martial_Arts/Muay_Thai/MA_Muay_Thai.csv \
  --target-csv \
  output_data/robot_motion/4-martial-arts-muay-thai-ma-muay-thai-e6e16bd6cfe7_from_g1_DR02.csv \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 120

/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv \
  dataset/MotionDecode/samples/1.1.Basic_Movement_Category/1.1.1.High_Dynamic_Movement/1.1.1.1.Standing_High_Jump/BM_Standing_High_Jump_00092.csv \
  --target-csv \
  output_data/robot_motion/1.1.Basic_Movement_Category/1.1.1.High_Dynamic_Movement/1.1.1.1.Standing_High_Jump/1-1-basic-movement-category-1-1-1-high-dynamic-movement-1-1-1-1-standing-high-jump-bm-standing-h-0010fca813c7_from_g1_DR02.csv \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60
```
### 25 条明显关节异常动作：重定向前后对比

说明：以下动作按自动 QA 严重程度排序。G1 位于 +Y，DR02 位于 -Y；播放到每条命令标注的重点时间后按空格暂停，再用 `,` / `.` 逐帧检查。

```bash
# [01] 武术：MA_Eagle_Claw_Fist_2
# 重点观察：t=22.208s，frame=2665；right_shoulder_z_joint 连续贴限 1819 帧；手臂 P95=25.5cm/141.3°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/4.Martial_Arts/Eagle_Claw_Fist_2/MA_Eagle_Claw_Fist_2.csv' \
  --target-csv 'output_data/robot_motion/qa06_eagle_claw_fist_2_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [02] 球类：BGI_Serve_00155
# 重点观察：t=6.175s，frame=741；left_shoulder_z_joint 连续贴限 497 帧；手臂 P95=10.0cm/161.0°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/3.3.Ball_Game_Interaction/3.3.6.Tennis/3.3.6.3.Serve/BGI_Serve_00155.csv' \
  --target-csv 'output_data/robot_motion/qa23_tennis_serve_155_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [03] 球类：BGI_Smash_00221
# 重点观察：t=5.933s，frame=712；left_shoulder_z_joint 连续贴限 413 帧；手臂 P95=10.8cm/156.0°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/3.3.Ball_Game_Interaction/3.3.2.Badminton/3.3.2.2.Smash/BGI_Smash_00221.csv' \
  --target-csv 'output_data/robot_motion/qa24_badminton_smash_221_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [04] 武术：MA_Long_Fist_3
# 重点观察：t=47.975s，frame=5757；left_shoulder_z_joint 连续贴限 480 帧；手臂 P95=10.6cm/87.9°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/4.Martial_Arts/Long_Fist_3/MA_Long_Fist_3.csv' \
  --target-csv 'output_data/robot_motion/qa02_long_fist_3_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [05] 球类：BGI_Serve_00005
# 重点观察：t=8.742s，frame=1049；left_shoulder_z_joint 连续贴限 800 帧；手臂 P95=11.3cm/163.6°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/3.3.Ball_Game_Interaction/3.3.6.Tennis/3.3.6.3.Serve/BGI_Serve_00005.csv' \
  --target-csv 'output_data/robot_motion/qa22_tennis_serve_05_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [06] 舞蹈：DAN_havana
# 重点观察：t=67.708s，frame=8125；right_shoulder_z_joint 连续贴限 775 帧；手臂 P95=8.7cm/44.0°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/5.Dance/5.3.Pop_Dance/20260108/havana/DAN_havana.csv' \
  --target-csv 'output_data/robot_motion/qa18_havana_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [07] 武术：MA_Muay_Thai
# 重点观察：t=29.567s，frame=3548；left_shoulder_z_joint 连续贴限 1141 帧；手臂 P95=8.8cm/90.6°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/4.Martial_Arts/Muay_Thai/MA_Muay_Thai.csv' \
  --target-csv 'output_data/robot_motion/qa01_muay_thai_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [08] 武术：MA_wubuquan_1
# 重点观察：t=20.717s，frame=2486；left_shoulder_z_joint 连续贴限 437 帧；手臂 P95=9.8cm/78.4°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/4.Martial_Arts/wubuquan_1/MA_wubuquan_1.csv' \
  --target-csv 'output_data/robot_motion/qa04_wubuquan_1_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [09] 情绪/表演：EESB_Intense_Joy_00058
# 重点观察：t=20.800s，frame=2496；left_shoulder_z_joint 连续贴限 426 帧；手臂 P95=8.8cm/77.6°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/1.10.Emotional_Expressions_and_Social_Behaviors/1.10.6.Dramatic_Emotional_Gestures/1.10.6.3.Intense_Joy/EESB_Intense_Joy_00058.csv' \
  --target-csv 'output_data/robot_motion/qa26_intense_joy_58_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [10] 球类：BGI_Spike_00112
# 重点观察：t=7.658s，frame=919；left_shoulder_z_joint 连续贴限 537 帧；手臂 P95=8.5cm/159.4°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/3.3.Ball_Game_Interaction/3.3.4.Volleyball/3.3.4.3.Spike/BGI_Spike_00112.csv' \
  --target-csv 'output_data/robot_motion/qa20_volleyball_spike_112_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [11] 舞蹈：DAN_bad
# 重点观察：t=68.900s，frame=8268；left_shoulder_z_joint 连续贴限 560 帧；手臂 P95=8.6cm/72.0°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/5.Dance/5.3.Pop_Dance/20260327/bad/DAN_bad.csv' \
  --target-csv 'output_data/robot_motion/qa12_bad_dance_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [12] 武术：MA_Long_Fist_2
# 重点观察：t=37.575s，frame=4509；right_shoulder_y_joint 连续贴限 413 帧；手臂 P95=8.7cm/23.6°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/4.Martial_Arts/Long_Fist_2/MA_Long_Fist_2.csv' \
  --target-csv 'output_data/robot_motion/qa07_long_fist_2_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [13] 情绪/表演：EESB_Intense_Joy_00041
# 重点观察：t=15.183s，frame=1822；right_shoulder_y_joint 连续贴限 659 帧；手臂 P95=6.5cm/20.4°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/1.10.Emotional_Expressions_and_Social_Behaviors/1.10.6.Dramatic_Emotional_Gestures/1.10.6.3.Intense_Joy/EESB_Intense_Joy_00041.csv' \
  --target-csv 'output_data/robot_motion/qa30_intense_joy_41_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [14] 情绪/表演：EESB_Theatrical_Exaggerated_Emo_00018
# 重点观察：t=37.400s，frame=4488；left_shoulder_z_joint 连续贴限 666 帧；手臂 P95=7.3cm/67.0°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/1.10.Emotional_Expressions_and_Social_Behaviors/1.10.8.Performative_Emotions/1.10.8.1.Theatrical_Exaggerated_Emo/EESB_Theatrical_Exaggerated_Emo_00018.csv' \
  --target-csv 'output_data/robot_motion/qa27_theatrical_emotion_18_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [15] 舞蹈：DAN_cheating_on_you
# 重点观察：t=44.217s，frame=5306；left_shoulder_y_joint 连续贴限 534 帧；手臂 P95=6.8cm/34.2°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/5.Dance/5.3.Pop_Dance/20260327/cheating_on_you/DAN_cheating_on_you.csv' \
  --target-csv 'output_data/robot_motion/qa16_cheating_on_you_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [16] 球类：BGI_Spike_00171
# 重点观察：t=6.842s，frame=821；left_shoulder_z_joint 连续贴限 584 帧；手臂 P95=8.3cm/165.6°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/3.3.Ball_Game_Interaction/3.3.4.Volleyball/3.3.4.3.Spike/BGI_Spike_00171.csv' \
  --target-csv 'output_data/robot_motion/qa25_volleyball_spike_171_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [17] 武术：MA_Long_Fist_11
# 重点观察：t=13.167s，frame=1580；left_shoulder_z_joint 连续贴限 241 帧；手臂 P95=9.7cm/78.9°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/4.Martial_Arts/Long_Fist_11/MA_Long_Fist_11.csv' \
  --target-csv 'output_data/robot_motion/qa05_long_fist_11_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [18] 舞蹈：DAN_the_girls
# 重点观察：t=37.117s，frame=4454；left_shoulder_z_joint 连续贴限 332 帧；手臂 P95=6.5cm/30.1°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/5.Dance/5.3.Pop_Dance/D0325/the_girls/DAN_the_girls.csv' \
  --target-csv 'output_data/robot_motion/qa13_the_girls_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [19] 舞蹈：DAN_light_rain
# 重点观察：t=16.258s，frame=1951；left_shoulder_z_joint 连续贴限 253 帧；手臂 P95=7.8cm/31.4°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/5.Dance/5.3.Pop_Dance/D0325/light_rain/DAN_light_rain.csv' \
  --target-csv 'output_data/robot_motion/qa15_light_rain_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [20] 球类：BGI_Spike_00087
# 重点观察：t=6.883s，frame=826；left_shoulder_z_joint 连续贴限 631 帧；手臂 P95=7.6cm/161.7°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/3.3.Ball_Game_Interaction/3.3.4.Volleyball/3.3.4.3.Spike/BGI_Spike_00087.csv' \
  --target-csv 'output_data/robot_motion/qa21_volleyball_spike_87_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [21] 武术：MA_Tiger_Fist_8
# 重点观察：t=3.683s，frame=442；right_shoulder_z_joint 连续贴限 233 帧；手臂 P95=8.8cm/32.4°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/4.Martial_Arts/Tiger_Fist_8/MA_Tiger_Fist_8.csv' \
  --target-csv 'output_data/robot_motion/qa03_tiger_fist_8_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [22] 武术：MA_Mantis_Fist_1
# 重点观察：t=12.017s，frame=1442；right_shoulder_z_joint 连续贴限 233 帧；手臂 P95=8.5cm/35.5°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/4.Martial_Arts/Mantis_Fist_1/MA_Mantis_Fist_1.csv' \
  --target-csv 'output_data/robot_motion/qa11_mantis_fist_1_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [23] 舞蹈：DAN_lips_hips_kiss
# 重点观察：t=13.158s，frame=1579；right_shoulder_z_joint 连续贴限 142 帧；手臂 P95=6.1cm/21.1°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/5.Dance/5.3.Pop_Dance/D0325/lips_hips_kiss/DAN_lips_hips_kiss.csv' \
  --target-csv 'output_data/robot_motion/qa14_lips_hips_kiss_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [24] 舞蹈：DAN_running
# 重点观察：t=96.833s，frame=11620；right_shoulder_z_joint 连续贴限 240 帧；手臂 P95=6.0cm/20.4°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/5.Dance/5.3.Pop_Dance/D0403/running/DAN_running.csv' \
  --target-csv 'output_data/robot_motion/qa17_running_dance_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

# [25] 武术：MA_Long_Fist_6
# 重点观察：t=10.300s，frame=1236；right_elbow_joint 连续贴限 133 帧；手臂 P95=6.4cm/20.4°
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv 'dataset/MotionDecode/samples/4.Martial_Arts/Long_Fist_6/MA_Long_Fist_6.csv' \
  --target-csv 'output_data/robot_motion/qa09_long_fist_6_from_g1_DR02.csv' \
  --source-fps 120 \
  --target-fps 120 \
  --render-fps 60

```
