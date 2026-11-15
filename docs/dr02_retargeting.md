# DR02 重定向

本文档定义 MotionDecode G1 动作到 DR02 的正式重定向流程、配置语义和质量边界。正式配置唯一来源为 `config/robot/DR02.yaml`；不要再为日常运行复制 DR02 YAML。

## 正式参数

正式 B2 参数为：

```yaml
temporal_continuity_regularization:
  enabled: true
  cost: 10
inter_frame_velocity_limit:
  enabled: true
  joint_velocity_limits:
    shoulder: 8
    elbow: 8
```

temporal 项对相邻帧的姿态变化施加软约束，FVL（inter-frame velocity limit）在配置指定的关节上施加帧间速度硬预算。二者只约束非 floating-base 关节；接触检测与接触锁定继续使用同一份正式配置中的 `contact_*` 和 `contact_links`。

B2 来自同一模型、任务权重、接触处理和 120 Hz 条件下的 25 条完整动作匹配比较。它是在肘部预算不超过 8 rad/s 的候选中，四末端和上肢跟踪的实际折中。这不是硬件安全认证：即使使用 B2，仍可能出现较长的关节贴限段和明显的有限差分加速度峰值。

## 首帧二阶段优化

`two_stage_optimization` 只允许 `first_frame_only`。第 0 帧按以下顺序执行：

1. 用 `root_target_keypoint`（默认 `hips_mean`）初始化 floating root 的 xyz 和 yaw，并清除 roll/pitch。
2. 重新刷新 normal 任务和非激活 contact 任务的目标。
3. 运行只含 coarse `FrameTask` 与基础关节限位的粗求解，作为 warm start。
4. 在同一构型上运行正式 fine IK。

第二步不可省略。root 初始化会改变 FK；若沿用初始化前写入的非激活接触目标，fine 求解会额外承受过期的接触残差。粗阶段不使用 temporal posture、FVL 或 contact-lock 任务。第 1 帧及以后完全回到普通 fine IK、接触锁定、temporal 和 FVL 路径，并以第 0 帧最终解为锚点。

粗阶段权重是可调的初始化权重，并非跨动作或跨机器人的最优标定。二阶段改善的对象是首帧初始化和目标跟踪；它不会消除不可达动作、关节范围、自碰或动力学可行性问题。

## 正式运行

单文件转换统一通过 `scripts/dr02_data_converter/convert_single_motiondecode_to_dr02.sh` 完成：

```bash
PYTHON_BIN=/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/convert_single_motiondecode_to_dr02.sh \
  --input dataset/MotionDecode/samples/4.Martial_Arts/Eagle_Claw_Fist_2/MA_Eagle_Claw_Fist_2.csv \
  --fps 120 --no-compare --overwrite
```

流程依次执行 MotionDecode adapter、G1 replay 生成 DR02 目标关键点、DR02 IK、NPZ 导出和（除非 `--no-compare`）G1/DR02 MuJoCo comparison。`--no-viewer` 是 `--no-compare` 的兼容别名；`--output-name` 可指定单个输出 stem；`PYTHON_BIN` 可显式指定解释器。

批处理使用 `run_dr02_batch.sh`。它镜像输入相对目录，以稳定 staging 名称避免并发碰撞，并在原子发布后写入 JSONL manifest 与每项日志。默认跳过完整的既有输出；部分输出集会失败而非被误用，`--overwrite` 才会安全重建。

## QA 与验证边界

`compare_g1_dr02_mujoco.py` 用于 G1/DR02 回放或 `--validate-only` 输入校验；`compare_dr02_retarget_quality.py` 对同一目标 PKL 的两份 DR02 CSV 检查有限值、fine-task 残差、四末端误差、关节限位和 FVL 违规。质量工具会拒绝 NaN/Inf，且 JSON 输出不写入非有限数值。

历史匹配比较还以独立 FK 实现复算了四末端/上肢误差、限位、速度、加速度、edge、轨迹弧长和活动范围，150 个候选—动作结果均通过；报告与独立实现的最大角度差为 `3.2836e-4` 度（容差 `5e-4`）。这说明该软件评估的计算一致性，不代表机器人已通过碰撞、力矩/电流、控制器跟踪、加速度/jerk 整形、热限制、载荷或硬件安全验证。部署前仍应在 MuJoCo 回放和下游控制器中完成相应验证。

参数选择与二阶段调试的详细证据已保留在：

- [DR02 temporal / velocity parameter analysis](dr02_temporal_velocity_parameter_analysis.md)
- [DR02 时间连续性 / 速度参数分析](dr02_temporal_velocity_parameter_analysis_zh.md)
- [DR02 二阶段 IK 调试记录](two_stage_optimization_debugging_zh.md)
- [`docs/data/`](data/) 中的候选配置、逐动作结果、动作等权/帧数加权汇总、最差情况、manifest、provenance 和独立审计结果。
