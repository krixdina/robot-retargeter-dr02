# DR02 二阶段 IK 调试记录

## 最终结论

二阶段优化变差的根因不是粗阶段权重本身，而是早期实现中的目标与时序处理错误。
最终方案保留 GMR 的“粗解仅用于首帧初始化”思想：第 0 帧先用目标
`hips_mean` 初始化 floating root 的 xyz/yaw，再运行不含时间项的 coarse IK，最后
运行原项目的 fine IK；第 1 帧起严格回到既有单阶段时间连续性和帧间速度约束路径。

最后发现的关键 bug 是：root 初始化会改变机器人 FK，但第一次 `update_targets(0)`
已经为**非激活**接触任务写入了“初始化前 body pose”。因此在 root 初始化后、coarse
和 fine 求解前必须再次调用 `update_targets(0)`，刷新普通任务和接触任务目标。这个
接触目标刷新（contact refresh）是最终 25 动作结果所使用的实现。

不需要 adaptive fallback；本轮证据也不支持把问题归因为“继续细调粗阶段权重”。

## 根因演进

1. **每帧 coarse 与时间约束冲突**：粗阶段没有 `PostureTask` 或
   `FrameVelocityLimit`，却改变同一个 `Configuration`；细阶段仍必须相对上一数据帧
   满足速度盒，粗解会先消耗关节位移预算，造成边界姿态。
2. **partial projection 旧方案不成立**：只把 coarse 解的受限切空间分量裁回速度盒，
   却保留浮动根和未受限关节的位移，破坏了手臂链的协同几何，满足速度数值并不等于
   保留一个合理 IK 姿态。
3. **root-first 是兼容前提**：没有 root xyz/yaw 初始化时，首帧 coarse/fine 容易在
   错误的全局根位姿附近求解；初始化后第 0 帧才是有效的粗到细 warm start。
4. **stale contact target 是最后缺陷**：非激活接触项应追随当前机器人 body pose。
   root-first 后如果不刷新目标，它仍锁在 root-first 前 FK 的位置，fine 阶段于是引入
   一个不应存在的残差。现已在 root 初始化后刷新全部 normal/contact targets。

## 关键消融

Eagle Claw（前 600 帧，早期固定 `temporal=6.5`、肩/肘 4 rad/s）的诊断结果如下。
这些结果用于确定机制，不是最终 25 动作参数推荐。

|方案|fine error 前100|fine error 全段|手位置误差 前100 / 全段 (m)|手方向误差 前100 / 全段 (rad)|速度违规|
|---|---:|---:|---:|---:|---:|
|基线单阶段|1.851|0.736|0.200 / 0.050|0.941 / 0.364|0|
|首帧二阶段、未初始化 root|2.650|2.950|0.200 / 0.152|0.841 / 0.692|0|
|首帧二阶段 + root xyz/yaw|**0.943**|**0.581**|**0.075 / 0.029**|**0.487 / 0.285**|0|
|每帧二阶段 + 链投影|1.756|0.664|0.204 / 0.055|0.857 / 0.293|0|
|每帧二阶段 + 全局投影|1.813|0.672|0.201 / 0.054|0.908 / 0.303|0|

由此排除“单纯增加/重调 coarse 权重”作为主因：在相同粗权重下，是否 root-first 和
是否让后续帧保持 legacy 时间语义，影响远大于权重微调。

## 最终 25 动作 QA（contact refresh）

正式实现以同一目标 PKL 下的 baseline 与 two-stage candidate 对比。原始逐动作 JSON/CSV
报告位于 `output_data/qa25_two_stage_final/reports/*_contactrefresh.{json,csv}`，体积较大，
不纳入版本库。

|判据 / 指标|最终结果|
|---|---:|
|动作数 / 总帧数|25 / 126405|
|finite、关节限位违规、速度预算违规|25/25 finite / 0 / 0|
|fine-task 全轨改善 / 近乎持平|24 / 1（持平差值 +0.000007）|
|fine-task 前100|25/25 改善，平均变化 **-0.27121**|
|四末端位置全轨|22 改善 / 3 变差，平均变化 **-0.0002366 m**|
|四末端位置前100|23 改善 / 2 变差，平均变化 **-0.006875 m**|
|四末端方向全轨|22 改善 / 3 变差，平均变化 **-0.003862 rad**|
|四末端方向前100|20 改善 / 5 变差，平均变化 **-0.06358 rad**|

这里的“改善”定义为 candidate 指标减 baseline 指标小于零；前 100 帧不足 100 帧时使用
该动作全部可用帧。结论是首帧 contact refresh 对开头姿态尤其有效，同时没有破坏 frame1+
的速度/连续性硬约束。

## 当前语义与适用边界

- `two_stage_optimization.application` 仅接受 `first_frame_only`：第 0 帧为
  root-first → 刷新 normal/contact targets → coarse FrameTask +
  `ConfigurationLimit` → fine normal/contact tasks；粗阶段不添加 `PostureTask`、
  `FrameVelocityLimit` 或接触锁定任务。
- 第 0 帧没有上一数据帧，fine 阶段也不使用 temporal posture 或 FVL；第 1 帧及以后
  完全复用项目原单阶段任务、接触锁定、时间连续性和速度硬约束，并以第 0 帧最终解为锚。
- 粗权重仍是 GMR 风格的可调初始化值（头/肩方向 100、普通方向 10、脚位置 50、手位置
  10），并非跨动作或跨机器人保证最优的标定参数。
- 此方案改善目标跟踪和首帧初始化，不会消除动作本身超出 DR02 关节范围或可达性的限制。
  对长时间贴限、自碰或动力学可行性，仍需单独检查 MuJoCo 回放与下游控制器。

## 回归保障

测试覆盖：frame 0 调用顺序（初次 targets → root init → targets refresh → coarse/fine）、
非激活接触目标确实从 root-first 后 FK 刷新、frame1+ 的 temporal/FVL 锚点语义，以及质量
对比工具对 NaN/Inf 的显式拒绝和严格 JSON 输出。关闭或省略二阶段配置仍走未改动的 legacy
调用路径。
