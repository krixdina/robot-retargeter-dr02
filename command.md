# MotionDecode 批量转换说明

本目录的 `command.sh` 是 MotionDecode G1 CSV 到 DR02 强化学习 NPZ 的全量批量入口。它调用 `scripts/dr02_data_converter/run_dr02_batch.sh`，每个输入文件由独立进程直接转换并只发布最终 NPZ；不会生成规范化 CSV、关键点 PKL 或机器人动作 CSV 等中间数据。

最终 NPZ 中根部四元数采用 `wxyz` 顺序。

## 运行前提

- 在仓库根目录执行命令：`/home/krixdina/robot_retargeter`。
- MotionDecode 数据集目录必须存在，且待转换 CSV 位于该目录内。
- Python 环境必须安装本项目转换依赖（含 MuJoCo、Mink 等）。`command.sh` 会优先选择仓库的 `.conda/bin/python`，其次选择 `/home/krixdina/miniconda3/envs/robot_retargeter/bin/python`，最后使用命令搜索到的 `python`。
- 请预留最终 NPZ、逐文件日志和 JSONL manifest 所需空间；中间 CSV/PKL 不会占用空间。

## 最简全量运行

数据集位于默认目录 `dataset/MotionDecode/samples` 时：

```bash
bash command.sh
```

该命令固定以 `120` FPS 转换全部 CSV，默认并发数为 `4`。

## 常用运行方式

指定 MotionDecode 输入根目录（也决定输出路径的镜像层级）：

```bash
MOTIONDECODE_ROOT=/data/MotionDecode/samples bash command.sh
```

指定解释器与并发数，例如使用 8 个独立转换进程：

```bash
PYTHON_BIN=/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
JOBS=8 \
bash command.sh
```

小批量试跑 10 个文件：

```bash
LIMIT=10 bash command.sh
```

推荐先用一小批数据确认环境、输出和性能，再以合适的 `JOBS` 启动全量任务：

```bash
MOTIONDECODE_ROOT=/data/MotionDecode/samples \
PYTHON_BIN=/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
JOBS=4 LIMIT=10 \
bash command.sh

MOTIONDECODE_ROOT=/data/MotionDecode/samples \
PYTHON_BIN=/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
JOBS=8 \
bash command.sh
```

## `command.sh` 可调环境变量

| 变量 | 默认值 | 含义 | 示例 | 注意事项 |
|---|---|---|---|---|
| `MOTIONDECODE_ROOT` | `${仓库根目录}/dataset/MotionDecode/samples` | 输入 MotionDecode CSV 的数据集根目录。`command.sh` 同时将其传为底层脚本的 `--input-dir` 和 `--dataset-root`。 | `MOTIONDECODE_ROOT=/data/MotionDecode/samples bash command.sh` | 目录必须存在；因为二者相同，调用 `command.sh` 时只能从该根目录开始扫描，不能仅指定其中某个子目录。 |
| `JOBS` | `4` | 同时运行的独立转换进程数。 | `JOBS=8 bash command.sh` | 必须为正整数。每个进程都会加载 MuJoCo/IK；请按 CPU、内存和磁盘吞吐调整。`command.sh` 的默认值为 4，但显式设置时不再由底层脚本限制为最多 4。 |
| `LIMIT` | `0` | 最多转换的文件数量。 | `LIMIT=10 bash command.sh` | `0` 表示不限制、转换所有匹配 CSV；必须为非负整数。文件按路径排序后取前 N 个。 |
| `PYTHON_BIN` | 自动选择 | 用于运行转换和 manifest 工具的 Python 解释器。 | `PYTHON_BIN=/path/to/python bash command.sh` | 必须指向可运行且已安装项目依赖的解释器。 |

`command.sh` 不读取位置参数；例如 `bash command.sh --limit 10` 不会把 `--limit` 透传给底层脚本。请改用上面的 `LIMIT=10 bash command.sh` 形式。

此外，`command.sh` 固定向底层脚本传入 `--fps 120`，不能通过环境变量调整 FPS。

## 输出、日志与 manifest

`command.sh` 的输出根目录固定为：

```text
output_data/dr02_data_converter/
├── npz/                 # 最终 NPZ，镜像输入 CSV 的相对目录
├── logs/                # 每个输入 CSV 一个转换日志（.csv.log）
└── manifests/           # 本次运行的 JSONL 结果清单
```

例如输入：

```text
dataset/MotionDecode/samples/category/action/motion.csv
```

对应最终文件为：

```text
output_data/dr02_data_converter/npz/category/action/motion_from_g1_DR02.npz
```

NPZ 文件先写入同目录的临时文件，成功后才原子替换为最终文件；正常完成的结果不会依赖任何中间数据。

## 覆盖、失败与复跑

经由 `command.sh` 启动时，不会传递 `--overwrite` 或 `--fail-fast`，因此行为如下：

- 若最终 NPZ 已存在，该文件会被跳过，不会覆盖。
- 某一个文件转换失败时，其他已排队文件仍会继续执行；最后命令会以非零状态退出。
- 重跑同一命令会自动跳过已成功发布的 NPZ，并重试此前失败或尚未生成的文件。
- 每次运行都会生成一个带 UTC 时间戳的 JSONL manifest，记录 `success`、`failed`、`skipped` 状态、耗时和日志路径。

如需显式覆盖、失败后停止继续调度、单独指定输入子目录、输出镜像根、FPS、manifest 或文件匹配模式，请直接调用底层脚本。

## 底层批量脚本的完整接口

在仓库根目录执行：

```bash
scripts/dr02_data_converter/run_dr02_batch.sh --input-dir DIR [options]
```

它支持以下参数：

| 参数 | 默认值 | 含义与示例 | 注意事项 |
|---|---|---|---|
| `--input-dir DIR` | 无，必填 | 扫描该目录及其子目录中的输入文件。示例：`--input-dir dataset/MotionDecode/samples/4.Martial_Arts` | 必须存在，且必须位于 `--dataset-root` 内。 |
| `--dataset-root DIR` | `dataset/MotionDecode/samples` | 用于计算输入相对路径并镜像到 NPZ 输出目录。示例：`--dataset-root /data/MotionDecode/samples` | 必须存在；不要把它设为 `--input-dir` 之外的目录。 |
| `--fps FPS` | `120` | 源动作采样率。示例：`--fps 120` | 必须为正整数。 |
| `--jobs N` | `min(4, CPU 核数)` | 独立转换进程数。示例：`--jobs 8` | 必须为正整数。 |
| `--manifest PATH` | `output_data/dr02_data_converter/manifests/dr02_batch_<UTC 时间戳>.jsonl` | 本次 JSONL 结果清单位置。示例：`--manifest output_data/dr02_data_converter/manifests/full_run.jsonl` | 路径必须位于仓库的 `output_data/` 下；相对路径按仓库根目录解释。 |
| `--overwrite` | 未启用 | 覆盖已存在的最终 NPZ。 | 仅替换最终 NPZ；请确认目标范围后再使用。 |
| `--limit N` | `0` | 最多处理 N 个文件。示例：`--limit 10` | `0` 表示全部；按排序后的路径取前 N 个。 |
| `--max-files N` | `--limit` 的别名 | 示例：`--max-files 10` | 与 `--limit` 行为相同。 |
| `--pattern GLOB` | `*.csv` | 按文件名 glob 筛选。示例：`--pattern 'MA_*.csv'` | 使用单引号避免 shell 提前展开；匹配不区分扩展名大小写。 |
| `--fail-fast` | 未启用 | 一旦已完成的转换进程报告失败，停止继续调度后续文件。 | 已经启动的并发任务仍会等待完成。 |
| `-h`、`--help` | 未启用 | 显示帮助并退出。 | 无。 |

底层脚本同样识别 `PYTHON_BIN` 环境变量。其自动选择顺序为：仓库 `.conda/bin/python`、仓库 `conda/bin/python`、`python`、`python3`。

### 指定输入子目录和输出镜像根

底层脚本的最终输出根仍固定为 `output_data/dr02_data_converter/npz/`；没有支持自定义 NPZ 输出根的参数。可以通过 `--dataset-root` 控制该固定输出根下的镜像相对路径：

```bash
PYTHON_BIN=/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
scripts/dr02_data_converter/run_dr02_batch.sh \
  --input-dir /data/MotionDecode/samples/4.Martial_Arts \
  --dataset-root /data/MotionDecode/samples \
  --fps 120 \
  --jobs 8 \
  --manifest output_data/dr02_data_converter/manifests/martial_arts.jsonl
```

### 覆盖已有结果并启用 fail-fast

```bash
PYTHON_BIN=/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
scripts/dr02_data_converter/run_dr02_batch.sh \
  --input-dir /data/MotionDecode/samples \
  --dataset-root /data/MotionDecode/samples \
  --jobs 8 \
  --overwrite \
  --fail-fast
```

## 后台运行、查看进度与中断

可将标准输出和标准错误写入总日志，在后台执行：

```bash
mkdir -p output_data/dr02_data_converter/logs
nohup bash command.sh > output_data/dr02_data_converter/logs/batch_launcher.log 2>&1 &
```

查看启动总日志：

```bash
tail -f output_data/dr02_data_converter/logs/batch_launcher.log
```

查看某个输入文件的详细转换日志（路径与输入 CSV 的相对路径一致，末尾添加 `.log`）：

```bash
tail -f output_data/dr02_data_converter/logs/4.Martial_Arts/Muay_Thai/MA_Muay_Thai.csv.log
```

按 `Ctrl-C`（前台）或向启动脚本进程发送 `SIGINT`/`SIGTERM`（后台）可中断任务。脚本会终止仍在运行的子进程并清理由这次中断留下的临时 NPZ；已成功发布的最终 NPZ 会保留。之后直接使用同一命令复跑即可跳过这些已完成文件。
