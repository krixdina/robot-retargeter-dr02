# DR02 正式命令

所有正式转换使用 `config/robot/DR02.yaml`。

## 单文件转换

```bash
PYTHON_BIN=/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/convert_single_motiondecode_to_dr02.sh \
  --input dataset/MotionDecode/samples/4.Martial_Arts/Eagle_Claw_Fist_2/MA_Eagle_Claw_Fist_2.csv \
  --fps 120 --no-compare --overwrite
```

可选参数：`--output-name NAME` 指定输出 stem；`--no-viewer` 是 `--no-compare` 的兼容别名。省略 `--no-compare` 会在转换后打开 MuJoCo 对比。

## 批量转换

```bash
PYTHON_BIN=/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/run_dr02_batch.sh \
  --input-dir dataset/MotionDecode/samples/4.Martial_Arts \
  --fps 120 --jobs 4
```

使用 `--limit N` 限制文件数，`--overwrite` 重建已存在输出，`--fail-fast` 在首个失败后停止调度新任务。批处理会维护 manifest、日志、相对目录镜像和原子发布。

## G1/DR02 MuJoCo 对比

```bash
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_g1_dr02_mujoco.py \
  --source-csv dataset/MotionDecode/samples/4.Martial_Arts/Eagle_Claw_Fist_2/MA_Eagle_Claw_Fist_2.csv \
  --target-csv output_data/robot_motion/MA_Eagle_Claw_Fist_2_from_g1_DR02.csv \
  --source-fps 120 --target-fps 120 --render-fps 120
```

添加 `--validate-only` 可在无 GUI 环境校验输入而不启动查看器。

## DR02 质量检查

下例比较同一目标关键点生成的两条 DR02 轨迹；`--config` 显式指定正式配置。

```bash
/home/krixdina/miniconda3/envs/robot_retargeter/bin/python \
  scripts/dr02_data_converter/compare_dr02_retarget_quality.py \
  --keypoints-pkl output_data/keypoints/DR02/MA_Eagle_Claw_Fist_2_from_g1_keypoints.pkl \
  --baseline-csv output_data/robot_motion/baseline_DR02.csv \
  --candidate-csv output_data/robot_motion/MA_Eagle_Claw_Fist_2_from_g1_DR02.csv \
  --config config/robot/DR02.yaml \
  --output-json /tmp/dr02_quality.json --output-csv /tmp/dr02_quality.csv
```
