### 单文件测试模式
```
scripts/dr02_data_converter/convert_single_motiondecode_to_dr02.sh \
  --input dataset/MotionDecode/samples/4.Martial_Arts/Muay_Thai/MA_Muay_Thai.csv \
  --fps 120 \
  --overwrite
```
### 批量数据转换测试
```
scripts/dr02_data_converter/run_dr02_batch.sh \
  --input-dir dataset/MotionDecode/samples/1.1.Basic_Movement_Category/1.1.1.High_Dynamic_Movement/1.1.1.1.Standing_High_Jump \
  --fps 120 \
  --jobs 4
```