# Train DriveVLA

## Convert uniad results from pth to json for LLaVA training

```shell
python drivevla/data_converter/uniad_converter/uniad_results_converter.py --nuscenes_split train
python drivevla/data_converter/uniad_converter/uniad_results_converter.py --nuscenes_split val
```

## Finetune DriveVLA

```shell
bash drivevla/finetune/finetune_eval_qwen2_5_0_5b_slurm.sh
```
