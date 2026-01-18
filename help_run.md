## Help Run Notes
to run a quick test run "python debug/bev_test.py" which will run vision tower but not in model context, in actual model context vision tower outputs are formatted further before feeding into downstream

to run evaluations run "bash scripts/eval_drivevla.sh checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct 1"
1st argument is path to checkpoint and 2nd argument is num GPUs e.g.:
```
bash scripts/eval_drivevla.sh /home/s56cai/ckpt/opendrivevla_0.5B_ckpt 8
```