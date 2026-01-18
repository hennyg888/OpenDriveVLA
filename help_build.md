## Help Build Notes
2025-11-19

If thirdparty modules, mmcv and mmdet3d, do not build automatically as triggered by pixi.toml, comment out the 2 lines building from "path = "third_party/\<module-name>" from the toml file and manually build them.

To manually build thirdparty modules, enter pixi shell, then enter module dir, and use `pip install .`


If starting pixi shell and encountering problems with flash-attn not finding torch, make sure only the `.pixi/envs` in the project dir is in use. Sometimes the system default python in `./local/usr` or some such other python also in PATH is erroneouslly used instead which separates dependent packages. If possible, remove the `site-packages` dir from other python installation to force all packages to belong to pixi's python. This python mismatch can also be the cause of packages built with numpy 2 vs numpy 1 incompatibility error.


If building mmcv-full with `MMCV_WITH_OPS="1"` but still encountering `ModuleNotFoundError: No module named 'mmcv._ext'` error, then manually build mmcv extensions by going into mmcv dir and using `python setup.py build_ext`
Once built, build files will persists across pixi shell restarts and cleans. To remove build files for thirdparty modules, use git module deep clean `git clean -dfx`

### With Inference Passing

if flash-attn doesn't install via pixi, comment out all mentions of flash-attn in pyproject.tom and pixi.toml and manually enter pixi shell then pip install flash-attn==2.5.7 or whichever version needed

exiting pixi shell then re-entering pixi shell sometimes will fix weird errors such as mmcv._ext compiled and installed but not importable and cannot be found

if running scripts in sub dirs and encountering errors like:
```
Failed to import llava_llama from llava.language_model.llava_llama. Error: cannot import name 'bev_pool_ext' from partially initialized module 'projects.mmdet3d_plugin.ops.bev_pool' (most likely due to a circular import) (/home/hhguo/OpenDriveVLA/.pixi/envs/default/lib/python3.10/site-packages/projects/mmdet3d_plugin/ops/bev_pool/__init__.py)
Traceback (most recent call last):
  File "/home/hhguo/OpenDriveVLA/debug/bev_test.py", line 10, in <module>
    from llava.utils import pad_bevfeature
  File "/home/hhguo/OpenDriveVLA/.pixi/envs/default/lib/python3.10/site-packages/llava/__init__.py", line 1, in <module>
    from .model import LlavaLlamaForCausalLM
```
try this pythonpath fix:
```
cd /home/hhguo/OpenDriveVLA
export PYTHONPATH=$PWD:$PYTHONPATH
python debug/bev_test.py
```
As running scripts not in the top level dir can cause module find errors.

if seeing "TypeError: 'MultiPolygon' object is not iterable"
just do "pip install shapely==1.8.5.post1", it's a bug with shapely package

