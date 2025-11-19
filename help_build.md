## Help Build Notes
2025-11-19

If thirdparty modules, mmcv and mmdet3d, do not build automatically as triggered by pixi.toml, comment out the 2 lines building from "path = "third_party/\<module-name>" from the toml file and manually build them.

To manually build thirdparty modules, enter pixi shell, then enter module dir, and use `pip install .`


If starting pixi shell and encountering problems with flash-attn not finding torch, make sure only the `.pixi/envs` in the project dir is in use. Sometimes the system default python in `./local/usr` or some such other python also in PATH is erroneouslly used instead which separates dependent packages. If possible, remove the `site-packages` dir from other python installation to force all packages to belong to pixi's python. This python mismatch can also be the cause of packages built with numpy 2 vs numpy 1 incompatibility error.


If building mmcv-full with `MMCV_WITH_OPS="1"` but still encountering `ModuleNotFoundError: No module named 'mmcv._ext'` error, then manually build mmcv extensions by going into mmcv dir and using `python setup.py build_ext`
Once built, build files will persists across pixi shell restarts and cleans. To remove build files for thirdparty modules, use git module deep clean `git clean -dfx`