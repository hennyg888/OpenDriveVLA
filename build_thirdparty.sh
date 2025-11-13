#!/usr/bin/env bash
cd third_party/mmcv_1_7_2
pip install .
cd ../mmdetection3d_1_0_0rc6
pip install .
cd ../..