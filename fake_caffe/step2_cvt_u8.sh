#!/bin/bash

echo "Optimazer IR to U8"

ir_xml=fake_model.xml
ir_bin=fake_model.bin
cfg=fake_model_quantization.json

pot=/opt/intel/openvino/deployment_tools/tools/post_training_optimization_toolkit/main.py

python3 $pot -c $cfg
