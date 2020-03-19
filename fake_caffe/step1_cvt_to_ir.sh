#!/bin/bash

#python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_caffe.py -d ./fake_model.prototxt -m ./fake_model.caffemodel --data_type FP32 --model_name yolo_v2_uint8_int8_weights_pertensor

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_caffe.py -d ./fake_model.prototxt -m ./fake_model.caffemodel --data_type FP32


