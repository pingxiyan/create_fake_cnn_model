#!/bin/bash

/opt/intel/openvino/deployment_tools/model_optimizer/mo_caffe.py -d ./fake_model.prototxt -m ./fake_model.caffemodel --data_type FP32


