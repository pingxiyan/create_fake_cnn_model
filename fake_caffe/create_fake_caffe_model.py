# -*- coding: utf-8 -*-
"""
Convert a YOLO's .cfg to Caffe's .prototxt
"""
from __future__ import print_function, division

import argparse
import os
import sys
import numpy as np

if 'GLOG_minloglevel' not in os.environ:
    os.environ['GLOG_minloglevel'] = '2'  # suppress verbose Caffe output

from caffe import layers as cl
from caffe import params as cp
import caffe

def create_prototxt():
    """ Create some fake layers: http://ethereon.github.io/netscope/#/editor """
    layers = []

    # Layer1: "Input":
    data_fields = dict(shape={"dim": [1, 3, 416, 416]})
    layers.append(cl.Input(name="data", **data_fields))

    # Layer2: "Pooling":
    layers.append(cl.Pooling(layers[0], name="Pooling1", pool=cp.Pooling.MAX,
                             stride_h=7, stride_w=7,
                             kernel_h=3, kernel_w=3,
                             pad_h=1, pad_w=1))

    # layer3: "" ->
    layers.append(cl.Pooling(layers[1], name="Pooling2", pool=cp.Pooling.MAX,
                             stride_h=7, stride_w=7,
                             kernel_h=3, kernel_w=3,
                             pad_h=1, pad_w=1))

    # layer4: "" ->
    layers.append(cl.Pooling(layers[2], name="Pooling3", pool=cp.Pooling.MAX,
                             stride=9,
                             kernel_size=3))

    # layer5: "" ->
    layers.append(cl.Convolution(layers[3], name="conv1",
                                 kernel_size=1, stride=1, num_output=1))

    # layer6: "" ->
    layers.append(cl.Convolution(layers[4], name="conv2",
                                 kernel_size=1, stride=1, num_output=125*13*13))

    # layer7: "" ->
    reshape_fields = dict(reshape_param={"shape": {"dim": [-1, 125, 13, 13]}})
    layers.append(cl.Reshape(layers[5], name="reshape", **reshape_fields))

    model = caffe.NetSpec()
    for layer in layers:
        setattr(model, layer.fn.params["name"], layer)

    return model

def get_expect_result():
    zero_point = 221
    scale = 0.3371347486972809

    # Read the entire file as a single byte string

    with open('expected_result_sim.dat', 'rb') as fl:
        allBinData = fl.read()
    floatData = np.zeros(125*13*13)

    mutable_bytes = bytearray(allBinData)

    print(type(mutable_bytes))
    print(len(mutable_bytes))

    print((mutable_bytes[0] - zero_point) * scale)
    print((mutable_bytes[1] - zero_point) * scale)
    print((mutable_bytes[2] - zero_point) * scale)
    print((mutable_bytes[125*13*13-1] - zero_point) * scale)

    float_data = np.zeros(len(mutable_bytes))
    i = 0
    for c in mutable_bytes:
        float_data[i] = (c-zero_point)*scale;
        i += 1
    return float_data;

def create_fake_weights_by_prototxt(cafffe_deploy_fn, cafffe_weight_fn):
    print("Start convert_weights")
    # ======================================================
    net = caffe.Net(cafffe_deploy_fn, caffe.TEST)
    print("Declare net success by caffe deploy")
    count = 0
    print("=============================================")
    print("check parameter layer...")
    for name, layer in zip(net.top_names, net.layers):
        if name not in net.params.keys():  # layer without parameters
            print("The layer don't have parameter:", name)
            continue

        print("  creating fake param {}, {}".format(name, layer.type))
        if name == 'conv1':
            print("  =================================")
            print("  create fake Convolution 1 weights")
            print("  type(net.params[name][1].data)=", type(net.params[name][1].data))
            bais_sz = np.prod(net.params[name][1].data.shape);
            weight_sz = np.prod(net.params[name][0].data.shape);

            print("  need size =", bais_sz)
            print("  need size =", weight_sz, "layout =", net.params[name][0].data.shape)

            print("  src data =", net.params[name][1].data, "layout =", net.params[name][1].data.shape)
            net.params[name][1].data[...] = np.ones(net.params[name][1].data.shape)
            net.params[name][0].data[...] = np.zeros(net.params[name][0].data.shape)
            print("  dst data =", net.params[name][1].data, "layout =", net.params[name][1].data.shape)
        elif name == "conv2":
            # READ expected last layer data.
            print("  =================================")
            print("  create fake Convolution 2 weights")

            print("  weight shape =", net.params[name][0].data.shape)
            print("  bais shape =", net.params[name][1].data.shape)

            # bais, set your expect param to here
            # net.params[name][1].data[...] = np.ones(net.params[name][1].data.shape)
            net.params[name][1].data[...] = get_expect_result()
            # weight
            net.params[name][0].data[...] = np.zeros(net.params[name][0].data.shape)
        else:
            print("  WARNING: unknown type {} for layer {}".format(layer.type, name))

    print('Converted {0} weights.'.format(count))
    net.save(cafffe_weight_fn)

import argparse
def ParsingParameter():
    print("Start parsing parameters.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", default="1", help="[1, 0], whether show debug log")
    args = parser.parse_args()
    print("Start parsing parameters finish!")
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    return str2bool(args.show)

def main():
    # get_expect_result()
    # return
    show_debug_info = ParsingParameter()

    """ script entry point """
    print("=======================================")
    print("show_debug_info =", show_debug_info)
    print("=======================================")

    filename = "fake_model"
    cafffe_deploy_fn = filename + ".prototxt"
    cafffe_weight_fn = filename + ".caffemodel"
    print("fake prototxt =", cafffe_deploy_fn);
    print("fake weights =", cafffe_weight_fn);

    model_prototxt = create_prototxt()

    # Write prototxt to file.
    with open(cafffe_deploy_fn, 'w') as fproto:
       fproto.write("{0}".format(model_prototxt.to_proto()))
    fproto.close()
    print("create fake_prototxt finish!")

    create_fake_weights_by_prototxt(cafffe_deploy_fn, cafffe_weight_fn)
    print("create_fake_weights_by_prototxt finish!")

if __name__ == '__main__':
    main()
