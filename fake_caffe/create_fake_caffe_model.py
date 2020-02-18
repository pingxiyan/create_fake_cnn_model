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

def load_configuration(fname):
    """ Load YOLO configuration file. """
    with open(fname, 'r') as fconf:
        lines = [l.strip() for l in fconf]

    config = []
    element = {}
    section_name = None
    for line in lines:
        if not line or line[0] == '#':  # empty or comment
            continue
        if line[0] == '[':  # new section
            if section_name:
                config.append((section_name, element))
                element = {}
            section_name = line[1:].strip(']')
        else:
            key, value = line.split('=')
            element[key] = value
    config.append((section_name, element))

    return config


## Layer parsing ##
###################

def get_layer_name(irlayer, count):
    return "{0}_{1}".format(irlayer.name, count)

def data_layer(name, n,c,h,w):
    """ add a data layer """
    params = {"batch":n, "channels":c, "width":w, "height":h}
    fields = dict(shape={"dim": [1, int(params["channels"]), int(params["height"]), int(params["width"])]})
    return cl.Input(name=name, **fields)

def activation_layer(previous, count, irlayer, lname, mode="ReLU"):
    """ create a non-linear activation layer """
    if mode=="ReLU":
        return cl.ReLU(previous, name=lname, in_place=True)
    elif mode=="leaky":
        return cl.ReLU(previous, name=lname, in_place=True, relu_param=dict(negative_slope=0.1))
    elif mode=="PReLU":
        # print(irlayer.weight_arr[0])
        # print(type(irlayer.weight_arr))
        # return cl.PReLU(previous, name=lname, in_place=True)
        alpha_val = irlayer.weight_arr[0]
        return cl.ReLU(previous, name=lname, in_place=True, relu_param=dict(negative_slope=alpha_val))
    elif mode=="elu":
        default_alpha = 1.0
        if irlayer.activation_alpha:
            default_alpha = irlayer.activation_alpha
        return cl.ELU(previous, name=lname, in_place=True, elu_param=dict(alpha=irlayer.activation_alpha))
    elif mode=="sigmoid":
        return cl.Sigmoid(previous, name=lname, in_place=True)
    else:
        raise ValueError("Activation mode not implemented: {0}".format(mode))

def add_data_option(fields, opt_name, arr):
    if len(arr) == 2:
        fields[opt_name+"_h"] = arr[0]
        fields[opt_name+"_w"] = arr[1]
    else:
        fields[opt_name] = arr[0]

def convolutional_layer(previous, name, irlayer, train=False, has_bn=False):
    """ create a convolutional layer given the parameters and previous layer """
    fields = dict(num_output=irlayer.conv_output)
    if len(irlayer.conv_dilations) == 2:
        if irlayer.conv_dilations[0] != irlayer.conv_dilations[1]:
            print("Current convolution layer don't support dilation have 2 parameter")
    add_data_option(fields, "dilation", [irlayer.conv_dilations[0]])

    if irlayer.conv_group != None:
        fields["group"] = int(irlayer.conv_group)
    add_data_option(fields, "kernel", irlayer.conv_kernel)

    if irlayer.conv_pads_begin[0] != irlayer.conv_pads_end[0]:
        print("Current don't support conv layer: pads_begin != pads_end")
    else:
        add_data_option(fields, "pad", irlayer.conv_pads_begin)

    add_data_option(fields, "stride", irlayer.conv_strides)
    return cl.Convolution(previous, name=name, **fields)

def eltwise_layer(inputlayer1, inputlayer2, name, irlayer):
    """ eltwise layer, default op SUM """
    eltwise_op = 1  # http://caffe.berkeleyvision.org/tutorial/layers/eltwise.html, default is sum
    eltwise_op_name = "sum"
    if irlayer.eltwise_operation == "sum":
        eltwise_op = 1
    elif irlayer.eltwise_operation == "max":
        eltwise_op = 2
    elif irlayer.eltwise_operation == "prod":
        eltwise_op = 0
    elif irlayer.eltwise_operation == "mul":
        eltwise_op = 0
    else:
        print("I don't know this eltwise_op:", irlayer.eltwise_operation)
        print("Using default eltwise:", eltwise_op, ", name =", eltwise_op_name)

    # l0_name = inputlayer1.fn.params["name"]
    # l1_name = inputlayer2.fn.params["name"]
    # print("l0_name = ", l0_name, " l1_name = ", l1_name)
    return cl.Eltwise(inputlayer1, inputlayer2, name=name, eltwise_param=dict(operation=eltwise_op))

def local_layer(previous, name, params, train=False):
    """ create a locally connected layer given the parameters and previous
    layer """
    if 'LocalConvolution' not in caffe.layer_type_list():
        raise ValueError("Layer not available: LocalConvolution")

    fields = dict(num_output=int(params["filters"]),
                  kernel_size=int(params["size"]))
    if "stride" in params.keys():
        fields["stride"] = int(params["stride"])

    if int(params.get("pad", 0)) == 1:    # use 'same' strategy for convolutions
        fields["pad"] = fields["kernel_size"]//2
    if train:
        fields.update(weight_filler=dict(type="gaussian", std=0.01),
                      bias_filler=dict(type="constant", value=0))
    return cl.LocalConvolution(previous, name=name, **fields)

def batchnorm_layer(previous, name, train=False):
    """ create a batch normalization layer given the parameters and previous
    layer """
    if not train:
        return cl.BatchNorm(previous, name=name, use_global_stats=True)
    return cl.BatchNorm(previous, name=name, include=dict(phase=caffe.TRAIN),
                        # suppress SGD on bn params for old Caffe versions
                        param=[dict(lr_mult=0, decay_mult=0)]*3,
                        use_global_stats=False)

def max_pooling_layer(previous, name, kernel_sz, stride_sz, pad_sz):
    """ create a max pooling layer """
    if len(kernel_sz) == 1:
        return cl.Pooling(
            previous, name=name, pool=cp.Pooling.MAX,
            kernel_size=kernel_sz[0], stride=stride_sz[0], pad = pad_sz[0])
    elif len(kernel_sz) == 2:
        return cl.Pooling(
            previous, name=name, pool=cp.Pooling.MAX,
            stride_h=stride_sz[0], stride_w=stride_sz[1], kernel_h=kernel_sz[0], kernel_w=kernel_sz[1], pad_h=pad_sz[0], pad_w = pad_sz[1])
    else:
        print("pooling kernel size dim only support 1 and 2")

def full_connect_layer(previous, name, irlayer):
    """ create a fc layer """
    fields = dict(num_output=irlayer.fc_out_size)
    return cl.InnerProduct(previous, name=name, inner_product_param=fields)

def soft_max_layer(previous, name):
    """ create a fc layer """
    return cl.Softmax(previous, name=name)

def global_pooling_layer(previous, name, mode="avg"):
    """ create a Global Pooling Layer """
    pool = cp.Pooling.AVE if mode == "avg" else cp.Pooling.MAX
    return cl.Pooling(previous, name=name, pool=pool, global_pooling=True)

def dense_layer(previous, name, params, train=False):
    """ create a densse layer """
    fields = dict(num_output=int(params["output"]))
    if train:
        fields.update(weight_filler=dict(type="gaussian", std=0.01),
                      bias_filler=dict(type="constant", value=0))
    return cl.InnerProduct(previous, name=name, inner_product_param=fields)


### layer aggregation ###
#########################

def add_convolutional_layer(layers, count, irlayer, inputlayers):
    layer_name = get_layer_name(irlayer, count)
    # print("add_convolutional_layer =",layer_name)
    has_batch_norm = False
    # has_batch_norm = (params.get("batch_normalize", '0') == '1')
    if len(inputlayers) != 1:
        print("Current convoluation layer only support input one layer, input:", len(inputlayers))

    layers.append(convolutional_layer(inputlayers[0], layer_name, irlayer, False, has_batch_norm))

# Scale_shift convert to conv1*1
def add_scale_shift_layer(layers, count, irlayer, inputlayers):
    layer_name = get_layer_name(irlayer, count)
    layers.append(cl.Scale(inputlayers[0], name=layer_name, bias_term=True, in_place=True))

def add_dense_layer(layers, count, params, train=False):
    """ add layers related to a connected block in YOLO to the layer list """
    layers.append(dense_layer(layers[-1], "fc{0}".format(count), params, train))
    if params["activation"] != "linear":
        layers.append(activation_layer(layers[-1], count, params["activation"]))

def add_local_layer(layers, count, params, train=False):
    """ add layers related to a connected block in YOLO to the layer list """
    layers.append(local_layer(layers[-1], "local{0}".format(count), params, train))
    if params["activation"] != "linear":
        layers.append(activation_layer(layers[-1], count, params["activation"]))

def add_pooling_layer(layers, lname, irlayer, inputlayers):
    if len(inputlayers) != 1:
        print("current pooling only support 1 input")

    if irlayer.pool_methed == "max":
        layers.append(max_pooling_layer(inputlayers[0], lname,
            irlayer.pool_kernel, irlayer.pool_stride, irlayer.pool_pads_begin))

def add_eltwise_layer(layers, count, irlayer, bottomlayer):
    if len(bottomlayer) != 2:
        print("Error: Eltwise need input 2 layers, real input:", len(bottomlayer))
        exit(0)
    layer_name = get_layer_name(irlayer, count)
    # print("l0_name = ", l0_name, " l1_name = ", l1_name)
    layers.append(eltwise_layer(bottomlayer[0], bottomlayer[1], layer_name, irlayer))

def add_softmax_layer(layers, count, irlayer, bottomlayer):
    layer_name = get_layer_name(irlayer, count)
    #print("add_softmax_layer layer_name =", len(bottomlayer))
    layers.append(soft_max_layer(bottomlayer[0], layer_name))

def add_concat_layer(layers, count, irlayer, bottomlayer):
    # print("add_concat_layer=Can't finish")
    layer_name = get_layer_name(irlayer, count)
    # print(type(bottomlayer))
    # print(len(bottomlayer))
    concat_layer=cl.Concat(*bottomlayer, name=layer_name)
    layers.append(concat_layer)

def add_flatten_layer(layers, count, irlayer, bottomlayer):
    # print("add_flatten_layer = Can't finish")
    layer_name = get_layer_name(irlayer, count)
    # print(len(bottomlayer))
    flatten_layer=cl.Flatten(bottomlayer[0], name=layer_name)
    layers.append(flatten_layer)

def add_activation_layer(layers, count, irlayer, bottomlayer, type):
    if len(bottomlayer) != 1:
        print("relu only input one layer")
    
    layer_name = get_layer_name(irlayer, count)
    if type=="ReLU":
        layers.append(activation_layer(bottomlayer[0], count, irlayer, layer_name, mode="ReLU"))
    elif type=="leaky":
        layers.append(activation_layer(bottomlayer[0], count, irlayer, layer_name, mode="leaky"))
    elif type == "Activation":
        layers.append(activation_layer(bottomlayer[0], count, irlayer, layer_name, mode=irlayer.activation_type))
    elif type == "PReLU":
        layers.append(activation_layer(bottomlayer[0], count, irlayer, layer_name, mode="PReLU"))
    else:
        print("I don't know this activation type:", type)
def add_reshape_layer(layers, count, irlayer, bottomlayer):
    layer_name = get_layer_name(irlayer, count)
    n,c,h,w=irlayer.nchw["n"],irlayer.nchw["c"],irlayer.nchw["h"],irlayer.nchw["w"]
    fields=dict()
    # if irlayer.axis == 2:
    #     fields = dict(reshape_param={"shape": {"dim": [2,0,-1,3]}})    
    # elif irlayer.axis == 3:
    #     fields = dict(reshape_param={"shape": {"dim": [2,3,0,-1]}})
    fields = dict(reshape_param={"shape": {"dim": [int(n), int(c), int(h), int(w)]}})
    # fields["axis"]=int(irlayer.axis)
    # fields["num_axes"]=int(irlayer.tiles)
    print(fields)
    reshape_layer=cl.Reshape(bottomlayer[0], name=layer_name, **fields)
    layers.append(reshape_layer)

def add_tile_layer(layers, count, irlayer, bottomlayer):
    layer_name = get_layer_name(irlayer, count)
    n,c,h,w=irlayer.nchw["n"],irlayer.nchw["c"],irlayer.nchw["h"],irlayer.nchw["w"]
    irlayer.bias_arr=[0]*c*h*w
    irlayer.weight_arr=[1]*c*h*w

    fields["axis"]=int(irlayer.axis)
    fields["num_axes"]=int(irlayer.tiles)
    print(fields)
    reshape_layer=cl.Reshape(bottomlayer[0], name=layer_name, **fields)
    layers.append(reshape_layer)

def find_inputlayer(irmodel, idx, caffelayers):
    inputlayers = []
    if idx < 0:
        print("Error: layer id shouldn't < 0")
        exit(0)
        return inputlayers

    for m in irmodel.edge:
        if m.to_layer == idx:
            #inputlayers.append(caffelayers[m.from_layer])
            inputlayers.append(caffelayers[m.from_layer])
            # print("from-layer={} to-layer={}, name={}".format(m.from_layer, m.to_layer, irmodel.layer.values()[idx].name))
    return inputlayers

def create_prototxt():
    """ Create some fake layers: http://ethereon.github.io/netscope/#/editor """
    layers = []

    # Layer1: "Input":
    layers.append(data_layer("data", 1, 3, 416, 416))

    # Layer2: "Pooling":
    layers.append(cl.Pooling(layers[0], name="Pooling1", pool=cp.Pooling.MAX, stride_h=2, stride_w=2,
            kernel_h=3, kernel_w=3,
            pad_h=1, pad_w =1))

    # layer3: "" // 1.3.43.43
    layers.append(cl.Pooling(layers[1], name="Pooling2", pool=cp.Pooling.MAX, stride_h=5, stride_w=5,
            kernel_h=5, kernel_w=5,
            pad_h=2, pad_w =2))

    # for l in irmodel.layer.values():
    #     section = l.type
    #     # print("Converting layer =", l.type)
    #     inputlayers = find_inputlayer(irmodel, count, layers)
    #     cur_layer_name = get_layer_name(l, count)
    #     # print("cur_layer_name =", cur_layer_name, " count =", count)

    #     if section == "Input":
    #         layers.append(data_layer(l.name, l, train))
    #     elif section == "ScaleShift":
    #         add_scale_shift_layer(layers, count, l, inputlayers)
    #     elif section == "Convolution":
    #         add_convolutional_layer(layers, count, l, inputlayers)
    #     elif section == "ReLU":
    #         add_activation_layer(layers, count, l, inputlayers, section)
    #     elif section == "Activation":
    #         add_activation_layer(layers, count, l, inputlayers, section)
    #     elif section == "Eltwise":
    #         add_eltwise_layer(layers, count, l, inputlayers)
    #     elif section == "Pooling":
    #         add_pooling_layer(layers, count, l, inputlayers)
    #     elif section == "FullyConnected":
    #         if len(inputlayers) != 1:
    #             print("Warning: FullyConnected only input one layer")
    #         layers.append(full_connect_layer(inputlayers[0], cur_layer_name, l))
    #     elif section == "SoftMax":
    #         add_softmax_layer(layers, count, l, inputlayers)
    #     elif section == "Concat":
    #         add_concat_layer(layers, count, l, inputlayers)
    #     elif section == "Flatten":
    #         add_flatten_layer(layers, count, l, inputlayers)
    #     elif section == "PReLU":
    #         add_activation_layer(layers, count, l, inputlayers, section)
    #     elif section == "Tile": # same to caffe.reshape
    #         add_reshape_layer(layers, count, l, inputlayers)
    #     else:
    #         print("WARNING: {0} layer not recognized".format(section))
    #     count += 1

    #     # if count==78 :
    #     #     break

    # # At last add softmax layer
    # if badd_softmax_layer:
    #     layers.append(soft_max_layer(layers[len(layers) - 1], "softmax"))

    model = caffe.NetSpec()
    for layer in layers:
        setattr(model, layer.fn.params["name"], layer)

    return model


def adjust_params(model, model_filename):
    """ Set layer parameters that depends on blob attributes.
    Blobs are available only in Net() objects, but NetSpec() or NetParameters()
    can't be used to create a Net(). So we write a first prototxt, we reload it,
    fix the missing parameters and save it again.
    """
    with open(model_filename, 'w') as fproto:
        fproto.write("{0}".format(model.to_proto()))

    net = caffe.Net(model_filename, caffe.TEST)
    for name, layer in model.tops.iteritems():
        if name.startswith("local"):
            width, height = net.blobs[name].data.shape[-2:]
            if width != height:
                raise ValueError(" Only square inputs supported for local layers.")
            layer.fn.params.update(
                local_region_number=width, local_region_ratio=1.0/width,
                local_region_step=1)

    return model

### Convert weight to caffemodel                               ###
##################################################################
def transpose_matrix(array, rows, cols):
    """ transpose flattened matrix """
    return array.reshape((rows, cols)).transpose().flatten()

def load_parameter(weights, layer_data, transpose=False):
    """  load Caffe parameters from IR bin weights """
    weights_buf = weights
    shape = layer_data.shape
    size = np.prod(shape)
    # print("shape = ", shape, ", weights.size = ", weights.size, " ,caffe need size = ", size)
    if size != len(weights_buf):
        print(" Layer too big: required {} weights, available {}".format(size, len(weights_buf)))
        exit(0)

    if transpose:
        layer_data[...] = np.reshape(
            transpose_matrix(weights_buf[:size], np.prod(shape[1:]), shape[0]), shape)
    else:
        layer_data[...] = np.reshape(weights_buf[:size], shape)

    return size

def get_current_ir_layer(ir_model, layername):
    real_layername = layername.rsplit("_", 1)[0]
    # print("real_layername = ", real_layername)
    irlayer = None
    for l in ir_model.layer.values():
        if l.name == real_layername:
            irlayer = l
            break
    return irlayer

def convert_weights(ir_model, caffe_model, cafffe_deploy_fn, cafffe_weight_fn):
    print("Start convert_weights")
    # ======================================================
    net = caffe.Net(cafffe_deploy_fn, caffe.TEST)
    print("Declare net success by caffe deploy")
    count = 0
    for name, layer in zip(net.top_names, net.layers):
        if name not in net.params.keys():  # layer without parameters
            continue
        if layer.type in ['BatchNorm']:
            continue   # handled within the convolutional layer

        irlayer = get_current_ir_layer(ir_model, name)
        if irlayer == None:
            print("Can't find layer ", name)
            continue

        print("  converting {}, {}".format(name, layer.type))
        if layer.type in ['Convolution', 'Scale', 'InnerProduct']:
            # bais
            if irlayer.bias_arr is not None:
                count += load_parameter(irlayer.bias_arr, net.params[name][1].data) # conv bias
            # weights
            count += load_parameter(irlayer.weight_arr, net.params[name][0].data) # conv weight
        # elif layer.type=='PReLU':
        #     # PReLU same to leakyReLU, but get alpha parameter by training. no bias.
        #     count += load_parameter(irlayer.weight_arr, net.params[name][0].data) # conv weight
        else:
            print("WARNING: unknown type {} for layer {}".format(layer.type, name))

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
    show_debug_info = ParsingParameter()

    """ script entry point """
    print("=======================================")
    print("show_debug_info =", show_debug_info)
    print("=======================================")

    filename = "fake_model"
    cafffe_deploy_fn = filename + ".prototxt"
    cafffe_weight_fn = filename + ".caffemodel"

    model_prototxt = create_prototxt()

    # Write prototxt to file.
    with open(cafffe_deploy_fn, 'w') as fproto:
       fproto.write("{0}".format(model_prototxt.to_proto()))
    fproto.close()

    print("create_prototxt finish!")
    #convert_weights(ir_model, model_prototxt, cafffe_deploy_fn, cafffe_weight_fn)
    print("convert_weights finish!")

if __name__ == '__main__':
    main()
