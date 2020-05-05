# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Unit tests for graph partitioning."""
import os
import numpy as np

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_begin, compiler_end

# Imports for expr
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.function import Function

# Onnx imports
from tvm.relay.converter import to_onnx

# CV22 compilation import
import sys
import subprocess
import onnx

cnn_utils_path = subprocess.check_output(['tv2', '-basepath', 'CnnUtils'])
cnn_utils_path = cnn_utils_path.decode().rstrip('\n')
tv2_p = cnn_utils_path + '/packages/'
if tv2_p not in sys.path:
    sys.path.append(tv2_p)
else:
    raise Exception('%s not found' % tv2_p)

import cvflow_backend
from cvflow_backend.ir_utils import ir_helper

def cvflow_compilation(model_path, graphdesc_path, output_name, output_folder='amba_tvm_test'):

    modelproto = onnx.load(model_path)

    graphdesc_bytes = None
    with open(graphdesc_path, mode='rb') as f:
        graphdesc_bytes = f.read()

    ckpt_ambapb = cvflow_backend.prepare(model_bytes=modelproto.SerializeToString(), \
                                         graph_desc_bytes=graphdesc_bytes, \
                                         framework='onnx', \
                                         metagraph_type='checkpoint', \
                                         output_name=output_name, \
                                         output_folder=output_folder, \
                                         log_dir=output_folder+'/logs')

    save_path = ir_helper.save_model(ckpt_ambapb, \
                                     output_name, \
                                     output_folder)
    print('Saved compiled model to: %s' % save_path)

    return save_path

def partitions_to_modules(mod):
    module_dict = {}
    for func in mod.get_global_vars():
        name = func.name_hint
        if "cv22" in name:
            mod = tvm.IRModule.from_expr(mod[name])
            new_mod = tvm.ir.module.IRModule()
            new_mod["main"] = mod[name]
            module_dict[name] = new_mod

    return module_dict


def test_cv22():
    #============= Constructing a simple graph ==============
    dtype = "float32"
    input_shape = (1, 3, 224, 224) # NCHW
    w1_shape    = (32, 3, 3, 3)    # OIHW
    input_shape1 = (1, 32, 224, 224) # NCHW

    data = relay.var('data0', shape=(input_shape), dtype=dtype)
    data1 = relay.var('data1', shape=(input_shape1), dtype=dtype)
    weight1 = relay.var('weight1', shape=(w1_shape), dtype=dtype)

    begin0 = relay.annotation.compiler_begin(data, "cv22")
    begin1 = relay.annotation.compiler_begin(weight1, "cv22")
    begin2 = relay.annotation.compiler_begin(data1, "cv22")
    node0  = relay.nn.conv2d(begin0,
                             begin1,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             kernel_layout = 'OIHW')
    node1  = relay.add(node0, begin2) # relu6g
    # whole graph offload
    out2 = relay.annotation.compiler_end(node1, "cv22")
    f2   = relay.Function([data, weight1], out2)
    mod2 = tvm.IRModule.from_expr(f2)
    print('---------- Annotated graph ----------')
    print(mod2.astext(show_meta_data=False))

    #============= Partitioning the graph ==============
    mod2_partition = transform.PartitionGraph()(mod2)
    print('---------- Partitioned graph ----------')
    print(mod2_partition.astext(show_meta_data=False))

    module_list = partitions_to_modules(mod2_partition)
    for name, module in module_list.items():
        onnx_path = name+".onnx"
        onnx_model = to_onnx(module, {}, name, path=onnx_path)
        print('Saved onnx file %s to disk\n' % onnx_path)

        # invoke cvflow compilation
        cvflow_compilation(model_path=onnx_path, \
                           graphdesc_path='splits_new.json', \
                           output_name=name)

if __name__ == '__main__':
    test_cv22()
