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

    data = relay.var('data', shape=(input_shape), dtype=dtype)
    weight1 = relay.var('weight1', shape=(w1_shape), dtype=dtype)

    begin0 = relay.annotation.compiler_begin(data, "cv22")
    begin1 = relay.annotation.compiler_begin(weight1, "cv22")
    node0  = relay.nn.conv2d(begin0,
                             begin1,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             kernel_layout = 'OIHW')
    node1  = relay.clip(node0, 0, 6) # relu6
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
        onnx_model = to_onnx(module, {}, name, path=name)

if __name__ == '__main__':
    test_cv22()
