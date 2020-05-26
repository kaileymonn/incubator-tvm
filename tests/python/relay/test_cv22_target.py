# licensed to the apache software foundation (asf) under one
# or more contributor license agreements.  see the notice file
# distributed with this work for additional information
# regarding copyright ownership.  the asf licenses this file
# to you under the apache license, version 2.0 (the
# "license"); you may not use this file except in compliance
# with the license.  you may obtain a copy of the license at
#
#   http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing,
# software distributed under the license is distributed on an
# "as is" basis, without warranties or conditions of any
# kind, either express or implied.  see the license for the
# specific language governing permissions and limitations
# under the license.
"""Unit tests for graph partitioning."""
import os
import numpy as np
from shutil import rmtree

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
import onnx

# CVFlow imports
from cvflow_compiler import partitions_to_modules,cvflow_compilation

TEST_CASES = ['full_graph_offload', 'hetero']

def test_cv22(test_case, debug=0):
    #============= Constructing a simple graph ==============
    dtype = "float32"
    i_shape = (1, 1, 2, 4) # NCHW
    o_shape = (1, 1, 2, 4) # NCHW

    data0 = relay.var('data0', shape=(i_shape), dtype=dtype)
    data1 = relay.var('data1', shape=(i_shape), dtype=dtype)
    data2 = relay.var('data2', shape=(i_shape), dtype=dtype)

    if test_case == 'full_graph_offload':
        begin0 = relay.annotation.compiler_begin(data0, "cv22")
        begin1 = relay.annotation.compiler_begin(data1, "cv22")
        begin2 = relay.annotation.compiler_begin(data2, "cv22")
        node0  = relay.multiply(begin0, begin1)
        node1  = relay.add(node0, begin2)
        out    = relay.annotation.compiler_end(node1, "cv22")

    elif test_case == 'hetero':
        begin0 = relay.annotation.compiler_begin(data0, "cv22")
        begin1 = relay.annotation.compiler_begin(data1, "cv22")
        node0  = relay.multiply(begin0, begin1)
        end0   = relay.annotation.compiler_end(node0, "cv22")
        out    = relay.add(end0, data2) # arm

    else:
        raise ValueError('Unknown test case type %s, supported %s' % (test_case, TEST_CASES))

    f2  = relay.Function([data0, data1, data2], out)
    mod = tvm.IRModule.from_expr(f2)

    print('---------- Annotated graph ----------')
    print(mod.astext(show_meta_data=False))

    # graph partitioning
    mod_partition = transform.PartitionGraph()(mod)
    print('---------- Partitioned graph ----------')
    print(mod_partition.astext(show_meta_data=False))

    # remove existing dir and create new
    output_folder = '/tmp/test_amba/'
    rmtree(output_folder, ignore_errors=True)
    output_folder += 'prepare/'
    os.makedirs(output_folder)

    module_list = partitions_to_modules(mod_partition)
    for name, module in module_list.items():
        # convert to onnx
        onnx_path = name+".onnx"
        onnx_model = to_onnx(module, graph_name=name)
        onnx.save(onnx_model, onnx_path)
        print('Saved onnx file %s to disk\n' % onnx_path)

        # invoke cvflow compilation
        save_path = cvflow_compilation(model_path=onnx_path, \
                                       output_name=name, \
                                       output_folder=output_folder)
        print('Saved compiled model to: %s\n' % save_path)

    # tvm compilation
    with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
        json, lib, params = relay.build(mod_partition, target='llvm')

    # Serialize
    with open('compiled.json', 'w') as f_graph_json:
        f_graph_json.write(json)
    with open('compiled.params', 'wb') as f_params:
        f_params.write(relay.save_param_dict(params))
    #lib.save('compiled.cv22')

    # test runtime
    i0_data = np.random.randint(0, 100, i_shape).astype(dtype)
    i1_data = np.random.randint(0, 100, i_shape).astype(dtype)
    i2_data = np.random.randint(0, 100, i_shape).astype(dtype)
    map_inputs = {"data0": i0_data, "data1": i1_data, "data2":i2_data}

    rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx=tvm.cpu())
    for name, data in map_inputs.items():
        rt_mod.set_input(name, data)
    rt_mod.run()
    tgt_out = tvm.nd.empty(o_shape, ctx=tvm.cpu())
    tgt_out = rt_mod.get_output(0, tgt_out)
    tgt_out = tgt_out.asnumpy()

    ref_out = (i0_data * i1_data) + i2_data

    if debug > 0:
        print("i0_data:")
        print(i0_data)
        print("i1_data:")
        print(i1_data)
        print("i2_data:")
        print(i2_data)
        print("out_data:")
        print(tgt_out)
        print("ref_data:")
        print(ref_out)
        print()

    if not np.allclose(tgt_out, ref_out):
        print('Target and reference does not match')
        return 0

    else:
        return 1

if __name__ == '__main__':
    num_passed  = 0
    for t in TEST_CASES:
        num_passed += test_cv22(test_case=t)

    print('Total number of test cases: %d' % len(TEST_CASES))
    print('Number of test cases passed: %d' % num_passed)
