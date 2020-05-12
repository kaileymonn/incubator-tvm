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
"""cvflow compilation code"""

import sys
import subprocess
import onnx
import tvm

cnn_utils_path = subprocess.check_output(['tv2', '-basepath', 'CnnUtils'])
cnn_utils_path = cnn_utils_path.decode().rstrip('\n')
tv2_p = cnn_utils_path + '/packages/'
if tv2_p not in sys.path:
    sys.path.append(tv2_p)
else:
    raise Exception('%s not found' % tv2_p)

import cvflow_backend
from cvflow_backend.ir_utils import ir_helper

def cvflow_compilation(model_path, graphdesc_path, output_name, output_folder):

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
