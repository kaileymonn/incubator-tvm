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

import os
import sys
import subprocess
import onnx
import tvm
import copy
import numpy as np
import json

cnn_utils_path = subprocess.check_output(['tv2', '-basepath', 'CnnUtils'])
cnn_utils_path = cnn_utils_path.decode().rstrip('\n')

tv2_p = cnn_utils_path + '/packages/'
if tv2_p not in sys.path:
    sys.path.append(tv2_p)
else:
    raise Exception('%s not found' % tv2_p)

tv2_p = cnn_utils_path + '/onnx/common'
if tv2_p not in sys.path:
    sys.path.append(tv2_p)
else:
    raise Exception('%s not found' % tv2_p)

import cvflow_backend
from cvflow_backend.ir_utils import ir_helper
import onnx_graph_utils as OnnxGraphUtils

def create_rand_dra(primary_inputs, output_folder):
    '''
    Create DRA files with random float32 values

    Inputs:
    primary_inputs: {input tensor name: shape}
    output_folder:  folder to store dra files in

    Outputs:
    dra_dict: {input tensor name: (shape, dra bin filename)}
    '''

    dra_dict = {}
    for i in primary_inputs:
        ishape = primary_inputs[i]

        data = np.random.randint(0, 100, ishape).astype(np.float32)

        fname = output_folder + i + '_dra.bin'
        data.tofile(fname)

        dra_fname = output_folder + i + '_dra.txt'
        with open(dra_fname, 'w') as dra_file:
            dra_file.write(fname)

        dra_dict[i] = (ishape, dra_fname)

    return dra_dict

def create_splits_json(dra_dict, primary_outputs, vp_name, output_folder):
    '''
    Create splits json for cvflow backend prepare

    Inputs:
    dra_dict: {input tensor name: (shape, dra bin filename)}
    vp_name:  vp subgraph name
    primary_outputs: {output tensor name: shape}

    Outputs:
    splits_json_dict: See spec
    '''

    begin_dict = {}
    for i in dra_dict:
        inp_dict = {}
        inp_dict['shape'] = dra_dict[i][0]
        inp_dict['dtype'] = 'float32'
        inp_dict['file']  = dra_dict[i][1]
        inp_dict['extn']  = 'bin'

        begin_dict[i] = inp_dict.copy()

    end_dict = {}
    for o in primary_outputs:
        end_dict[o] = {'dtype': 'float32'}

    attr_dict = {}
    attr_dict['cnngen_flags'] = '-c coeff-force-fx16,act-force-fx16'
    attr_dict['vas_flags']    = '-auto -v'

    vp_dict = {}
    vp_dict['type']  = 'ORCVP'
    vp_dict['begin'] = begin_dict
    vp_dict['end']   = end_dict
    vp_dict['attr']  = attr_dict

    splits_json = {}
    splits_json[vp_name] = copy.deepcopy(vp_dict)

    splits_json_fname = output_folder + vp_name + '_splits.json'
    with open(splits_json_fname, 'w') as json_file:
        json.dump(splits_json, json_file, indent=1)

    return splits_json_fname

def set_env_variable(key, value):
    os.environ[key] = value

def cvflow_compilation(model_path, output_name, output_folder):

    if not output_folder.endswith('/'):
         output_folder += '/'

    # get graph info
    model, load_info = OnnxGraphUtils.load_model(model_path)
    model_summary, gs_recs = OnnxGraphUtils.determine_graph_info(model)

    primary_inputs  = model_summary['PRIMARY_INPUTS']
    primary_outputs = model_summary['PRIMARY_OUTPUTS']

    # create random images for dra
    dra_dict = create_rand_dra(primary_inputs, output_folder)

    # create splits json file
    graphdesc_path = create_splits_json(dra_dict, primary_outputs, output_name, output_folder)    

    # set outputs list as env variable
    # this will be used by codegen
    pr_outputs = ','.join(list(primary_outputs.keys()))
    set_env_variable('CV22_OUTPUTS_LIST', pr_outputs)

    graphdesc_bytes = None
    with open(graphdesc_path, mode='rb') as f:
        graphdesc_bytes = f.read()

    modelproto = onnx.load(model_path)
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
