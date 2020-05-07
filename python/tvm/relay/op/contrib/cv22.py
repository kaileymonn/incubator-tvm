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
# pylint: disable=invalid-name, unused-argument
"""CV22 library supported operators.
"""
from ... import op as reg

target = "target.cv22"

def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by CV22.
    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.
    Returns
    -------
    f : callable
        A function that returns if the operator is supported by CV22.
    """
    @reg.register(op_name, "target.cv22")
    def _func_wrapper(attrs, args):
        return supported

    return _func_wrapper

_register_external_op_helper("log")
_register_external_op_helper("sqrt")
_register_external_op_helper("rsqrt")
_register_external_op_helper("exp")
_register_external_op_helper("sigmoid")
_register_external_op_helper("add")
_register_external_op_helper("subtract")
_register_external_op_helper("multiply")
_register_external_op_helper("divide")
#_register_external_op_helper("mod") # TBD, Can be supported by using two or more SGL nodes
_register_external_op_helper("tanh")
_register_external_op_helper("concatenate")
_register_external_op_helper("expand_dims")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("nn.log_softmax")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.dropout")
_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.biad_add")

register_external_op_helper("nn.conv2d")
register_external_op_helper("nn.conv2d_transpose")
register_external_op_helper("nn.dense")
register_external_op_helper("nn.max_pool2d")
register_external_op_helper("nn.avg_pool2d")
register_external_op_helper("nn.global_max_pool2d")
register_external_op_helper("nn.global_avg_pool2d")
register_external_op_helper("nn.upsampling")
register_external_op_helper("nn.batch_flatten")
register_external_op_helper("nn.pad")
register_external_op_helper("nn.lrn")
register_external_op_helper("nn.l2_normalize")
register_external_op_helper("nn.contrib_conv2d_winograd_without_weight_transform")

register_external_op_helper("nn.leaky_relu")
register_external_op_helper("nn.prelu")
register_external_op_helper("reshape")
register_external_op_helper("reshape_like")
register_external_op_helper("copy")
register_external_op_helper("transpose")
register_external_op_helper("squeeze")
register_external_op_helper("floor")
register_external_op_helper("ceil")
register_external_op_helper("sign")
register_external_op_helper("trunc")
register_external_op_helper("clip")
register_external_op_helper("round")
register_external_op_helper("abs")
register_external_op_helper("negative")
register_external_op_helper("take")
register_external_op_helper("zeros")
register_external_op_helper("zeros_like")
register_external_op_helper("ones")
register_external_op_helper("ones_like")
register_external_op_helper("gather_nd")
register_external_op_helper("full")
register_external_op_helper("full_like")
register_external_op_helper("cast")
register_external_op_helper("reinterpret")
register_external_op_helper("split")
register_external_op_helper("arange")
register_external_op_helper("stack")
register_external_op_helper("repeat")
register_external_op_helper("tile")
register_external_op_helper("reverse")

register_external_op_helper("maximum")
register_external_op_helper("minimum")
register_external_op_helper("power")
register_external_op_helper("argmax")
register_external_op_helper("argmin")
register_external_op_helper("sum")
register_external_op_helper("max")
register_external_op_helper("min")
register_external_op_helper("mean")
register_external_op_helper("prod")
register_external_op_helper("strided_slice")
register_external_op_helper("broadcast_to")

