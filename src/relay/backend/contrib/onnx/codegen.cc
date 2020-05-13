/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/backend/contrib/onnx/codegen.cc
 * \brief Implementation of ONNX codegen APIs.
 */

#include <sstream>
#include<stdio.h>

#include "../../utils.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

class ONNXModuleCodegen {
 public:
	ONNXModuleCodegen(){}
  runtime::Module CreateONNXModule(const ObjectRef& ref){

	auto mod = Downcast<IRModule>(ref);
	String codes = (*to_onnx_)(mod);
	/* Here, use String instead of std::string, because some byte info
	 * will lost while calling PackedFunc object.
	 */
	const auto* pf = runtime::Registry::Get("runtime.ONNXModuleCreate");
    CHECK(pf != nullptr) << "Cannot find onnx module to create the external runtime module";
    return (*pf)(codes, "onnx");
  }

 private:
  /*!
   * \brief The python function to convert relay module to onnx module.
   * \return byte array -> String
   */
  const PackedFunc* to_onnx_ = runtime::Registry::Get("tvm.relay.converter.to_onnx");
};



runtime::Module ONNXCompiler(const ObjectRef& ref) {
  ONNXModuleCodegen onnx;
  return onnx.CreateONNXModule(ref);
}
TVM_REGISTER_GLOBAL("relay.ext.onnx").set_body_typed(ONNXCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

