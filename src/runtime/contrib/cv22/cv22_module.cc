/* * Licensed to the Apache Software Foundation (ASF) under one
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
 * \file runtime/contrib/cv22/cv22_module.cc
 * \brief CV22Module is the runtime module for cv22 backend.
 */

#include <stdlib.h>
#include <tvm/node/serialization.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../../file_util.h"
#include "cv22_module.h"

namespace tvm {
namespace runtime {

/*! \brief A module for CV22 runtime. */
class CV22Module : public runtime::ModuleNode {
 public:
  explicit CV22Module(
      const std::unordered_map<std::string, std::string>& serialized_subgraphs)
      : serialized_subgraphs_(serialized_subgraphs) {
      LOG(INFO) << "CV22Module Constructor";
  }

  ~CV22Module() {
      LOG(INFO) << "CV22Module Destructor";
  }

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    // Returning nullptr tells TVM that the function is not in this module, so
    // it can look for the correct one.
    auto it_subgraph = serialized_subgraphs_.find(name);
    LOG(INFO) << "CV22Module GetFunction init";
    if (it_subgraph == serialized_subgraphs_.end()) {
      return PackedFunc(nullptr);
    }
    // Generate an external packed function
    return PackedFunc([this, name](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      LOG(INFO) << "CV22Module GetFunction PackedFunc";
    });
  }

  const char* type_key() const { return "cv22"; }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    CHECK_EQ(fmt, type_key()) << "Can only save to format=" << type_key();
    SaveBinaryToFile(file_name, SerializeModuleToString());
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(SerializeModuleToString());
  }

 static Module LoadFromFile(const std::string& path) {
    std::ifstream filep(path);
    filep.seekg(0, std::ios::end);
    size_t size = filep.tellg();
    std::string serialized_module(size, ' ');
    filep.seekg(0);
    filep.read(&serialized_module[0], size);
    return CreateModuleFromString(serialized_module);
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string serialized_module;
    stream->Read(&serialized_module);
    return CreateModuleFromString(serialized_module);
  }

 private:
  /*! \brief Relay program serialized using SaveJSON */
  std::unordered_map<std::string, std::string> serialized_subgraphs_;

  /*! \brief Serialize this module to a string. To be used during codegen. */
  std::string SerializeModuleToString() {
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    writer.BeginObject();
    writer.WriteObjectKeyValue("subgraphs", serialized_subgraphs_);
    writer.EndObject();
    return os.str();
  }

  /*! \brief Load serialized module from string created by SerializeModuleToString. */
  static Module CreateModuleFromString(const std::string& str) {
    std::unordered_map<std::string, std::string> serialized_subgraphs;
    std::istringstream is(str);
    dmlc::JSONReader reader(&is);
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("subgraphs", &serialized_subgraphs);
    helper.ReadAllFields(&reader);
    auto n = make_object<CV22Module>(serialized_subgraphs);
    return Module(n);
  }

 };

Module CV22ModuleCreate(
    const std::unordered_map<std::string, std::string>& serialized_subgraphs) {
  LOG(INFO) << "In CV22ModuleCreate";
  auto n = make_object<CV22Module>(serialized_subgraphs);
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_cv22")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = CV22Module::LoadFromFile(args[0]);
});

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_cv22")
.set_body_typed(CV22Module::LoadFromBinary);

}  // namespace runtime
}  // namespace tvm
