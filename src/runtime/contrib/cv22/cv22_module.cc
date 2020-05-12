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
#include <sys/stat.h>
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
      const std::unordered_map<std::string, subgraph_attr_t>& cv22_subgraphs)
      : cv22_subgraphs_(cv22_subgraphs) {
      LOG(INFO) << "CV22Module Constructor";
      for (auto it = cv22_subgraphs_.begin(); it != cv22_subgraphs_.end(); it++) {
          subgraph_attr_t& attr = it->second;
          subgr_fnames_.insert( std::pair<std::string,std::string>(it->first,attr.filename) );
      }
  }

  ~CV22Module() {
      LOG(INFO) << "CV22Module Destructor";
  }

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    // Returning nullptr tells TVM that the function is not in this module, so
    // it can look for the correct one.
    auto it_subgraph = cv22_subgraphs_.find(name);
    if (it_subgraph == cv22_subgraphs_.end()) {
      return PackedFunc(nullptr);
    }
    // Generate an external packed function
    return PackedFunc([this, name](tvm::TVMArgs args, tvm::TVMRetValue* rv) {

      LOG(INFO) << "CV22Module GetFunction PackedFunc";

      LOG(INFO) << "Filename: " << cv22_subgraphs_[name].filename;
      std::string cmd = "evaluate.py --metagraph " + cv22_subgraphs_[name].filename;

      std::string inp_dir = "/tmp/test_amba/";

      std::vector<std::string>& inputs = cv22_subgraphs_[name].inputs;
      for (size_t i = 0; i < inputs.size(); ++i) {
          LOG(INFO) << "Input " << i << ": " << inputs[i];

          DLTensor* arg = args[i];

          float* data = reinterpret_cast<float*>(arg->data);

          int buf_size = 1;
          LOG(INFO) << "Shape:";
          for (int j = 0; j < arg->ndim; ++j) {
               LOG(INFO) << arg->shape[j];
               buf_size *= arg->shape[j];
          }
          LOG(INFO) << "Size: " << buf_size;

          std::string in_fname = inp_dir + inputs[i] + ".bin";

          std::ofstream fout;
          fout.open(in_fname, std::ios::binary);

          if (fout.is_open()) {
              fout.write((char*) data, buf_size*sizeof(float));
          }
          else std::cout << "Unable to open file";

          fout.close();

          cmd += " --inputdata " + inputs[i] + "=" + in_fname;
      }

      std::vector<std::string>& outputs = cv22_subgraphs_[name].outputs;
      for (size_t o = 0; o < outputs.size(); ++o) {
          LOG(INFO) << "Output " << o << ": " << outputs[o];
      }
      cmd += " --output_folder /tmp/test_amba/eval/outputs --log_dir /tmp/test_amba/eval/logs";

      LOG(INFO) << "Cmd: " << cmd;

      system(cmd.c_str());

      if ((args.size() - inputs.size()) != 1) {
          // (TBD) error: only one output case supported
      }

      // (TBD) 
      std::string out_fname = "/tmp/test_amba/eval/outputs/node_3_iter_0.bin";
      std::ifstream fin;
      fin.open(out_fname, std::ios::binary);

      if (fin.is_open()) {
          int out_idx = inputs.size();
          DLTensor* arg = args[out_idx];
          char* data = reinterpret_cast<char*>(arg->data);

          int size = fin.tellg();
          fin.seekg(0, std::ios::beg);
          fin.read(data, size*4);
          fin.close();
       }

       else std::cout << "Unable to open file";

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
  std::unordered_map<std::string, subgraph_attr_t> cv22_subgraphs_;
  std::unordered_map<std::string, std::string> subgr_fnames_;

  /*! \brief Serialize this module to a string. To be used during codegen. */
  std::string SerializeModuleToString() {
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    writer.BeginObject();
    writer.WriteObjectKeyValue("subgraphs", subgr_fnames_);
    writer.EndObject();
    return os.str();
  }

  /*! \brief Load serialized module from string created by SerializeModuleToString. */
  static Module CreateModuleFromString(const std::string& str) {
    std::unordered_map<std::string, subgraph_attr_t> cv22_subgraphs;
    std::unordered_map<std::string, std::string> subgr_fnames;

    std::istringstream is(str);
    dmlc::JSONReader reader(&is);
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("subgraphs", &subgr_fnames);
    helper.ReadAllFields(&reader);

    for (auto it = subgr_fnames.begin(); it != subgr_fnames.end(); it++) {
        cv22_subgraphs[it->first].filename = it->second;
    }

    auto n = make_object<CV22Module>(cv22_subgraphs);
    return Module(n);
  }

 };

Module CV22ModuleCreate(
    const std::unordered_map<std::string, subgraph_attr_t>& cv22_subgraphs) {
  LOG(INFO) << "In CV22ModuleCreate";
  auto n = make_object<CV22Module>(cv22_subgraphs);
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
