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
 * \file source_module.cc
 * \brief Source code module, only for viewing
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <fstream>
#include "codegen_source_base.h"
#include "../../runtime/file_util.h"
#include "../../runtime/meta_data.h"

namespace tvm {
namespace codegen {

using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;

using runtime::GetFileFormat;
using runtime::GetMetaFilePath;
using runtime::FunctionInfo;
using runtime::SaveBinaryToFile;

// Simulator function
class SourceModuleNode : public runtime::ModuleNode {
 public:
  SourceModuleNode(std::string code,
                   std::string fmt)
      : code_(code), fmt_(fmt) {}
  const char* type_key() const {
    return "source";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return PackedFunc();
  }

  std::string GetSource(const std::string& format) final {
    return code_;
  }

 protected:
  std::string code_;
  std::string fmt_;
};

runtime::Module SourceModuleCreate(std::string code, std::string fmt) {
  auto n = make_object<SourceModuleNode>(code, fmt);
  return runtime::Module(n);
}

// Simulator function
class CSourceModuleNode : public runtime::ModuleNode {
 public:
  CSourceModuleNode(std::string code,
                   std::string fmt)
      : code_(code), fmt_(fmt) {}
  const char* type_key() const {
    return "c";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "C Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return PackedFunc();
  }

  std::string GetSource(const std::string& format) final {
    return code_;
  }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "cc") {
      CHECK_NE(code_.length(), 0);
      SaveBinaryToFile(file_name, code_);
    } else {
      CHECK_EQ(fmt, fmt_)
          << "Can only save to format=" << fmt_;
    }
  }

 protected:
  std::string code_;
  std::string fmt_;
};

runtime::Module CSourceModuleCreate(std::string code, std::string fmt) {
  auto n = make_object<CSourceModuleNode>(code, fmt);
  return runtime::Module(n);
}

// supports limited save without cross compile
class DeviceSourceModuleNode final : public runtime::ModuleNode {
 public:
  DeviceSourceModuleNode(std::string data,
                         std::string fmt,
                         std::unordered_map<std::string, FunctionInfo> fmap,
                         std::string type_key,
                         std::function<std::string(const std::string&)> fget_source)
    : data_(data),
      fmt_(fmt),
      fmap_(fmap),
      type_key_(type_key),
      fget_source_(fget_source) {}

  PackedFunc GetFunction(
        const std::string& name,
        const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return PackedFunc();
  }

  std::string GetSource(const std::string& format) final {
    if (fget_source_ != nullptr) {
      return fget_source_(format);
    } else {
      return data_;
    }
  }

  const char* type_key() const {
    return type_key_.c_str();
  }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    CHECK_EQ(fmt, fmt_)
        << "Can only save to format=" << fmt_;
    std::string meta_file = GetMetaFilePath(file_name);
    SaveMetaDataToFile(meta_file, fmap_);
    SaveBinaryToFile(file_name, data_);
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

 private:
  std::string data_;
  std::string fmt_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
  std::string type_key_;
  std::function<std::string(const std::string&)> fget_source_;
};

runtime::Module DeviceSourceModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string type_key,
    std::function<std::string(const std::string&)> fget_source) {
  auto n = make_object<DeviceSourceModuleNode>(data, fmt, fmap, type_key, fget_source);
  return runtime::Module(n);
}

// Simulator function
class ONNXModuleNode : public runtime::ModuleNode {
 public:
	ONNXModuleNode(std::string code,
                   std::string fmt)
      : code_(code), fmt_(fmt) {}
  const char* type_key() const {
    return "onnx";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "ONNX Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return PackedFunc();
  }

  std::string GetSource(const std::string& format) final {
	return code_;
  }

  std::vector<std::string> Split(const std::string s, char delim) {
	  std::vector<std::string> result;
      std::stringstream ss (s);
      std::string item;
      while (getline (ss, item, delim)) {
          result.push_back(item);
      }

      return result;
  }
  std::string ConvertEscape(std::string data){
	  std::stringstream ss;
	  for(size_t i=0;i <data.size();i++){
		  if (data[i] == '\\'){
			  switch(data[++i])
			  {
			  	  case 'x':{
			  		  std::string byte = data.substr(++i,2);
			  		  ss<<(char)std::stoi(byte,0,16);
			  		  i += 1;
			  		  break;
			  	  }case 't':{
			  		  ss<<'\t';
			  		  break;
			  	  }
			  	  case 'n':{
			  		  ss<<'\n';
			  		  break;
			  	  }
			  	  default:{
			  		  ss<<'\\'<<data[i];
			  		  break;
			  	  }
			  }
		  }
		  else{
			  ss<<data[i];
		  }
	  }
	  return ss.str();
  }
  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string folder;
    size_t pos = file_name.find_last_of("\\/");
    if(pos!=std::string::npos){
    	folder = file_name.substr(0,pos+1);
    }else{
    	folder = file_name+"/";
    }
    auto datas = Split(code_,'>');
    if (fmt == "onnx") {
      CHECK_NE(code_.size(), 0);
      std::stringstream ss;
	  for(auto data : datas){
		  auto split_data = Split(data,'<');
		  ss<<folder<<split_data[0].c_str()<<"."<<fmt;
		  SaveBinaryToFile(ss.str(), ConvertEscape(split_data[1]));
		  ss.str("");
		  ss.clear();

	  }
    } else {
      CHECK_EQ(fmt, fmt_)
          << "Can only save to format=" << fmt_;
    }
  }

 protected:
  std::string code_;
  std::string fmt_;
};

runtime::Module ONNXModuleCreate(String code, std::string fmt) {
	/* Here, use String instead of std::string, because some byte info
	 * will lost while calling PackedFunc object.
	 */
  auto n = make_object<ONNXModuleNode>(code, fmt);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.SourceModuleCreate")
.set_body_typed(SourceModuleCreate);

TVM_REGISTER_GLOBAL("runtime.CSourceModuleCreate")
.set_body_typed(CSourceModuleCreate);

TVM_REGISTER_GLOBAL("runtime.ONNXModuleCreate")
.set_body_typed(ONNXModuleCreate);
}  // namespace codegen
}  // namespace tvm
