#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#include <dmlc/logging.h>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <dirent.h>
#include <sys/types.h>
#include <algorithm>
#include <dlfcn.h>
// #include <dlpack/dlpack.h>

namespace fs = std::filesystem;
using namespace tvm::runtime;
extern "C" {
    void createPillars(
        const float* points, int num_points,
        float* tensor_out, int* indices_out,
        int maxPointsPerPillar,
        int maxPillars,
        float xStep,
        float yStep,
        float xMin,
        float xMax,
        float yMin,
        float yMax,
        float zMin,
        float zMax,
        bool printTime
    );

    void createPillarsTargetC(
        const float* objectPositions,
        const float* objectDimensions,
        const float* objectYaws,
        const int* objectClassIds,
        const float* anchorDimensions,
        const float* anchorZHeights,
        const float* anchorYaws,
        float positiveThreshold,
        float negativeThreshold,
        int nbObjects,
        int nbAnchors,
        int nbClasses,
        int downscalingFactor,
        float xStep,
        float yStep,
        float xMin,
        float xMax,
        float yMin,
        float yMax,
        float zMin,
        float zMax,
        int xSize,
        int ySize,
        float* tensor_out,
        int* posCnt_out,
        int* negCnt_out,
        bool printTime
    );
    void createPillarsTargetC(
        const float* objectPositions,
        const float* objectDimensions,
        const float* objectYaws,
        const int* objectClassIds,
        const float* anchorDimensions,
        const float* anchorZHeights,
        const float* anchorYaws,
        float positiveThreshold,
        float negativeThreshold,
        int nbObjects,
        int nbAnchors,
        int nbClasses,
        int downscalingFactor,
        float xStep,
        float yStep,
        float xMin,
        float xMax,
        float yMin,
        float yMax,
        float zMin,
        float zMax,
        int xSize,
        int ySize,
        float* tensor_out,
        int* posCnt_out,
        int* negCnt_out,
        bool printTime
    );
}


std::vector<std::vector<float>> read_all_bin_files(const std::string& dir_path, std::vector<std::string>& file_list) {
    std::vector<std::vector<float>> all_data;

    for (int i = 0; i < 10; ++i) {
        std::ostringstream filename;
        filename << dir_path << std::setfill('0') << std::setw(6) << i << ".bin";
        std::string full_path = filename.str();
        file_list.push_back(full_path);

        std::ifstream input(full_path, std::ios::binary);
        if (!input) {
            std::cerr << "Failed to open file: " << full_path << std::endl;
            all_data.push_back({});
            continue;
        }
        std::vector<float> data;
        float val;
        while (input.read(reinterpret_cast<char*>(&val), sizeof(float))) {
            data.push_back(val);
        }
        all_data.push_back(std::move(data));
    }

    return all_data;
}

int main() {
    DLDevice dev;
    dev.device_type = DLDeviceType::kDLCUDA;  // or kDLCPU if using CPU
    dev.device_id = 0;

    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("/home/tersiteab/Documents/PointCloudResearch/PointPillars_HSC/pointpillars_gpu.so");
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");
    void* handle = dlopen("/home/tersiteab/Documents/PointCloudResearch/PointPillars_HSC/pointpillars_gpu.so", RTLD_LAZY | RTLD_LOCAL);
// ÷÷÷

    constexpr int N = 10; // Number of points
    constexpr int maxPointsPerPillar = 100;
    constexpr int maxPillars = 12000;
    constexpr float xMin = 0.0f, xMax =80.64f;
    constexpr float yMin = -40.32f, yMax = 40.32f;
    constexpr float zMin = -1.0f, zMax = 3.0f;
    constexpr float xStep = 0.16f;
    constexpr float yStep = 0.16f;


    
    //update this path
    std::string dir_path = "/home/tersiteab/Documents/PointCloudResearch/PointPillars_HSC/dataset/kitti/training/velodyne/"; // Update this to your full path

    std::vector<std::string> bin_files;
    std::vector<std::vector<float>> stored_inputs = read_all_bin_files(dir_path, bin_files);

    std::cout << "Loaded " << stored_inputs.size() << " files." << std::endl;
    constexpr int tensorSizePerBatch = maxPillars * maxPointsPerPillar * 7;
    constexpr int indicesSizePerBatch = maxPillars * 3;
    int batchSize=10;
    // Just a verification printout (optional)
    // std::vector<float> batch_tensor_out(batchSize * tensorSizePerBatch);
    // std::vector<int> batch_indices_out(batchSize * indicesSizePerBatch);
    std::vector<std::vector<float>> batch_tensor_out(batchSize, std::vector<float>(tensorSizePerBatch));
    std::vector<std::vector<int>> batch_indices_out(batchSize, std::vector<int>(indicesSizePerBatch));


    for (size_t i = 0; i < stored_inputs.size(); ++i) {
        std::cout << "File: " << bin_files[i] << " loaded with " << stored_inputs[i].size() << " points.\n";
        constexpr int tensorSize = 1 * maxPillars * maxPointsPerPillar * 7;
        constexpr int indicesSize = 1 * maxPillars * 3;

        std::vector<float> tensor_out(tensorSize, 0);
        std::vector<int> indices_out(indicesSize, 0);
        int n = stored_inputs[i].size()/4;
        createPillars(
            stored_inputs[i].data(),n ,
            tensor_out.data(), indices_out.data(),
            maxPointsPerPillar, maxPillars,
            xStep, yStep,
            xMin, xMax,
            yMin, yMax,
            zMin, zMax,
            true // printTime
        );
        // std::memcpy(batch_tensor_out.data() + i * tensorSizePerBatch, tensor_out.data(), tensorSizePerBatch * sizeof(float));

        std::cout<<"After Pillarization "<<tensor_out.size()<<" and indices "<<indices_out.size()<<std::endl;
        // std::memcpy(batch_tensor_out.data() + i * tensorSizePerBatch, tensor_out.data(), tensorSizePerBatch * sizeof(float));
        // std::memcpy(batch_indices_out.data() + i * indicesSizePerBatch, indices_out.data(), indicesSizePerBatch * sizeof(int));
        
        batch_tensor_out[i] = tensor_out;
        batch_indices_out[i] = indices_out;

    }
    std::vector<float> flat_tensor;
    std::vector<int> flat_indices;
    for (int i = 0; i < batchSize; ++i) {
        flat_tensor.insert(flat_tensor.end(), batch_tensor_out[i].begin(), batch_tensor_out[i].end());
        flat_indices.insert(flat_indices.end(), batch_indices_out[i].begin(), batch_indices_out[i].end());
    }

    int64_t input_shape[4] = {batchSize, maxPillars, maxPointsPerPillar, 7};
    int64_t indices_shape[3] = {batchSize, maxPillars, 3};
    std::cout<<"shape:::::::>"<<batch_indices_out.size()<<std::endl;
   
    NDArray tensor_host = NDArray::Empty({batchSize, maxPillars, maxPointsPerPillar, 7}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    NDArray indices_host = NDArray::Empty({batchSize, maxPillars, 3}, {kDLInt, 32, 1}, {kDLCPU, 0});
    
    std::memcpy(tensor_host->data, flat_tensor.data(), flat_tensor.size() * sizeof(float));
    std::memcpy(indices_host->data, flat_indices.data(), flat_indices.size() * sizeof(int));
    std::cout<<"shape:::::::>"<<batch_indices_out.size()<<std::endl;
    // Allocate final input arrays on GPU
    NDArray tensor_nd = NDArray::Empty(tensor_host.Shape(), tensor_host.DataType(), dev);
    NDArray indices_nd = NDArray::Empty(indices_host.Shape(), indices_host.DataType(), dev);
    
    // Copy to CUDA device
    tensor_nd.CopyFrom(tensor_host);
    indices_nd.CopyFrom(indices_host);

    gmod.GetFunction("set_input")("pillars_input", tensor_nd);
    gmod.GetFunction("set_input")("pillars_indices", indices_nd);

    gmod.GetFunction("run")();

    // auto list_func = gmod->GetFunction("");
    // for (const auto& name : list_func) {
    //     std::cout << "Available function: " << name << std::endl;
    // }

    tvm::runtime::PackedFunc get_num_outputs = gmod.GetFunction("get_num_outputs");
    int num_outputs = get_num_outputs();
    std::cout << "Number of outputs: " << num_outputs << std::endl;

    // NDArray out = gmod.GetFunction("get_output")(0);

    // NDArray out0 = gmod.GetFunction("get_output")(0); // occupancy
    // NDArray out1 = gmod.GetFunction("get_output")(1); // position
    // NDArray out2 = gmod.GetFunction("get_output")(2); // size
    // NDArray out3 = gmod.GetFunction("get_output")(3); // angle
    // NDArray out4 = gmod.GetFunction("get_output")(4); // heading
    // NDArray out5 = gmod.GetFunction("get_output")(5); // classification
    
    // const DLTensor* out_tensor = out0.operator->();

    // std::cout << "Output dtype: " 
    //         << out_tensor->dtype.code << ", bits: " 
    //         << out_tensor->dtype.bits << ", lanes: " 
    //         << out_tensor->dtype.lanes << std::endl;


    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");

    // Optional: name the outputs for readability
    std::vector<std::string> output_names = {
        "angle", "classification", "heading", "location", "occupancy", "size"
    };
    std::vector<std::string> output_shapez = {
        "10x252x252x4", "10x252x252x4x4", "10x252x252x4", "10x252x252x4x3", "10x252x252x4", "10x252x252x4x3"
    };
    
    for (int i = 0; i < num_outputs; ++i) {
        NDArray output = get_output(i);
        // const DLTensor* out_tensor = output.operator->();
        // float* out_data = static_cast<float*>(out_tensor->data);
        // std::cout<<"got here"<<out_tensor<<std::endl;
        std::cout << "Output " << i << " (" << output_names[i] << "): ";

        DLDevice device = output->device;
        if (device.device_type == kDLCUDA) {
            std::cout << "Output is on GPU (CUDA), device ID: " << device.device_id << std::endl;
        } else if (device.device_type == kDLCPU) {
            std::cout << "Output is on CPU." << std::endl;
        }

        NDArray output_cpu = NDArray::Empty(output.Shape(), output.DataType(), {kDLCPU, 0});
        output_cpu.CopyFrom(output);

        const DLTensor* out_tensor = output_cpu.operator->();
        float* out_data = static_cast<float*>(out_tensor->data);


        int64_t total_elems = 1;
        std::cout << "Output " << i << " shape: [ ";

        for (int j = 0; j < out_tensor->ndim; ++j) {
            total_elems *= out_tensor->shape[j];
            std::cout << out_tensor->shape[j] << " ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Output from model structure" << i << " (" << output_shapez[i] << "): ";

        std::cout << "Output has " << total_elems << " elements." << std::endl;

        std::cout << "Sample output " << i << ": ";
        for (int j = 0; j < std::min<int64_t>(10, total_elems); ++j) {
            std::cout << out_data[j] << " ";
        }
        std::cout << "\n\n";
        // std::cout<<"got here 2"<<total_elems<<std::endl;
        // for (int j = 0; j < std::min<int64_t>(10, total_elems); ++j) {
        //     std::cout << out_data[j] << " ";
        // }
        // std::cout << std::endl;
    
        // (Optional) Store or further process the result
        // You can get shape, size, etc. if needed
    }
            
    // std::cout << "Output ndim: " << out_tensor->ndim << std::endl;
    // std::cout << "Output shape: ";
    // for (int i = 0; i < out_tensor->ndim; ++i) {
    //     std::cout << out_tensor->shape[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout<<"got here"<<std::endl;
    // std::vector<std::string> output_names = {
    //     "occupancy", "position", "size", "angle", "heading", "classification"
    // };
    

    // for (int i = 0; i < output_names.size(); ++i) {
    //     auto oo =  gmod.GetFunction("get_output");
    //     NDArray out = gmod.GetFunction("get_output")(i);
    //     std::cout<<"got here"<<std::endl;
    //     float* out_data = static_cast<float*>(out->data);
    //     std::cout << output_names[i] << " output sample: ";
    //     for (int j = 0; j < 10; ++j) {
    //         std::cout << out_data[j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // NDArray output = gmod.GetFunction("get_output")(0);
    // float* out_data = static_cast<float*>(output->data);
    // std::cout << "Output sample: ";
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << out_data.size() << " ";
    // }
    // std::cout << std::endl;
    // NDArray output_host = NDArray::Empty(output.Shape(), output.DataType(), {kDLCPU, 0});
    // output_host.CopyFrom(output);

    // float* out_data = static_cast<float*>(output_host->data);
    // std::cout << "Output sample: ";
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << out_data[i] << " ";
    // }
    // std::cout << std::endl;
    //         // NDArray tensor_nd = NDArray::Empty({batchSize, maxPillars, maxPointsPerPillar, 7}, {kDLFloat, 32, 1}, dev);
    // NDArray indices_nd = NDArray::Empty({batchSize, maxPillars, 3}, {kDLInt, 32, 1}, dev);


    // assert(flat_tensor.size() == batchSize * tensorSizePerBatch && "Mismatch in tensor size");
    // assert(flat_indices.size() == batchSize * indicesSizePerBatch && "Mismatch in indices size");

    // std::cout << "Expected tensor size: " << batchSize * tensorSizePerBatch
    //       << ", actual: " << flat_tensor.size() << std::endl;
   
    // std::memcpy(tensor_nd->data, flat_tensor.data(), flat_tensor.size() * sizeof(float));
    // std::memcpy(indices_nd->data, flat_indices.data(), flat_indices.size() * sizeof(int));

    // gmod.GetFunction("set_input")("pillars_input", tensor_nd);
    // gmod.GetFunction("set_input")("pillars_indices", indices_nd);
    // gmod.GetFunction("run")();

    // NDArray output = gmod.GetFunction("get_output")(0);
    // float* out_data = static_cast<float*>(output->data);

    // std::cout << "Output sample: ";
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << out_data[i] << " ";
    // }
    // std::cout << std::endl;
    // std::memcpy(tensor_nd->data, flat_tensor.data(), flat_tensor.size() * sizeof(float));
    // std::memcpy(indices_nd->data, flat_indices.data(), flat_indices.size() * sizeof(int));

    // gmod.GetFunction("set_input")("pillars_input", tensor_nd);
    // gmod.GetFunction("set_input")("pillars_indices", indices_nd);
    // gmod.GetFunction("run")();

    // NDArray output = gmod.GetFunction("get_output")(0);
    // float* out_data = static_cast<float*>(output->data);

    // std::cout << "Output sample: ";
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << out_data[i] << " ";
    // }
    // std::cout << std::endl;
    // Allocate DLTensors
    // DLTensor tensor_dl, indices_dl;
    // DLManagedTensor tensor_mgr, indices_mgr;

    // tensor_dl.data = flat_tensor.data();
    // tensor_dl.device = dev;
    // tensor_dl.ndim = 4;
    // tensor_dl.dtype = {kDLFloat, 32, 1};
    // tensor_dl.shape = input_shape;
    // tensor_dl.strides = nullptr;
    // tensor_dl.byte_offset = 0;

    // indices_dl.data = flat_indices.data();
    // indices_dl.device = dev;
    // indices_dl.ndim = 3;
    // indices_dl.dtype = {kDLInt, 32, 1};
    // indices_dl.shape = indices_shape;
    // indices_dl.strides = nullptr;
    // indices_dl.byte_offset = 0;

    // // Convert to NDArray
    // NDArray tensor_nd = NDArray::Empty({batchSize, maxPillars, maxPointsPerPillar, 7}, {kDLFloat, 32, 1}, dev);
    // NDArray indices_nd = NDArray::Empty({batchSize, maxPillars, 3}, {kDLInt, 32, 1}, dev);

    // // Copy raw data into the allocated, aligned TVM buffer
    // std::memcpy(tensor_nd->data, batch_tensor_out.data(), batch_tensor_out.size() * sizeof(float));
    // std::memcpy(indices_nd->data, batch_indices_out.data(), batch_indices_out.size() * sizeof(int));

    // gmod.GetFunction("set_input")("pillars_input", tensor_nd);
    // gmod.GetFunction("set_input")("pillars_indices", indices_nd);
    // gmod.GetFunction("run")();

    // NDArray output = gmod.GetFunction("get_output")(0);
    // float* out_data = static_cast<float*>(output->data);

    // std::cout << "Output sample: ";
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << out_data[i] << " ";
    // }
    // std::cout << std::endl;

    return 0;
}


// g++ -std=c++17 -I~/tvm/include -I~/tvm/3rdparty/dmlc-core/include -I~/tvm/3rdparty/dlpack/include \
//     -L~/tvm/build -ltvm_runtime -o pointpillars_driver pp_cpp.cpp
