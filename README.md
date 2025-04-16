# PointPillars Inference with TVM + Custom Preprocessing (Pillarization)

This README documents the full process of deploying a PointPillars model using TVM for GPU inference, with preprocessing done via a custom C++ shared library.

---

## 1. TVM Installation

You need TVM with CUDA support enabled.

```bash
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
mkdir build
cp cmake/config.cmake build

# Edit build/config.cmake
# Set these:
# set(USE_CUDA ON)
# set(USE_LLVM /path/to/llvm-config)
# set(BUILD_STATIC_RUNTIME OFF)

cd build
cmake ..
make -j$(nproc)

```
Then either set python path or install it as a package on your conda environment (I used the latter)

Setting path: 
```
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

OR 

```
cd to/tvm/path/python
pip install -e .
```
## 2. Build Shared Library for Pillarization (C++)
In src/

```
g++ -std=c++17 -O3 -fPIC -shared point_pillars_.cpp -o libpillars.so

```
(optional) Test the shared library with test.cpp
## 3. Model Compilation
Export Tesnorflow model (optional as it is already available)

```
python export_model.py
```

convert tensorflow model to onnx. The model accepts two models inputs: pillars_input:0[10,12000,100,7] and pillars_indices:0[10,12000,3], so pass that as inputs

```
python -m tf2onnx.convert \
  --saved-model saved_model_dir4 \
  --output model_.onnx \
  --opset 17 \
  --inputs "pillars_input:0[10,12000,100,7],pillars_indices:0[10,12000,3]"
```

Then compile the model using TVM for CUDA target as follow.

```
python tvm_conv.py
```

## 4. Lastly, putting all together
In cpp/
- set the appropriate paths in CMakeLists.txt and main.cpp. Do the usual build+execute

```
mkdir build && cd build
cmake ..
make 
./pointpillars_driver
```

Find installation of the original code on _README.md