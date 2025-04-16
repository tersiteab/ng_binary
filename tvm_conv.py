
import onnx
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
import tensorflow as tf
from tvm.relay.op.contrib.cutlass import partition_for_cutlass



# Load ONNX model


model=tf.saved_model.load("saved_model_dir4")
print(list(model.signatures.keys()))  # typically ['serving_default']
infer = model.signatures["serving_default"]
# print([tensor.name for tensor in infer.inputs])



onnx_model = onnx.load("model_.onnx")

# Convert ONNX to Relay
input_shapes = {
    "pillars_input:0": (10, 12000, 100, 7),
    "pillars_indices:0": (10, 12000, 3),
}
# shape_dict = {"input_1": (10, 12000, 100, 7)}  # Use actual input name and shape
mod, params = relay.frontend.from_onnx(onnx_model, input_shapes)
print("SUCCESS???")

target_gpu = tvm.target.Target("cuda")

with tvm.transform.PassContext(opt_level=3):
    lib_gpu = relay.build(mod, target=target_gpu, params=params)

# Save
lib_gpu.export_library("pointpillars_gpu.so")