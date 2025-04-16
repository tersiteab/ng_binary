import os
from glob import glob
import numpy as np
import tensorflow as tf
from config import Parameters
from network import build_point_pillar_graph
import pandas as pd
import time

MODEL_ROOT = "logs"



params = Parameters()
params.batch_size = 10

pillar_net = build_point_pillar_graph(params)
pillar_net.load_weights(os.path.join(MODEL_ROOT, "model.h5"))
tf.saved_model.save(pillar_net, 'saved_model_dir4')