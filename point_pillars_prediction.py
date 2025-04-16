import os
from glob import glob
import numpy as np
import tensorflow as tf
from processors_static import SimpleDataGenerator
from inference_utils import generate_bboxes_from_pred, GroundTruthGenerator, focal_loss_checker, rotational_nms
from readers import KittiDataReader
from config import Parameters
from network import build_point_pillar_graph
import pandas as pd
import time
DATA_ROOT = "/home/tersiteab/Documents/PointCloudResearch/PointPillars_HSC/dataset/kitti/training"
MODEL_ROOT = "/home/tersiteab/Documents/PointCloudResearch/PointPillars_HSC/logs"




if __name__ == "__main__":

    params = Parameters()
    params.batch_size = 10
    
    pillar_net = build_point_pillar_graph(params)
    pillar_net.load_weights(os.path.join(MODEL_ROOT, "model.h5"))
    # tf.saved_model.save(pillar_net, 'saved_model_dir4')
    pillar_net.summary()
# 
    data_reader = KittiDataReader()
    begin_idx = 0
    end_idx = 15

    lidar_files = sorted(glob(os.path.join(DATA_ROOT, "velodyne", "*.bin")))[begin_idx:end_idx]
    label_files = sorted(glob(os.path.join(DATA_ROOT, "label_2", "*.txt")))[begin_idx:end_idx]
    calibration_files = sorted(glob(os.path.join(DATA_ROOT, "calib", "*.txt")))[begin_idx:end_idx]
    assert len(lidar_files) == len(label_files) == len(calibration_files), "Input dirs require equal number of files."
    eval_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_files, label_files, calibration_files)
    # spa = calculate_sparsity(pillar_net)
    print("BATCH SIZE:",params.batch_size)
    # print(eval_gen.__getitem__(9))
    pillars = []

    s = time.time()
    occupancy, position, size, angle, heading, classification = pillar_net.predict(eval_gen,
                                                                                   batch_size=params.batch_size)
    # e = time.time()
    # print("Time of inference", e-s)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    # set_boxes, confidences = [], []
    # print(len(occupancy.shape))
    # loop_range = occupancy.shape[0] if len(occupancy.shape) == 4 else 1
    # print(loop_range)
    # for i in range(loop_range):
    #     v = generate_bboxes_from_pred(occupancy[i], position[i], size[i], angle[i], heading[i],
    #                                                classification[i], params.anchor_dims, occ_threshold=0.3)
    #     # print("------------>",v)#,occupancy[i],classification[i])
    #     if len(v)>0:
    #         set_boxes.append(v)
    #         confidences.append([float(boxes.conf) for boxes in set_boxes[-1]])
    # print('Scene 1: Box predictions with occupancy > occ_thr: ', len(set_boxes[0]))
    
    # if len(set_boxes)>0:

    # # NMS
    #     nms_boxes = rotational_nms(set_boxes, confidences, occ_threshold=0.3, nms_iou_thr=0.5)

    #     print('Scene 1: Boxes after NMS with iou_thr: ', len(nms_boxes[0]))

    # # Do all the further operations on predicted_boxes array, which contains the predicted bounding boxes
    #     gt_gen = GroundTruthGenerator(data_reader, label_files, calibration_files, network_format=False)
    #     gt_gen0 = GroundTruthGenerator(data_reader, label_files, calibration_files, network_format=True)
    #     for seq_boxes, gt_label, gt0 in zip(nms_boxes, gt_gen, gt_gen0):
    #         print("---------- New Scenario ---------- ")
    #         focal_loss_checker(gt0[0], occupancy[0], n_occs=-1)
    #         print("---------- ------------ ---------- ")
    #         for gt in gt_label:
    #             print(gt)
    #         for pred in seq_boxes:
    #             print(pred)



'''

python -m tf2onnx.convert \
  --saved-model saved_model_dir3 \
  --output model.onnx \
  --opset 17 \
  --inputs "pillars_input:0[10,12000,100,7],pillars_indices:0[10,12000,3]"

'''