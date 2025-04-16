from typing import List
import numpy as np
import tensorflow as tf

# from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.utils import Sequence
from config import Parameters
# from point_pillars import createPillars, createPillarsTarget, some_thing
from readers import DataReader, Label3D
from sklearn.utils import shuffle
import sys
import pandas as pd
import time

import ctypes

# Load the shared library
lib = ctypes.cdll.LoadLibrary("/home/tersiteab/Documents/PointCloudResearch/PointPillars_HSC/src/libpillars.so")

# Define createPillars signature
lib.createPillars.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
    ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float,
    ctypes.c_bool
]

# Define createPillarsTarget signature
lib.createPillarsTargetC.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # objectPositions
    ctypes.POINTER(ctypes.c_float),  # objectDimensions
    ctypes.POINTER(ctypes.c_float),  # objectYaws
    ctypes.POINTER(ctypes.c_int),    # objectClassIds

    ctypes.POINTER(ctypes.c_float),  # anchorDimensions
    ctypes.POINTER(ctypes.c_float),  # anchorZHeights
    ctypes.POINTER(ctypes.c_float),  # anchorYaws

    ctypes.c_float,  # positiveThreshold
    ctypes.c_float,  # negativeThreshold

    ctypes.c_int,  # nbObjects
    ctypes.c_int,  # nbAnchors
    ctypes.c_int,  # nbClasses
    ctypes.c_int,  # downscalingFactor

    ctypes.c_float, ctypes.c_float,  # xStep, yStep
    ctypes.c_float, ctypes.c_float,  # xMin, xMax
    ctypes.c_float, ctypes.c_float,  # yMin, yMax
    ctypes.c_float, ctypes.c_float,  # zMin, zMax

    ctypes.c_int,  # xSize
    ctypes.c_int,  # ySize

    ctypes.POINTER(ctypes.c_float),  # tensor_out
    ctypes.POINTER(ctypes.c_int),    # posCnt_out
    ctypes.POINTER(ctypes.c_int),    # negCnt_out

    ctypes.c_bool  # printTime
]


class PillarizationLibrary:
    def __init__(self, lib):
        self.lib = lib

    def create_pillars(self, points, max_points_per_pillar, max_pillars, x_step, y_step, x_min, x_max, y_min, y_max, z_min, z_max, print_time=False):
        num_points = points.shape[0]
        tensor_out = np.zeros((1, max_pillars, max_points_per_pillar, 7), dtype=np.float32)
        indices_out = np.zeros((1, max_pillars, 3), dtype=np.int32)
        # print("before lib",(points).shape)
        self.lib.createPillars(
            points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(num_points),
            tensor_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            indices_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(max_points_per_pillar),
            ctypes.c_int(max_pillars),
            ctypes.c_float(x_step), ctypes.c_float(y_step),
            ctypes.c_float(x_min), ctypes.c_float(x_max),
            ctypes.c_float(y_min), ctypes.c_float(y_max),
            ctypes.c_float(z_min), ctypes.c_float(z_max),
            ctypes.c_bool(print_time)
        )
        return tensor_out, indices_out

    def create_pillars_target(self, target_positions, target_dimensions, target_yaw, target_class, anchor_dims, anchor_z, anchor_yaw,
                              positive_threshold, negative_threshold, nb_classes, downscaling_factor,
                              x_step, y_step, x_min, x_max, y_min, y_max, z_min, z_max, print_time=False):
        nb_objects = target_positions.shape[0]
        nb_anchors = anchor_dims.shape[0]
        x_size = int(np.floor((x_max - x_min) / (x_step * downscaling_factor)))
        y_size = int(np.floor((y_max - y_min) / (y_step * downscaling_factor)))

        tensor_out = np.zeros((nb_objects, x_size, y_size, nb_anchors, 10), dtype=np.float32)
        pos_cnt = np.zeros((1,), dtype=np.int32)
        neg_cnt = np.zeros((1,), dtype=np.int32)
        # print("FROM the other place", type(nb_classes))
        self.lib.createPillarsTargetC(
            target_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            target_dimensions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            target_yaw.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            target_class.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            anchor_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            anchor_z.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            anchor_yaw.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(positive_threshold),
            ctypes.c_float(negative_threshold),
            ctypes.c_int(nb_objects),
            ctypes.c_int(nb_anchors),
            ctypes.c_int(nb_classes),
            ctypes.c_int(downscaling_factor),
            ctypes.c_float(x_step),
            ctypes.c_float(y_step),
            ctypes.c_float(x_min),
            ctypes.c_float(x_max),
            ctypes.c_float(y_min),
            ctypes.c_float(y_max),
            ctypes.c_float(z_min),
            ctypes.c_float(z_max),
            ctypes.c_int(x_size),
            ctypes.c_int(y_size),
            tensor_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pos_cnt.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            neg_cnt.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_bool(print_time)
        )

        return tensor_out, int(pos_cnt[0]), int(neg_cnt[0])



def select_best_anchors(arr):
    dims = np.indices(arr.shape[1:])
    # arr[..., 0:1] gets the occupancy value from occ in {-1, 0, 1}, i.e. {bad match, neg box, pos box}
    ind = (np.argmax(arr[..., 0:1], axis=0),) + tuple(dims)
    return arr[ind]


class DataProcessor(Parameters):

    def __init__(self):
        super(DataProcessor, self).__init__()
        anchor_dims = np.array(self.anchor_dims, dtype=np.float32)
        self.anchor_dims = anchor_dims[:, 0:3]
        self.anchor_z = anchor_dims[:, 3]
        self.anchor_yaw = anchor_dims[:, 4]
        # Counts may be used to make statistic about how well the anchor boxes fit the objects
        self.pos_cnt, self.neg_cnt = 0, 0
        self.pillars_lib = PillarizationLibrary(lib)

    @staticmethod
    def transform_labels_into_lidar_coordinates(labels: List[Label3D], R: np.ndarray, t: np.ndarray):
        transformed = []
        for label in labels:
            label.centroid = label.centroid @ np.linalg.inv(R).T - t
            label.dimension = label.dimension[[2, 1, 0]]
            label.yaw -= np.pi / 2
            while label.yaw < -np.pi:
                label.yaw += (np.pi * 2)
            while label.yaw > np.pi:
                label.yaw -= (np.pi * 2)
            transformed.append(label)
        return labels

    def make_point_pillars(self, points: np.ndarray):
        #print('points.shape',points.shape)
        assert points.ndim == 2
        assert points.shape[1] == 4
        assert points.dtype == np.float32

        pillars, indices = self.pillars_lib.create_pillars(points,
                                         self.max_points_per_pillar,
                                         self.max_pillars,
                                         self.x_step,
                                         self.y_step,
                                         self.x_min,
                                         self.x_max,
                                         self.y_min,
                                         self.y_max,
                                         self.z_min,
                                         self.z_max,
                                         False)
        #print('pillar pillar',pillars.shape)
        #print('indicies', indices)
        #x = some_thing(20)
        #print('function test', x)

        return pillars, indices

    def make_ground_truth(self, labels: List[Label3D]):

        # filter labels by classes (cars, pedestrians and Trams)
        # Label has 4 properties (Classification (0th index of labels file),
        # centroid coordinates, dimensions, yaw)
        labels = list(filter(lambda x: x.classification in self.classes, labels))

        if len(labels) == 0:
            pX, pY = int(self.Xn / self.downscaling_factor), int(self.Yn / self.downscaling_factor)
            a = int(self.anchor_dims.shape[0])
            return np.zeros((pX, pY, a), dtype='float32'), np.zeros((pX, pY, a, self.nb_dims), dtype='float32'), \
                   np.zeros((pX, pY, a, self.nb_dims), dtype='float32'), np.zeros((pX, pY, a), dtype='float32'), \
                   np.zeros((pX, pY, a, self.nb_classes), dtype='float64')

        # For each label file, generate these properties except for the Don't care class
        target_positions = np.array([label.centroid for label in labels], dtype=np.float32)
        target_dimension = np.array([label.dimension for label in labels], dtype=np.float32)
        target_yaw = np.array([label.yaw for label in labels], dtype=np.float32)
        target_class = np.array([self.classes[label.classification] for label in labels], dtype=np.int32)

        assert np.all(target_yaw >= -np.pi) & np.all(target_yaw <= np.pi)
        assert len(target_positions) == len(target_dimension) == len(target_yaw) == len(target_class)
        # print("downscaling_factor type:",type(self.x_step))
        target, pos, neg = self.pillars_lib.create_pillars_target(target_positions,
                                               target_dimension,
                                               target_yaw,
                                               target_class,
                                               self.anchor_dims,
                                               self.anchor_z,
                                               self.anchor_yaw,
                                               self.positive_iou_threshold,
                                               self.negative_iou_threshold,
                                               self.nb_classes,
                                               self.downscaling_factor,
                                               self.x_step,
                                               self.y_step,
                                               self.x_min,
                                               self.x_max,
                                               self.y_min,
                                               self.y_max,
                                               self.z_min,
                                               self.z_max,
                                               False)
        self.pos_cnt += pos
        self.neg_cnt += neg
        # print('target shape',target.shape)
        #print('target',target)
        # print('pos',pos)
        # print('neg',neg)
        # print("nb_classes",self.nb_classes)
        # print("anchor_dims",self.anchor_dims.shape)
        # return a merged target view for all objects in the ground truth and get categorical labels
        sel = select_best_anchors(target)
        ohe = tf.keras.utils.to_categorical(sel[..., 9], num_classes=self.nb_classes, dtype='float64')

        return sel[..., 0], sel[..., 1:4], sel[..., 4:7], sel[..., 7], sel[..., 8], ohe


class SimpleDataGenerator(DataProcessor, Sequence):
    """ Multiprocessing-safe data generator for training, validation or testing, without fancy augmentation """

    def __init__(self, data_reader: DataReader, batch_size: int, lidar_files: List[str], label_files: List[str] = None,
                 calibration_files: List[str] = None):
        super(SimpleDataGenerator, self).__init__()
        self.data_reader = data_reader
        self.batch_size = batch_size
        self.lidar_files = lidar_files
        self.label_files = label_files
        self.calibration_files = calibration_files

        assert (calibration_files is None and label_files is None) or \
               (calibration_files is not None and label_files is not None)

        if self.calibration_files is not None:
            assert len(self.calibration_files) == len(self.lidar_files)
            assert len(self.label_files) == len(self.lidar_files)

    def __len__(self):
        return len(self.lidar_files) // self.batch_size

    def __getitem__(self, batch_id: int):
        file_ids = np.arange(batch_id * self.batch_size, self.batch_size * (batch_id + 1))
        #         print("inside getitem")
        # print(file_ids)
        pillars = []
        voxels = []
        occupancy = []
        position = []
        size = []
        angle = []
        heading = []
        classification = []
        # print("LIDAR FILE NO", file_ids.shape)
# 
        for i in file_ids:
            # print("fine no ",i, self.lidar_files[i])
            lidar = self.data_reader.read_lidar(self.lidar_files[i])
            # For each file, dividing the space into a x-y grid to create pillars
            # Voxels are the pillar ids
            s = time.time()
            # print(lidar.shape)
            pillars_, voxels_ = self.make_point_pillars(lidar) # input
            e = time.time()
            # print("Pillarization ouputs",pillars_.shape,voxels_.shape)
            pillars.append(pillars_)
            voxels.append(voxels_)

            if self.label_files is not None:
                label = self.data_reader.read_label(self.label_files[i])
                R, t = self.data_reader.read_calibration(self.calibration_files[i])
                # Labels are transformed into the lidar coordinate bounding boxes
                # Label has 7 values, centroid, dimensions and yaw value.
                label_transformed = self.transform_labels_into_lidar_coordinates(label, R, t)
                # These definitions can be found in point_pillars.cpp file
                # We are splitting a 10 dim vector that contains this information.
                occupancy_, position_, size_, angle_, heading_, classification_ = self.make_ground_truth(    #output
                    label_transformed)
                #print("occupancy, position, size, angle, heading, classification",  np.array(occupancy_).shape, np.array(position_).shape, np.array(size_).shape, np.array(angle_).shape, np.array(heading_).shape, np.array(classification_).shape)
                occupancy.append(occupancy_)
                position.append(position_)
                size.append(size_)
                angle.append(angle_)
                heading.append(heading_)
                classification.append(classification_)

        pillars = np.concatenate(pillars, axis=0)
        voxels = np.concatenate(voxels, axis=0)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(voxels,"pillars",pillars)
        if self.label_files is not None:
            occupancy = np.array(occupancy)
            position = np.array(position)
            size = np.array(size)
            angle = np.array(angle)
            heading = np.array(heading)
            classification = np.array(classification)
            # print("pillars, voxels, occupancy.shape, position.shape, size.shape, angle.shape, heading.shape, classification.shape",pillars.shape, voxels.shape,occupancy.shape, position.shape, size.shape, angle.shape, heading.shape, classification.shape)
            #df = pd.DataFrame (position[0,:,:,:,:])
            #filepath = 'my_excel_file.xlsx'
            #df.to_excel(filepath, index=False)

            # print("here?")
            return [pillars, voxels], [occupancy, position, size, angle, heading, classification]
        else:
            return [pillars, voxels]

    def on_epoch_end(self):
        #         print("inside epoch")
        if self.label_files is not None:
            self.lidar_files, self.label_files, self.calibration_files = \
                shuffle(self.lidar_files, self.label_files, self.calibration_files)
