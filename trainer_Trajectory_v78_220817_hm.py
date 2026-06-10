# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:58:00 2022

@author: hossam.m
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.get("TRAJ_GPU", "1")

import logging
import math
import sys

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # 屏蔽 C++ 层的 INFO/WARNING
tf.get_logger().setLevel('ERROR')                  # 只保留 ERROR 级别的 TF 日志

import dataset_Trajectory_v78_220817_hm
import cnn_gru_v7_Trajectory_v78_220817_hm

# ★ 新增: 用 round.json 注入的 TRAJ_* 覆盖 options (没有 env 时行为不变)
from env_overrides import apply_trainer_env_overrides


# tf.config.experimental_run_functions_eagerly(True)  # DISABLED: run @tf.function in graph mode for speed


def train_finetune(model_file, dataset_descriptor_file, log_directory):
    logging.info('Using dataset from %s', dataset_descriptor_file)
    logging.info('Model to be saved to %s', model_file)
    logging.info('Logging to %s', log_directory)
    splits, future_length = dataset_Trajectory_v78_220817_hm.load_data(dataset_descriptor_file, True, True, False)
    options = {
        'dim_rnns': [128, 64],
        'dim_feature': 2,
        'distance_weight': 10,
        'angle_weight': 10,
        'time_weight': 3,
        'fit_weight': 10,
        'distance_tolerance_loose': 1.5,
        'distance_tolerance_strict': 3.0,
        'angle_tolerance': 5 * math.pi / 180,
        'hardness': 99.5,
        'optimizers': [
            tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1),
            tf.keras.optimizers.Adam(learning_rate=0.0002, clipnorm=1),
            tf.keras.optimizers.Adam(learning_rate=0.00004, clipnorm=1),
            tf.keras.optimizers.Adam(learning_rate=0.000008, clipnorm=1)
        ],
        'dropout_rate': 0.00,
        'patience': 50,
        'dispFreq': 500,
        'validFreq': 500,
        'saveFreq': 500,
        'sampleFreq': 500,
        'reload_history': False
    }

    # ★ 新增: env 覆盖 (TRAJ_DISTANCE_WEIGHT / ANGLE_WEIGHT / TIME_WEIGHT /
    #   FIT_WEIGHT / DIST_TOL_* / HARDNESS / PATIENCE / LR0..3)。没有 env 时原样返回。
    options = apply_trainer_env_overrides(options)

    logging.info('Options: %s', options)
    cnn_gru_v7_Trajectory_v78_220817_hm.train(
        train_set=dataset_Trajectory_v78_220817_hm.StaticDataset(splits['train_set'], future_length),
        valid_set=splits['valid_set'],
        saveto=model_file,
        tmpdir=log_directory,
        **options)


def train1(model_file, dataset_descriptor_file, log_directory):
    logging.info('Using dataset from %s', dataset_descriptor_file)
    logging.info('Model to be saved to %s', model_file)
    logging.info('Logging to %s', log_directory)
    splits, future_length = dataset_Trajectory_v78_220817_hm.load_data(dataset_descriptor_file, True, True, False)
    options = {
        'dim_rnns': [128, 64],
        'dim_feature': 2,
        'distance_weight': 10,
        'angle_weight': 10,
        'time_weight': 1,
        'fit_weight': 10,
        'distance_tolerance_loose': 1.5,
        'distance_tolerance_strict': 3.0,
        'angle_tolerance': 5 * math.pi / 180,
        'hardness': 99.5,
        'optimizers': [
            tf.keras.optimizers.Adam(learning_rate=0.001),
            tf.keras.optimizers.Adam(learning_rate=0.0002),
            tf.keras.optimizers.Adam(learning_rate=0.00004),
            tf.keras.optimizers.Adam(learning_rate=0.000008)
        ],
        'dropout_rate': 0.00,
        'patience': 200,
        'dispFreq': 500,
        'validFreq': 4000,
        'saveFreq': 4000,
        'sampleFreq': 4000,
        'reload_history': True
    }
    logging.info('Options: %s', options)
    cnn_gru_v7_Trajectory_v78_220817_hm.train(
        train_set=dataset_Trajectory_v78_220817_hm.StaticDataset(splits['train_set'], future_length),
        valid_set=splits['valid_set'],
        saveto=model_file,
        tmpdir=log_directory,
        **options)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('trainer.py MODEL_FILE DATASET_DESCRIPTOR_FILE LOG_DIRECTORY', file=sys.stderr)
        sys.exit(1)
    logging.basicConfig(level=logging.INFO, style='{', datefmt='%Y-%m-%d %H:%M:%S',
                        format='{asctime} {levelname}: {message}')
    model_file = sys.argv[1]
    dataset_descriptor_file = sys.argv[2]
    log_directory = sys.argv[3]
    # train1(model_file,dataset_descriptor_file,log_directory)
    train_finetune(model_file, dataset_descriptor_file, log_directory)



