# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:10:00 2022

@author: hossam.m
"""

import pandas as pd
import numpy
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger
import argparse
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dropout
# from load_args import load_args
import math
import logging
import json
# import tensorflow_probability as tfp
import sys
import random
import time

from tensorflow.python.eager import context
import math
from scipy import signal

#TUNE_TIME_ONLY=False
#TUNE_POLY_ONLY = False
#USE_APL_LOSS = False
#USE_TIME_CEIL, TIME_CEIL = False, 0.5

from env_overrides import flags_from_env
_F = flags_from_env()
TUNE_TIME_ONLY = _F["TUNE_TIME_ONLY"]
TUNE_POLY_ONLY = _F["TUNE_POLY_ONLY"]
USE_APL_LOSS   = _F["USE_APL_LOSS"]
USE_TIME_CEIL  = _F["USE_TIME_CEIL"]
TIME_CEIL      = _F["TIME_CEIL"]
APL_TARGET     = _F["APL_TARGET"]          # 提到模块级, 让 train_step 用它

if TUNE_TIME_ONLY == True:
    print("!!!!!!!!!!!!!!!!! Currently on Tune time only mode !!!!!!!!!!!!!!!!")
if TUNE_POLY_ONLY:
    print("!!!!!!!!!!!!!!!!! Currently on Tune poly only mode !!!!!!!!!!!!!!!!")
if USE_APL_LOSS:
    print("!!!!!!!!!!!!!!!!! Currently using apl floow loss !!!!!!!!!!!!!!!!")
if USE_TIME_CEIL:
    print("!!!!!!!!!!!!!!!!! Currently using time ceil !!!!!!!!!!!!!!!!")

