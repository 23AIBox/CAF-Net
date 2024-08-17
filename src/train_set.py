# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.abspath(".."))

import numpy
from src import config
from src.utils.LoadDataset import getEmbeddingAndLabel, getEmbeddingAndLabelMulti
from tensorflow.python.keras import backend as K
from loguru import logger

s = 584949
p = 7371
w0 = s / (2 * (s - p))
w1 = s / (2 * p)


def weighted_bce(y_true, y_pred):
    global w0
    global w1
    return (w1) * y_true * (-K.log(y_pred * 0.999999 + 0.000001)) + (w0) * (
        1.0 - y_true) * (-K.log((1.0 - y_pred) * 0.999999 + 0.000001))


def load_train_set(ds='I-1'):
    if isinstance(ds, str):
        x, y = getEmbeddingAndLabel(ds)
    elif isinstance(ds, list):
        x, y = getEmbeddingAndLabelMulti(ds, use_cache=False)

    logger.info("total:", y.shape, " positive:%d" % (numpy.count_nonzero(y)))
    global s
    global p
    global w0
    global w1
    p = numpy.count_nonzero(y)
    s = y.shape[0]
    w0 = s / (2 * (s - p))
    w1 = s / (2 * p)
    x = x.reshape(len(x), config.seq_len, 8)
    return x, y
