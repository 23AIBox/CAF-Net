# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.abspath(".."))

from src.model import *
from src.train_finetune import Net_training_finetune
from src.train_set import load_train_set
from sklearn.model_selection import train_test_split
import numpy
from loguru import logger


def train():
    Net_model(attn_cnt=2, use_lstm=True)
    x, y = load_train_set()
    # Net_training_without_validate(x, y, 60)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    Net_training_finetune(X_train, y_train, X_test, y_test, 10)


def testModel():
    model = Net_model()
    dense = getDense()
    dense.set_weights(model.layers[-1].get_weights())
    logger.info(model.to_json())
    x = numpy.random.random(size=(200, seq_len, 8))
    y = numpy.ones((200, ))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(x, y, batch_size=20, epochs=4)


if __name__ == "__main__":
    train()
    # testModel()
    pass
