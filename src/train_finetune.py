# -*- coding: utf-8 -*-
import datetime
import os
import sys

sys.path.append(os.path.abspath(".."))

import pickle as pkl
import pathlib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback

from src.model import Net_model, getEmbModel, getDense
import train_set
from src.utils.bsmote import borderline_smote
import tensorflow as tf
import numpy as np
from loguru import logger

dir_weights = '../weights/'
dir_logs = '../logs/'
dir_models = "../models"
lr = 5e-5
batchsize = 5000


def Net_training_finetune(
    X_train,
    y_train,
    X_test,
    y_test,
    EPOCHS,
    is_reg=False,
):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    pathlib.Path(dir_logs).mkdir(exist_ok=True)
    pathlib.Path(dir_weights).mkdir(exist_ok=True)

    model = Net_model()
    model_name = model.name

    adam_opt = Adam(learning_rate=lr)
    loss_func = None
    if not is_reg:
        loss_func = train_set.weighted_bce
    else:
        loss_func = 'mse'
    model.compile(loss=loss_func, optimizer=adam_opt, metrics=['accuracy'])
    model.fit(
        X_train,
        y_train,
        batch_size=batchsize,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(X_test, y_test),
        callbacks=[
            TensorBoard(log_dir=f"{dir_logs}/{now}"),
            ModelCheckpoint(filepath=f"{dir_weights}/best.h5",
                            save_best_only=True,
                            monitor='val_accuracy',
                            save_weights_only=True,
                            verbose=0),
            ModelCheckpoint(
                filepath=
                f"{dir_weights}/best-{now}_{{epoch:03d}}-{{val_accuracy:.4f}}.h5",
                save_best_only=True,
                monitor='val_accuracy',
                save_weights_only=True,
                verbose=1,
            ),
            EarlyStopping(monitor='val_accuracy', patience=10),
        ],
    )
    # SAVE Model
    model_json = model.to_json()
    pathlib.Path(dir_models).mkdir(exist_ok=True)
    pathlib.Path(dir_weights).mkdir(exist_ok=True)
    with open(f"{dir_models}/{model_name}.json", "w") as f_model:
        f_model.write(model_json)
    with open(f"{dir_weights}/adam_weights.pkl", "wb") as f_optimize:
        pkl.dump(adam_opt.get_weights(), f_optimize)
    with open((f"{dir_weights}/adam_weights-{now}.pkl"), "wb") as f_optimize:
        pkl.dump(adam_opt.get_weights(), f_optimize)
    logger.info("saved optimizer")
    # 数据增强

    # embedding
    logger.info('embedding')
    embModel = getEmbModel()
    embModel.set_weights(model.layers[-2].get_weights())
    embModel.compile(run_eagerly=True)
    x_emb_train = embModel.predict(X_train, batch_size=batchsize)
    logger.info('X_train_emb')
    x_emb_test = embModel.predict(X_test, batch_size=batchsize)
    logger.info('X_test')
    embModel.save_weights((f"{dir_weights}/embModel-{now}.h5"))
    embModel.save_weights(f"{dir_weights}/embModel.h5")
    # y = tf.concat([y_train, y_test], axis=0)
    x_emb = tf.reshape(
        # tf.concat([x_emb_train, x_emb_test], axis=0), # no data augmentation for validation data
        x_emb_train,
        (len(y_train), model.emb_shape[0] * model.emb_shape[1]),
    )
    # logger.info('x_emb, ytrain', type(x_emb), type(y_train))
    # logger.info('x_emb, y_train', x_emb.shape, x_emb.ndim, y_train.shape)
    x_train, y_train = borderline_smote(
        x_emb,
        y_train,
        train_set.s / train_set.p,
    )
    x_train = np.reshape(
        x_train, (len(y_train), model.emb_shape[0], model.emb_shape[1]))
    logger.info(x_train.shape)

    # dense
    dense = getDense(shape=(model.emb_shape[0], model.emb_shape[1]))
    dense.set_weights(model.layers[-1].get_weights())
    dense.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=lr),
                  metrics=['accuracy'])
    dense.fit(
        x_train,
        y_train,
        batch_size=batchsize,
        # : EPOCH
        epochs=3 + EPOCHS // 2,
        shuffle=True,
        validation_data=(x_emb_test, y_test),
        callbacks=[
            TensorBoard(log_dir=f"{dir_logs}/{now}_dense"),
            ModelCheckpoint(
                filepath=f"{dir_weights}/dense-best.h5",
                save_best_only=True,
                monitor='val_accuracy',
                save_weights_only=True,
                verbose=0,
            ),
            ModelCheckpoint(
                filepath=
                f"{dir_weights}/dense-best-{now}_{{epoch:03d}}-{{val_accuracy:.4f}}.h5",
                save_best_only=True,
                monitor='val_accuracy',
                save_weights_only=True,
                verbose=1,
            ),
            EarlyStopping(monitor='val_accuracy', patience=10),
        ],
    )
    dense.save_weights(f"{dir_weights}/dense-{now}.h5")
    logger.info('saved dense weights')
    model.layers[-1].set_weights(dense.get_weights())
    with open(f"{dir_models}/{model_name}-{now}.json", "w") as f_model:
        f_model.write(model_json)
    model.save_weights(f"{dir_weights}/{model_name}.h5")
    model.save_weights(f"{dir_weights}/{model_name}-{now}.h5")
    logger.info('saved all net weights')
