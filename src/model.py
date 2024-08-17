# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.abspath(".."))
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LSTM, Bidirectional, BatchNormalization
from src.utils.attention_module import *
from src import config
import gc
from loguru import logger


def conv1d_channel(
    x,
    filters,
    kernel_size,
    strides=1,
    padding="same",
    activation="relu",
    use_bias=True,
    name=None,
    trainable=True,
):
    x = layers.Conv1D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=name,
        trainable=trainable,
    )(x)
    if activation is not None:
        x = layers.Activation(
            activation,
            name=(name + "_activation" if name else None),
        )(x)
    return x


embModel = None


def getEmbModel(
    enc=8,
    conv_cnt=1,
    conv_config=None,
    attn_cnt=6,
    num_heads=4,
    key_size=24,
    use_lstm=False,
    lstm_units=16,
    concat=False,
):
    global embModel
    if embModel:
        return embModel
    inputs = Input(shape=(config.seq_len, enc), name="input")
    # convolution layer
    mixed = inputs
    for _ in range(conv_cnt):
        if not conv_config:
            mixed = layers.Concatenate(axis=-1)([
                inputs,
                conv1d_channel(mixed, 4, 1),
                conv1d_channel(mixed, 4, 2),
                conv1d_channel(mixed, 4, 3),
                conv1d_channel(mixed, 4, 5),
            ])
        else:
            tmp = [inputs]
            for x in conv_config:
                tmp.append(conv1d_channel(mixed, x['filters'],
                                          x['kernel']))  #注意不能用迭代器的写法
            mixed = layers.Concatenate(axis=-1)(tmp)
    # attention
    out = mixed
    for i in range(attn_cnt):
        out = TransformerBlock(
            channels=key_size,
            dropout_rate=0.05,
            attention_kwargs=dict(
                value_size=key_size // num_heads,
                key_size=key_size,
                num_heads=num_heads,
                relative_position_symmetric=False,
                num_relative_position_features=None,
                relative_position_functions=[
                    'positional_features_exponential',
                    'positional_features_central_mask',
                    'positional_features_gamma',
                ],
                positional_dropout_rate=0.01,
                attention_dropout_rate=0.05,
                name='L%d' % i,
            ),
        )(out, is_training=True)

    out = BatchNormalization()(out)
    # bi-LSTM
    if use_lstm:
        conv_out_shape = (config.seq_len, 24)
        out = Reshape(conv_out_shape)(out)
        out = Bidirectional(
            LSTM(lstm_units,
                 return_sequences=True,
                 input_shape=conv_out_shape,
                 name="LSTM_out"))(out)
    # blstm_out = Bidirectional(
    # 	GRU(16, return_sequences=True),
    # 	backward_layer=GRU(16, go_backwards=True, return_sequences=True))(mixed)
    # out = Flatten()(mixed)

    # concat all hidden output
    if concat:
        out = layers.Concatenate(axis=-1)([inputs, mixed, out])
    embModel = Model(inputs, out, name='emb')
    logger.info(embModel.summary())
    return embModel


denseModel = None


def getDense(shape=(config.seq_len, 24)):
    global denseModel
    if denseModel:
        return denseModel
    # dense layer
    inputs = Input(shape=shape)
    x = inputs
    x = Flatten()(x)
    x = Dense(
        72,
        activation=relu,
        kernel_regularizer=regularizers.l2(),
    )(x)
    x = Dense(config.seq_len, activation=relu)(x)
    x = layers.Dropout(rate=0.4)(x)
    prediction = Dense(1, activation="sigmoid", name="dense_output")(x)
    dense = Model(inputs, prediction, name='dense')
    logger.info(dense.summary())
    return dense


net = None
model_name = 'cafnet'


def Net_model(
    enc=8,
    conv_cnt=1,
    conv_config=None,
    attn_cnt=4,
    num_heads=4,
    key_size=24,
    use_lstm=False,
    lstm_units=16,
    concat=False,
    name=model_name,
):
    global net
    if net is not None:
        return net
    emb = getEmbModel(
        enc=enc,
        conv_cnt=conv_cnt,
        conv_config=conv_config,
        attn_cnt=attn_cnt,
        num_heads=num_heads,
        key_size=key_size,
        use_lstm=use_lstm,
        lstm_units=lstm_units,
        concat=concat,
    )
    dim = 0
    if use_lstm:
        dim = (lstm_units * 2) if (not concat) else (lstm_units * 2 + 24 + enc)
    else:
        dim = 24 if (not concat) else (24 + enc + 24)
    dense = getDense(shape=(config.seq_len, dim))
    inputs = Input(shape=(config.seq_len, enc), name='input_net')
    x = emb(inputs)
    prediction = dense(x)
    net = Model(inputs, prediction, name=name)
    net.emb_shape = (config.seq_len, dim)
    logger.info(net.summary())
    logger.info(net.emb_shape)
    return net


def get_reset(
    enc=8,
    conv_cnt=1,
    conv_config=None,
    attn_cnt=4,
    num_heads=4,
    key_size=24,
    use_lstm=False,
    lstm_units=16,
    concat=False,
    name=model_name,
):
    global net
    global embModel
    global denseModel
    sess = K.get_session()
    K.clear_session()
    sess.close()
    sess = K.get_session()
    # 	cuda.select_device(0)
    # 	cuda.close()
    del net
    del embModel
    del denseModel
    net = None
    embModel = None
    denseModel = None
    logger.info(gc.collect())
    net = Net_model(
        enc=enc,
        conv_cnt=conv_cnt,
        conv_config=conv_config,
        attn_cnt=attn_cnt,
        num_heads=num_heads,
        key_size=key_size,
        use_lstm=use_lstm,
        lstm_units=lstm_units,
        concat=concat,
        name=name,
    )
    return net
