# -*- coding: utf-8 -*-
import os
import pickle
from typing import List

import pandas
import numpy as np
from tqdm import tqdm
from loguru import logger
from src import config


def getDataFrame(name: str, use_cache: bool = True):
    assert os.path.exists('../data')
    if not use_cache or not os.path.exists('../data/%s.pkl' % name):
        dataset = pandas.read_csv('../data/%s.csv' % name, delimiter=',')
        dataset.to_pickle(path='../data/%s.pkl' % name)
    else:
        logger.info('load %s data from pickle' % name)
        dataset = pandas.read_pickle('../data/%s.pkl' % name)
    return dataset


def emb(b: str):
    emb_map = {
        'A': [1, 0, 0, 0, 0],
        'T': [0, 1, 0, 0, 0],
        'G': [0, 0, 1, 0, 0],
        'C': [0, 0, 0, 1, 0],
        # 'N': [0.25, 0.25, 0.25, 0.25, 0],
        '_': [0, 0, 0, 0, 1],
        '-': [0, 0, 0, 0, 0]
    }
    # N :the same base at off seq
    # II-3,5,6
    return emb_map[b.upper()]


def encode_10bit(on_emb, off_emb):
    emb_10bit = []
    for j in tqdm(range(len(on_emb))):
        on = on_emb[j]
        off = off_emb[j]
        on_off_dim10_codes = []
        for i in range(config.seq_len):
            on_off_dim10_codes.append(np.concatenate((on[i], off[i])))
        emb_10bit.append(np.array(on_off_dim10_codes))
    return np.array(emb_10bit)


def encode_8bit(on_bases, off_bases, on_emb, off_emb):
    emb_8bit = []
    direction_dict = {'A': 5, 'G': 4, 'C': 3, 'T': 2, '_': 1}
    for j in tqdm(range(len(on_bases))):
        on_seq = on_bases[j]
        off_seq = off_bases[j]
        on = on_emb[j]
        off = off_emb[j]
        on_off_dim8_codes = []
        for i in range(len(on_seq)):
            diff_code = np.bitwise_or(on[i], off[i])
            on_b = on_seq[i]
            off_b = off_seq[i]
            if on_b == "N":
                on_b = off_b
            dir_code = np.zeros(3)
            if (on_b == "-" or off_b == "-"
                    or direction_dict[on_b] == direction_dict[off_b]):
                dir_code[2] = 1
            else:
                # direction src 的决定
                if direction_dict[on_b] > direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            on_off_dim8_codes.append(np.concatenate((diff_code, dir_code)))
        emb_8bit.append(np.array(on_off_dim8_codes))
    return np.array(emb_8bit)


def encode_7bit(on_bases, off_bases, on_emb, off_emb):
    emb_7bit = []
    direction_dict = {'A': 5, 'G': 4, 'C': 3, 'T': 2, '_': 1}
    for j in tqdm(range(len(on_bases))):
        on_seq = on_bases[j]
        off_seq = off_bases[j]
        on = on_emb[j]
        off = off_emb[j]
        on_off_dim7_codes = []
        for i in range(len(on_seq)):
            diff_code = np.bitwise_or(on[i], off[i])
            on_b = on_seq[i]
            off_b = off_seq[i]
            if on_b == "N":
                on_b = off_b
            dir_code = np.zeros(2)
            if (on_b == "-" or off_b == "-"
                    or direction_dict[on_b] == direction_dict[off_b]):
                pass
            else:
                # direction src 的决定
                if direction_dict[on_b] > direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))
        emb_7bit.append(np.array(on_off_dim7_codes))
    return np.array(emb_7bit)


def getEmbeddingAndLabelMulti(names: List[str], use_cache: bool = True):
    dss = [getEmbeddingAndLabel(name, use_cache=use_cache) for name in names]
    return np.concatenate([ds[0] for ds in dss
                           ]), np.concatenate([ds[1] for ds in dss])


def getEmbeddingAndLabel(name: str, use_cache: bool = True):
    """
	provide DataFrame with 3 columns : `on`, `off`, `val`
	:param name: dataset_name to load
	:param use_cache: use cached pickle file, this would accelerate the loading
	        but may cause other problem
	:return: DataFrame
	"""
    if not use_cache or not os.path.exists('../data/%s-8b.pkl' % name):
        raw = getDataFrame(name, use_cache=use_cache)
        on = raw['on']
        off = raw['off']
        val = raw['val']
        on = [('-' * (config.seq_len - len(seg)) + seg).upper() for seg in on]
        off = [('-' * (config.seq_len - len(seg)) + seg).upper()
               for seg in off]
        on_emb = [[emb(b) for b in seg] for seg in on]
        off_emb = [[emb(b) for b in seg] for seg in off]
        x = encode_8bit(on, off, np.array(on_emb), np.array(off_emb))
        y = np.array(val)
        pickle.dump([x, y], open('../data/%s-8b.pkl' % name, 'wb'))
    else:
        logger.info('load %s data from pickle' % name)
        if name == 'crispor':
            [x, y] = pickle.load(
                open('../data/%s-8b.pkl' % name, 'rb'),
                encoding='iso-8859-1',
            )
            # x = x.reshape((x.shape[0], 1, x.shape[1], 8))
        else:
            x, y = pickle.load(open('../data/%s-8b.pkl' % name, 'rb'))
    logger.info((x.shape, y.shape))
    x = x.reshape((len(x), 1, config.seq_len, 8))
    return x, y


def getEmbeddingAndLabel_10b(name: str, use_cache: bool = True):
    """
	provide DataFrame with 3 columns : `on`, `off`, `val`
	:param name: dataset_name to load
	:param use_cache: use cached pickle file, this would accelerate the loading
	        but may cause other problem
	:return: DataFrame
	"""
    if not use_cache:
        raw = getDataFrame(name, use_cache=use_cache)
        on = raw['on']
        off = raw['off']
        val = raw['val']
        on = [('-' * (config.seq_len - len(seg)) + seg).upper() for seg in on]
        off = [('-' * (config.seq_len - len(seg)) + seg).upper()
               for seg in off]
        on_emb = [[emb(b) for b in seg] for seg in on]
        off_emb = [[emb(b) for b in seg] for seg in off]
        x = encode_10bit(np.array(on_emb), np.array(off_emb))
        y = np.array(val)
        pickle.dump([x, y], open('../data/%s-10b.pkl' % name, 'wb'))
    else:
        logger.info('load %s data from pickle' % name)
        x, y = pickle.load(open('../data/%s-10b.pkl' % name, 'rb'))
    x = x.reshape((len(x), 1, config.seq_len, 10))
    return x, y


def getEmbeddingAndLabel_7b(name: str, use_cache: bool = True):
    """
	provide DataFrame with 3 columns : `on`, `off`, `val`
	:param name: dataset_name to load
	:param use_cache: use cached pickle file, this would accelerate the loading
	        but may cause other problem
	:return: DataFrame
	"""
    if not use_cache:
        raw = getDataFrame(name, use_cache=use_cache)
        on = raw['on']
        off = raw['off']
        val = raw['val']
        on = [('-' * (config.seq_len - len(seg)) + seg).upper() for seg in on]
        off = [('-' * (config.seq_len - len(seg)) + seg).upper()
               for seg in off]
        on_emb = [[emb(b) for b in seg] for seg in on]
        off_emb = [[emb(b) for b in seg] for seg in off]
        x = encode_7bit(on, off, np.array(on_emb), np.array(off_emb))
        y = np.array(val)
        pickle.dump([x, y], open('../data/%s-7b.pkl' % name, 'wb'))
    else:
        logger.info('load %s data from pickle' % name)
        x, y = pickle.load(open('../data/%s-7b.pkl' % name, 'rb'))
    x = x.reshape((len(x), 1, config.seq_len, 7))
    return x, y


def testload():
    # getEmbeddingAndLabel_7b('I-1', use_cache=False)
    # getEmbeddingAndLabel_10b('I-1', use_cache=False)
    # getEmbeddingAndLabel('II-1-23', use_cache=False)
    # getEmbeddingAndLabel('crispor', use_cache=False)
    # getEmbeddingAndLabel('II-1-23')
    # getEmbeddingAndLabel('deep-train', use_cache=False)
    # getEmbeddingAndLabel('deep-test', use_cache=False)
    # getEmbeddingAndLabel('dc-test', use_cache=False)
    # getEmbeddingAndLabel_7b('I-2', use_cache=False)
    # getEmbeddingAndLabel_10b('I-2', use_cache=False)
    getEmbeddingAndLabel('I-1', use_cache=False)
    getEmbeddingAndLabel('I-2', use_cache=False)
    # x,y = getEmbeddingAndLabel('crsql', use_cache=False)


# 	x,y = getEmbeddingAndLabel('crisprsql-reg', use_cache=False)
# logger.info(x[1],y[1])

# return getEmbeddingAndLabel('II-2', use_cache=False)

if __name__ == '__main__':
    data = testload()
