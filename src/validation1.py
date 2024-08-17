# 1: with DNA/RNA bulges

import os
import sys

sys.path.append(os.path.abspath(".."))
from src import config

config.seq_len = 24
from loguru import logger
from src.train_finetune import *

logger.add(f"{dir_logs}/validation - 1 _{{time}}.log")
# CUDA_VISIBLE_DEVICES=2 DEVICE=GPU:0 python validation1.py
from sklearn import metrics
from tensorflow.python.keras.saving.model_config import model_from_json
from src.model import *
from utils.LoadDataset import *

from src.train_set import load_train_set
from sklearn.model_selection import train_test_split
from time import time
from utils.monitor import *

dir_result = f'../result/{datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")}'

time_start = time.time()
Net_model(attn_cnt=4, use_lstm=True)
x, y = load_train_set('I-1')
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=1 / 7)
logger.warning(f'[preprocess]{time.time()-time_start}')
# Train
monitor(Net_training_finetune, 'train', X_train, y_train, X_val, y_val, 10)

# Test set
X_test, y_test = getEmbeddingAndLabel('I-2')
X_test = X_test.reshape((len(X_test), config.seq_len, 8))


# Evaluate
def evaluate(model_file: str, weights_file: str):
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    logger.info(loaded_model.summary())
    loaded_model.load_weights(weights_file)
    logger.info('predicting')
    y_pred = loaded_model.predict(X_test, batch_size=5000).flatten()
    prc, rec, thr = metrics.precision_recall_curve(y_test, y_pred)
    prc[tuple([rec == 0])] = 1.0
    prc_auc = metrics.auc(rec, prc)

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    #TODO result supl
    pathlib.Path(dir_result).mkdir(exist_ok=True)
    with open(f'{dir_result}/{loaded_model.name}-{prc_auc}.json', 'w') as f:
        f.write(loaded_model_json)
    loaded_model.save_weights(f'{dir_result}/{loaded_model.name}-{prc_auc}.h5')
    pandas.DataFrame(data={
        'precision': prc,
        'recall': rec,
    }).to_csv(f'{dir_result}/{loaded_model.name}-{prc_auc}-pr.csv')
    pandas.DataFrame(data={
        'fpr': fpr,
        'tpr': tpr,
    }).to_csv(f'{dir_result}/{loaded_model.name}-{prc_auc}-fprtpr.csv')
    pandas.DataFrame(data={
        'prediction': y_pred,
        'label': y_test,
    }).to_csv(f'{dir_result}/{loaded_model.name}-{prc_auc}-pred.csv')


monitor(
    evaluate,
    'test',
    f'{dir_models}/{model_name}.json',
    f'{dir_weights}{model_name}.h5',
)
