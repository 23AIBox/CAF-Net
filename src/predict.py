'''
score for predicting single guide and single potential off-target site

guide_seq         :      reference target sequence, also decided as guide sequence
off_seq           :      one potential off_target sequence that close to true target
'''
import argparse
import yaml
import sys
import re
import os

sys.path.append(os.path.abspath(".."))
from src import config

config.seq_len = 24
# CUDA_VISIBLE_DEVICES=2 DEVICE=GPU:0 python predict.py --guide_seq ACCGAGGAGTGAGTAGTCCTGGG --off_seq TCCGAGGAGTGAGTAGTCCTGGG
from tensorflow.python.keras.saving.model_config import model_from_json
from utils.LoadDataset import *


def load_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        sys.exit(1)
    except yaml.YAMLError:
        sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predictor for single guide and off-target pair")
    parser.add_argument(
        '--guide_seq',
        type=str,
        help='reference target sequence, also decided as guide sequence')
    parser.add_argument(
        '--off_seq',
        type=str,
        help='one potential off_target sequence that close to true target')

    args = parser.parse_args()
    return args


def main():

    args = parse_arguments()
    conf = load_config('./config.yaml')
    on = ('-' * (config.seq_len - len(args.guide_seq)) +
          args.guide_seq).upper()
    off = ('-' * (config.seq_len - len(args.off_seq)) + args.off_seq).upper()

    if bool(re.match(r'^[ATGCN_-]*$', on)) and bool(
            re.match(r'^[ATGCN_-]*$', off)):
        loaded_model = model_from_json(open(conf.get('model'), 'r').read())
        print(loaded_model.summary())
        loaded_model.load_weights(conf.get('weight'))
        on_emb = [emb(b) for b in on]
        off_emb = [emb(b) for b in off]
        direction_dict = {'A': 5, 'G': 4, 'C': 3, 'T': 2, '_': 1}
        on_off_dim8_codes = []
        for i in range(len(on)):
            diff_code = np.bitwise_or(on_emb[i], off_emb[i])
            on_b = on[i]
            off_b = off[i]
            if on_b == "N":
                on_b = off_b
            dir_code = np.zeros(3)
            if (on_b == "-" or off_b == "-"
                    or direction_dict[on_b] == direction_dict[off_b]):
                dir_code[2] = 1
            else:
                if direction_dict[on_b] > direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            on_off_dim8_codes.append(np.concatenate((diff_code, dir_code)))
        x = np.array([on_off_dim8_codes])
        x = x.reshape((len(x), config.seq_len, 8))
        print(np.round(loaded_model.predict(x).item(), 6))
    else:
        print('please check input, sequence should contains only "ATGCN_-"')


if __name__ == "__main__":
    main()
