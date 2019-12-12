# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os

import torch
import random
import numpy as np

from tqdm import tqdm
# from torch.

"""
I will heavily rely on the features of the library transformers by hugginface
"""

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# I added conll2002 data to this project, let's define some constants related to paths and stuff
# I will give names to the constants to reflect the names in the course assignment, sadly
# the names of the original files are a bit different

# DATA_DIR = os.path.join("kaggle", "input", "conll-corpora", "conll2002")
# TRAIN_FILENAME = "esp.train"
# DEV_FILENAME = "esp.testa"
# TEST_FILENAME = "esp.testb"


DATA_DIR = "conll2002-data"
TRAIN_FILENAME = "esp.train"
DEV_FILENAME = "esp.testa"
TEST_FILENAME = "esp.testb"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


class Example():
    def __init__(self, words, labels=None):
        self.words = words
        self.labels = labels


def read_examples(path, w_labels=True):
    """
    Esta funcion lee el archivo y retorna un lista de ejemplos.
    Para esta tarea no nos interesan los POS
    """
    examples = []
    with open(path, "r") as file:
        words = []
        labels = []
        for line in file:
            if line == "\n":
                examples.append(Example(words, labels if w_labels else None))
                words = []
                labels = []
                continue
            word, _, label = line.split()
            words.append(word)
            labels.append(label)
    return examples


def examples2features(examples, tokenizer, label_list, max_length=512):
    """
    Voy a usar mBERT, asique asique necesito el tokenizador de BERT pa
    poder convertir los ejemplos en vectores
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in enumerate(tqdm(examples)):

        tokens = []
        labels = []
        for word, label in zip(example.words, example.labels):
            ...


def main(passed_args=None):
    examples = read_examples(os.path.join(DATA_DIR, TRAIN_FILENAME))
    breakpoint()


if __name__ == '__main__':
    main()
