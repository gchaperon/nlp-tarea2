import torch
import argparse
import random
import numpy as np
import pprint

import logging

from transformers import (
    BertForTokenClassification,
    BertTokenizer,
    BertConfig,
)

logger = logging.getLogger(__name__)


LABEL_LIST = ('B-PER', 'B-MISC', 'I-ORG', 'B-ORG',
              'I-LOC', 'B-LOC', 'I-PER', 'O', 'I-MISC')

LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class InputExample():
    """
    This class should contain words already tokenized to assure
    the max length is respected
    """
    def __init__(self, tokens_a, labels_a, tokens_b, labels_b):
        self.tokens_a = tokens_a
        self.labels_a = labels_a
        self.tokens_b = tokens_b
        self.labels_b = labels_a





def main(passed_args=None):
    # TODO: cambiar defaults
    # TODO: agregar requireds
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="conll2002", type=str)
    parser.add_argument("--model", default="bert-base-multilingual-cased", type=str)
    parser.add_argument("--output-dir", default="outputs", type=str)

    # Hyperparams to perform search
    parser.add_argument("--learn-rate", default=5e-5, type=float)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=3, type=int)

    # General options
    parser.add_argument("--do-lower-case", action="store_true")
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args(passed_args)

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0

    pprint.pprint(vars(args))


    set_seed(args)
    breakpoint()



if __name__ == '__main__':
    main()
