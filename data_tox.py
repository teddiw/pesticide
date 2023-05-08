import random
import math
import os
import pickle
from collections import defaultdict, namedtuple
import string

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # turn off since we're using multiple threads for loading anyway

from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, LlamaTokenizer # , GPT2Tokenizer, GPT2Model
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from util import suppress_stdout
from constants import *

class Dataset:
    def __init__(self, data_file, seed, batch_size):
        print('loading data')
        random.seed(seed)
        self.batch_size = batch_size

        parameter_path = '/self/scr-sync/nlp/huggingface_hub_llms/llama-7b'
        self.tokenizer = LlamaTokenizer.from_pretrained(parameter_path)
        self.tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        self.gpt_pad_id = self.tokenizer.encode(PAD_TOKEN)[0] # TEDDI TODO change name from self.gpt_pad_id

        self.data = pd.read_csv(data_file)
        self.data = self.data[['comment_text', 'toxic']]        
        self.data = self.data.to_numpy()
        self.max_length = 0
        for i in range(self.data.shape[0]):
            encoded_X = self.tokenizer.encode(self.data[i, 0])
            if (len(encoded_X) > self.max_length):
                self.max_length = len(encoded_X)

        print('done loading '+data_file)
        print('Size:')
        print(len(self.data))

    def __getitem__(self, i):
        X = self.data[i, 0]
        X = self.tokenizer.encode(X)
        X = torch.tensor(X)
        length = torch.tensor(len(X))
        y = torch.tensor(self.data[i, 1])
        return X, y, length, self.gpt_pad_id, self.max_length

    def __len__(self):
        return self.data.shape[0]
