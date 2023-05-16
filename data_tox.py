import random
import math
import os
import pickle
from collections import defaultdict, namedtuple
import string

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # turn off since we're using multiple threads for loading anyway

from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from util import suppress_stdout
from constants import *

def prep_one(comment):
    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    self.gpt_pad_id = self.tokenizer.encode('[PAD]')[0]
    
    
class Dataset:
    def __init__(self, data_file, seed, batch_size, model_str):
        print('loading data')
        random.seed(seed)
        self.batch_size = batch_size


        # if (model_str == 'llama'):
        #     parameter_path = '/self/scr-sync/nlp/huggingface_hub_llms/llama-7b'
        #     self.tokenizer = LlamaTokenizer.from_pretrained(parameter_path)
        if (model_str == 'gpt2'):
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            print('tokenizer not implemented')

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.gpt_pad_id = self.tokenizer.encode('[PAD]')[0]

        self.data = pd.read_csv(data_file) 
        self.data = self.data[['comment_text', 'toxic']]
        self.data = self.data.to_numpy()
        # self.max_length = 1024
        # for i in range(self.data.shape[0]):
        #     encoded_X = self.tokenizer.encode(self.data[i, 0])
        #     if (len(encoded_X) > self.max_length):
        #         self.max_length = len(encoded_X)

        print('done loading')
        print('Size:')
        print(len(self.data))

    def __getitem__(self, i):
        X = self.data[i, 0]
        X = self.tokenizer.encode(X)
        X = torch.tensor(X)
        length = torch.tensor(len(X))
        y = torch.tensor(self.data[i, 1])
        return X, y, length 

    def __len__(self):
        return self.data.shape[0]
