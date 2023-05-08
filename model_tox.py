import math

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed # , GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2Config, GPT2ForSequenceClassification, GPT2LMHeadModel, MarianTokenizer

from my_constants import *
from util import pad_mask

class ToxicityModel(nn.Module):
    def __init__(self, args, gpt_pad_id, verbose=True):
        super(ToxicityModel, self).__init__()

        self.rnn = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=3, bidirectional=False, dropout=0) # want it to be causal so we can learn all positions
        self.out_linear = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, inputs, lengths, run_classifier=False):
        """
        inputs: token ids, batch x seq, right-padded with 0s
        lengths: lengths of inputs; batch
        future_words: batch x N words to check if not predict next token, else batch
        log_probs: N
        syllables_to_go: batch
        """
        rnn_output, _ = self.rnn(inputs)
        return self.out_linear(rnn_output).squeeze(2)
            
