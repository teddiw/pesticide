import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from transformers import set_seed, GPT2Tokenizer, GPT2Model, GPT2Config

class GPT2Tox(nn.Module):
    def __init__(self):
        super(GPT2Tox, self).__init__()
        config = GPT2Config()
        self.model = GPT2Model(config)
        self.model_output_dim = (1024, 768)
        self.clf = nn.Linear(768, 1) 

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        clf_outputs = self.clf(outputs.last_hidden_state)
        return clf_outputs
       
    def predict(self, token_inputs):
        print('token_inputs.shape', token_inputs.shape)
        input_len = token_inputs.shape[1]
        print('input_len', input_len)
        max_length = 1024
        attention_mask = torch.zeros([1, max_length]).to(torch.float32)
        if input_len < max_length:
            token_inputs = torch.cat([token_inputs.to('cpu'), torch.unsqueeze(torch.zeros(max_length - input_len).long(), dim=0)], dim=1) # pad with zeros if <1024
            print('token_inputs.shape', token_inputs.shape)
        elif input_len >= max_length:
            print('Sequence too long; sequence over 1024 tokens')
            return
        attention_mask[0][:input_len] = 1
        input_ids = torch.unsqueeze(token_inputs.to(torch.int32), 0)
        self.eval()
        with torch.no_grad():
            # scores shape: torch.Size([1, batch_size, 1024, 1])
            scores = torch.sigmoid(self.forward(input_ids.to('cuda'), attention_mask.to('cuda')))
            #     attention_mask = torch.zeros([len(batch), max_length]).to(torch.float32)
            # input_length = np.sum(attention_mask, axis=1)-1
            print('scores', scores)
            prob = scores[0][0][input_len][0]
            print('prob.item()', prob.item())
        return prob.item()
