from argparse import ArgumentParser
import math
import string

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import pipeline, set_seed, LlamaForCausalLM, LlamaTokenizer
from detoxify import Detoxify
from my_constants import *

def get_P_of_o_given_x(line, base_model, base_tokenizer, device='cpu'):
    encoded_input = [base_tokenizer.encode(it, return_tensors='pt').to(device) for it in [line]]
    encoded_input = encoded_input[0]
    encoded_input = encoded_input[:, :-1]
    print('encoded_input', encoded_input[0])
    for i in range(3, len(encoded_input[0])):
        curr_encoded_input = encoded_input[0][:i]
        print('curr_encoded_input at '+str(i), curr_encoded_input)
        curr_encoded_input = torch.cat([curr_encoded_input], dim=0).to(device)
        next_encoded_token = encoded_input[0][i]
        print('next_encoded_token', next_encoded_token)
        model_logits = base_model(encoded_input)[0][:, -1, :]
        print('model_logits.shape', model_logits.shape) # how to get the logit corresponding to next_encoded_token?
        print('model_logits', model_logits)
        return

def main(args):
    f = open(args.result_file, 'r')
    lines = f.readlines()
    f.close()
    # lines = pd.read_csv(args.result_file, sep="<s>", header=None)
    # lines = lines.to_numpy()
    print('LINES', lines)

    parameter_path = '/self/scr-sync/nlp/huggingface_hub_llms/llama-7b'
    tokenizer = LlamaTokenizer.from_pretrained(parameter_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    print('here 1')
    # base_model = LlamaForCausalLM.from_pretrained(parameter_path).half().to('cpu')
    print('here 2')
    # base_model.eval()
    
    toxicity = np.zeros(len(lines))
    fluency = np.zeros(len(lines))
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
        toxicity[i] = Detoxify('original').predict(lines[i])['toxicity']
        fluency[i] = 0
    df = pd.DataFrame({'result': lines, 'toxicity': toxicity, 'fluency': fluency})
    print(df)
    df.to_csv('result_metrics.csv')
    
    # get_P_of_o_given_x(lines[0], base_model, tokenizer)
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--result_file", type=str)
    args = parser.parse_args()

    main(args)
