import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
import string
from collections import defaultdict
import threading

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2Model, AutoTokenizer, AutoModelWithLMHead, LlamaForCausalLM, LlamaTokenizer
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params
from constants import *
from detoxify import Detoxify

def main(args):
    if (args.model_string == "gpt2-medium"):
        tokenizer = AutoTokenizer.from_pretrained(args.model_string)
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        pad_id = tokenizer.encode(PAD_TOKEN)[0]
        print('Loaded '+args.model_string+' tokenizer...')
        model = AutoModelWithLMHead.from_pretrained(args.model_string).to(args.device)
        model.eval()
        print('Loaded '+args.model_string+' model...')
    elif (args.model_string == 'llama'):
        parameter_path = '/self/scr-sync/nlp/huggingface_hub_llms/llama-7b'
        tokenizer = LlamaTokenizer.from_pretrained(parameter_path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # use PAD_TOKEN instead of '[PAD]'
        pad_id = tokenizer.encode('[PAD]')[0] # use PAD_TOKEN instead of '[PAD]'
        print('Loaded '+args.model_string+' tokenizer...')
        model = LlamaForCausalLM.from_pretrained(parameter_path).half().to(args.device)
        print('Loaded '+args.model_string+' model...')
        model.eval()
    else:
        print('Model '+args.model_string+' not implemented')

    file_lock = threading.Lock()
    file_lock.acquire()
    with open(args.prefix_file, 'r') as rf:
        lines = rf.readlines()
    file_lock.release()
    os.makedirs('toxicity_results', exist_ok=True)
    output_file = 'toxicity_results/'+args.tag+'_top'+str(args.precondition_topk)+'_ntokens'+str(args.num_tokens)+'_toxicity_output.txt'
    with open(output_file, 'a') as f: 
        for line in tqdm(lines, total=len(lines)):
            decoded_result = predict_toxic_completion(model, tokenizer, None, line, args.precondition_topk, args.device, args.num_tokens, args.condition_lambda)    
            file_lock.acquire()
            f.write('\n')
            f.write(''.join(decoded_result))
            file_lock.release()
            print(decoded_result)
    return 

def predict_toxic_completion(base_model, base_tokenizer, toxicity_clf, prefix, precondition_topk, device, num_tokens, condition_lambda):
    encoded_input = [base_tokenizer.encode(it, return_tensors='pt').to(device) for it in [prefix]]
    encoded_input = encoded_input[0]
    encoded_input = encoded_input[:, :-1]
    encoded_input = torch.cat([encoded_input], dim=0).to(args.device)
    for _ in range(num_tokens):
        model_logits = base_model(encoded_input)[0][:, -1, :] # batch x vocab
        top_logits, top_indices = model_logits.topk(precondition_topk, dim=1)
        top_token_ls = [base_tokenizer.decode(idx) for idx in top_indices[0]]
        # print('top tokens are:', top_token_ls)
        reweighted_logp_terms = np.array([np.zeros(precondition_topk)])
        for i in range(precondition_topk):
            curr_encoded_input = torch.cat([encoded_input, torch.tensor([[top_indices[0][i]]]).to(args.device)], dim=1)
            curr_decoded_input = [base_tokenizer.decode(encoded_input_token) for encoded_input_token in curr_encoded_input][0]
            results = Detoxify('original').predict(curr_decoded_input)
            # token_p = 1/(1 + np.exp(-top_logits[0][i]))
            # reweighted_logp_terms[0][i] = np.log(results['toxicity']) + np.log(token_p)
            toxicity_p = results['toxicity']
            toxicity_logit = np.log(toxicity_p/(1-toxicity_p))
            top_logits[0][i] += condition_lambda*toxicity_logit
        
        post_probs = F.softmax(top_logits, dim=1)
        # post_probs = F.softmax(reweighted_logp_terms, dim=1)
        batch_size = 1 
        next_indices = top_indices[torch.arange(batch_size).to(top_indices.device), torch.multinomial(post_probs, 1).flatten()] # batch
        encoded_input = torch.cat([encoded_input, next_indices.unsqueeze(1)], dim=1) # batch x seq+1
        
    decoded_result = [base_tokenizer.decode(encoded_input_token) for encoded_input_token in encoded_input]
    return decoded_result

if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--prefix_file', type=str, default='/nlp/u/worledge/fudge/toxicity_data/toxicity_prefixes.txt', required=False, help='file of prefix lines for toxicity generation')
    parser.add_argument('--model_string', type=str, default='gpt2-medium', required=False, help='base model and tokenizer')
    parser.add_argument('--precondition_topk', type=int, default=30, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--num_tokens', type=int, default=30, help='number of tokens to generate')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--tag', type=str, default='', help='tag for output file name')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
