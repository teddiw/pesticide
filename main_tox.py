import os
import random
import time
import pickle
import math
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from data_tox import Dataset
from model_tox import ToxicityModel
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params, pad_mask
from constants import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def my_collate(batch):
    # X, y, length, self.gpt_pad_id
    pad_id = batch[0][3]
    inputs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch]).to(torch.float32)
    lengths = torch.LongTensor([b[2] for b in batch])
    max_length = batch[0][4] # HIDDEN_DIM
    for i in range(len(inputs)):
        if len(inputs[i]) < max_length:
            inputs[i] = torch.cat([inputs[i], torch.zeros(max_length - len(inputs[i])).long()+pad_id], dim=0) # actually 0 is fine as pad since it's masked out
    
    # inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
    inputs = torch.unsqueeze(torch.stack(inputs, dim=0).to(torch.float32), 0)
    # inputs = pack_padded_sequence(inputs, lengths.cpu(), enforce_sorted=False)
    # inputs = torch.tensor(inputs)
    return inputs, labels, lengths

def train(model, train_dataset, optimizer, criterion, epoch, args):
    model.train()
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)
    loss_meter = AverageMeter('loss', ':6.4f')
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter], prefix='Training: ')
    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        inputs, labels, lengths = batch 
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        scores = model(inputs, lengths, run_classifier=True)
        loss = criterion(scores.flatten(), labels.flatten().float())
        optimizer.zero_grad() # can also be model.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.detach(), len(labels))
        if batch_num % args.train_print_freq == 0:
            progress.display(batch_num)
    progress.display(total_length)
    return loss_meter.avg


def validate(model, val_dataset, criterion, epoch, args):
    model.eval()
    random.seed(0)
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)
    loss_meter = AverageMeter('loss', ':6.4f')
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter], prefix='Validation: ')
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
            inputs, labels, lengths = batch 
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            scores = model(inputs, lengths, run_classifier=True)
            loss = criterion(scores.flatten(), labels.flatten().float())
            loss_meter.update(loss.detach(), len(labels))
            if batch_num % args.train_print_freq == 0:
                progress.display(batch_num)
    progress.display(total_length)
    return loss_meter.avg


def main(args):
    train_dataset = Dataset(args.train_data_file, args.seed, args.batch_size) 
    val_dataset = Dataset(args.val_data_file, args.seed, args.batch_size)

    os.makedirs(args.save_dir, exist_ok=True)
    model = ToxicityModel(args, train_dataset.gpt_pad_id)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('num params', num_params(model))
    criterion = nn.BCEWithLogitsLoss().to(args.device)
    best_val_metric = 1e8
    for epoch in range(args.epochs):
        print("Training: Epoch {} at {}".format(epoch, time.ctime()))
        train_loss = train(model, train_dataset, optimizer, criterion, epoch, args)
        print("TRAINING Loss is:", train_loss)
        if epoch % args.validation_freq == 0:
            print("Validation: Epoch {} at {}".format(epoch, time.ctime()))
            val_loss = validate(model, val_dataset, criterion, epoch, args)
            print("VALIDATION Loss is:", val_loss)
            if val_loss < best_val_metric:
                print('!!!! new best val metric', val_loss)
                best_val_metric = val_loss
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_metric': best_val_metric,
                    'optimizer': optimizer.state_dict(),
                    'args': args
                }, os.path.join(args.save_dir, 'model_best.pth.tar'))
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_metric': val_loss,
                'optimizer': optimizer.state_dict(),
                'args': args
            }, os.path.join(args.save_dir, 'model_epoch' + str(epoch) + '.pth.tar'))

if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--train_data_file', type=str, required=False)
    parser.add_argument('--val_data_file', type=str, required=False)
    parser.add_argument('--test_data_file', type=str, required=False)

    # SAVE/LOAD
    parser.add_argument('--save_dir', type=str, required=True, help='where to save ckpts')
    parser.add_argument('--ckpt', type=str, default=None, help='load ckpt from file if given')

    # TRAINING
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epoch_max_len', type=int, default=None, help='max batches per epoch if set, for more frequent validation')
    parser.add_argument('--validation_freq', type=int, default=1, help='validate every X epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--num_workers', type=int, default=20, help='num workers for data loader')
    parser.add_argument('--evaluate', action='store_true', default=False)

    # PRINTING
    parser.add_argument('--train_print_freq', type=int, default=100, help='how often to print metrics (every X batches)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.evaluate:
        assert args.ckpt is not None
    
    main(args)
