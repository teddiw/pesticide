from argparse import ArgumentParser
from gpt2_model_tox import GPT2Tox
import torch.nn as nn
import torch
import wandb
from data_tox import Dataset
from tqdm import tqdm
import os
from util import save_checkpoint
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """ fn from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '65229'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    """ fn from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html """
    dist.destroy_process_group()

def custom_collate(batch):
    # X, y, length
    inputs = [b[0] for b in batch] # list of token ids for each comment in batch
    labels = torch.tensor([b[1] for b in batch]).to(torch.float32) # toxic/non-toxic label for each comment in batch
    lengths = torch.LongTensor([b[2] for b in batch]) # original number of token ids for each comment in batch
    max_length = 1024 # hidden dim of gpt2-medium
    attention_mask = torch.zeros([len(batch), max_length]).to(torch.float32)
    formatted_labels[i][:lengths[i]] = labels[i]
    inputs = torch.unsqueeze(torch.stack(inputs, dim=0).to(torch.int32), 0)
    return inputs, formatted_labels, attention_mask, lengths

def train(model, train_dataset, optimizer, rank, lr=1e-3, batch_size=128, num_workers=2):
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(rank)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)
    epoch_loss = 0
    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        inputs, labels, attention_mask, lengths = batch
        inputs = inputs.to(rank)
        labels = labels.to(rank)
        attention_mask = attention_mask.to(rank)

        scores = model(inputs, attention_mask)

        # Call criterion for each element in batch (as though it were a batch of its own). 
        # When calling criterion, only call on the relevant parts of the score and label vectors. 
        # Sum losses across all calls to criterion.
        loss = 0
        for i in range(inputs.shape[0]):
            # scores shape: torch.Size([1, batch_size, 1024, 1])
            comment_scores = scores[0][i][:][:]
            comment_labels = labels[i][:]
            comment_attention_mask = attention_mask[i][:]
            sliced_loss = criterion(comment_scores.flatten()[:lengths[i]], comment_labels.flatten()[:lengths[i]])
            loss += sliced_loss

        total_batch_loss = loss / sum(lengths)

        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()
        if rank == 0:
            wandb.log({"train_batch_loss": total_batch_loss.item()}) # TEDDI TODO add accuracy
        epoch_loss += total_batch_loss.item()
    epoch_loss = epoch_loss/len(loader)
    if rank == 0:
        wandb.log({"train_epoch_loss": epoch_loss})
    return epoch_loss

def evaluate(model, val_dataset, rank, lr=1e-3, batch_size=128, num_workers=2):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(rank)
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate)
    epoch_loss = 0
    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        inputs, labels, attention_mask, lengths = batch
        inputs = inputs.to(rank)
        labels = labels.to(rank)
        attention_mask = attention_mask.to(rank)

        scores = model(inputs, attention_mask)


        # Call criterion for each element in batch (as though it were a batch of its own). 
        # When calling criterion, only call on the relevant parts of the score and label vectors. 
        # Sum losses across all calls to criterion.
        loss = 0
        for i in range(inputs.shape[0]):
            comment_scores = scores[0][i][:][:]
            comment_labels = labels[i][:]
            comment_attention_mask = attention_mask[i][:]
            sliced_loss = criterion(comment_scores.flatten()[:lengths[i]], comment_labels.flatten()[:lengths[i]])
            loss += sliced_loss

        total_batch_loss = loss / sum(lengths)
        epoch_loss += total_batch_loss.item()
    return epoch_loss/len(loader)


def main(rank, world_size, args):

    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    if args.ckpt: # TODO probably change for DPP. Also, currently, no special action is taken with a model loaded from ckpt
        checkpoint = torch.load(args.ckpt, map_location=rank)
        start_epoch = checkpoint['epoch'] + 1
        best_val_metric = checkpoint['best_metric']
        model_args = checkpoint['args']
        model = GPT2Tox()
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(rank)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loaded model from ckpt!')
    else:
        model = GPT2Tox().to(rank)
        model = DDP(model, device_ids=[rank])

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dataset = Dataset(args.train_data, 0, args.batch_size, 'gpt2')
    val_dataset = Dataset(args.val_data, 0, args.batch_size, 'gpt2')
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_metric = 1e8
    wandb.init(
        # set the wandb project where this run will be logged
        project="FUDGE GPT2 CLF",

        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "architecture": "gpt2",
            "dataset": "original detoxify",
            "num_epochs": args.num_epochs,

        }
    )
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_dataset, optimizer, rank, lr=args.lr,  batch_size=args.batch_size, num_workers=2)
        val_loss = evaluate(model, val_dataset, rank, lr=args.lr, batch_size=args.batch_size, num_workers=2)

        if rank == 0:
            wandb.log({"val_epoch_loss": val_loss})
            if val_loss < best_val_metric:
                print('new best val metric', val_loss)
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

    cleanup()

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--lr', type=float, required=False, default=1e-3)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--num_epochs', type=int, required=False, default=100)
    parser.add_argument('--save_dir', type=str, required=False, default='./ckpt/model1')
    parser.add_argument('--ckpt', type=str, required=False, default=None)
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    mp.spawn(main,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

