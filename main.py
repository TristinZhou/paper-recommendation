import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchtext
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from parser import parameter_parser
from utils import set_manual_seed
from models import TripletMarginRankingLoss
from models import TransformerPool
from models import Lookahead, RAdam
from models import init_weights
from utils import ranking_metric
from dataset import TripletDataset, TestDataset

args = parameter_parser()
GLOVE = torchtext.vocab.GloVe(name='6B', dim=300)


def train_worker(dataset, device, rank=0, world_size=None):
    torch.cuda.set_device(device)
    criterion = TripletMarginRankingLoss(args.loss_margin)
    model = TransformerPool(args.vocab_size, args.embedding_dim,
                            args.hidden_dim, pre_trained=GLOVE)
    if args.re_train:
        model.load_state_dict(torch.load(
            args.train_model, map_location='cuda:{}'.format(device)))
    else:
        model.apply(init_weights)
    model, criterion = model.to(device), criterion.to(device)
    triplet_dataset = TripletDataset(dataset)

    in_distributed_mode = True if world_size else False
    if in_distributed_mode:
        rank, device = torch.distributed.get_rank(), torch.cuda.current_device()
        print("rank:{}, device:{}".format(rank, device))

    if in_distributed_mode:
        model = DistributedDataParallel(
            model, device_ids=[device])
        datasampler = DistributedSampler(triplet_dataset)
        dataloader = DataLoader(triplet_dataset, shuffle=False,
                                pin_memory=True, num_workers=0,
                                batch_size=args.batch_size, sampler=datasampler)
    else:
        dataloader = DataLoader(triplet_dataset, shuffle=True,
                                pin_memory=True, num_workers=4,
                                batch_size=args.batch_size)

    optimizer = RAdam(
        model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.t_max, eta_min=args.eta_min)

    model.train()
    best_avg_loss = None
    t1 = time.time()
    for epoch in range(args.epoch):
        datasampler.set_epoch(epoch) if in_distributed_mode else None
        total_loss = []
        bar = tqdm(desc='EPOCH {:02d}'.format(epoch), total=len(
            dataloader), leave=False) if rank == 0 else None

        for triplet in dataloader:
            optimizer.zero_grad()
            anchor, positive, negative = model(triplet)
            loss = criterion(anchor, positive, negative)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss.append(loss.item())
            bar.update() if rank == 0 else None

        if rank == 0:
            bar.close()
            epoch_avg_loss = np.mean(total_loss)
            scheduler.step(epoch_avg_loss)
            print("Epoch {:02d}, Time {:.02f}s, AvgLoss {:.08f}, lr {:.8f}".format(
                epoch, time.time()-t1, epoch_avg_loss, optimizer.param_groups[0]['lr']))
            if best_avg_loss is None or epoch_avg_loss < best_avg_loss:
                best_avg_loss = epoch_avg_loss
                state_dict = model.module.state_dict() if in_distributed_mode else model.state_dict()
                torch.save(state_dict, args.model_path)
            t1 = time.time()
        scheduler.step(epoch)
        torch.cuda.empty_cache()
    return


def eval_worker(model, dataset, device, rank=0, world_size=None, mgr=None):
    torch.cuda.set_device(device)
    if isinstance(model, str):
        model = TransformerPool(args.vocab_size, args.embedding_dim,
                                args.hidden_dim, pre_trained=GLOVE)
        model = model.to(device)
        model.load_state_dict(torch.load(
            args.model_path, map_location='cuda:{}'.format(device)))

    in_distributed_mode = True if world_size else False
    if in_distributed_mode:
        rank, device = torch.distributed.get_rank(), torch.cuda.current_device()
        print("rank:{}, device:{}".format(rank, device))

    testset = TestDataset(dataset)
    if in_distributed_mode:
        datasampler = DistributedSampler(testset, shuffle=False)
        dataloader = DataLoader(testset, pin_memory=True,
                                num_workers=0, batch_size=1, sampler=datasampler)
    else:
        dataloader = DataLoader(testset,
                                pin_memory=True, num_workers=4, batch_size=1)

    model.eval()
    embeddings = testset.dataframe().iloc[0:0]
    with torch.no_grad():
        bar = tqdm(desc='EVAL', total=len(dataloader),
                   leave=True) if rank == 0 else None
        for input in dataloader:
            try:
                input = list(map(lambda s: s.cuda(), input))
                output = model(input)
            except KeyError:
                pass
            except RuntimeError:  # cuda out of memory
                pass
            else:
                output = torch.cat((output[0], output[1]))
                output = output.reshape(-1, embeddings.columns.size,
                                        args.embedding_dim)
                output = pd.DataFrame(
                    output.tolist(), columns=embeddings.columns)
                embeddings = embeddings.append(output, ignore_index=True)
            finally:
                bar.update() if rank == 0 else None
        bar.close() if rank == 0 else None

    if in_distributed_mode:
        mgr[rank] = embeddings
    else:
        return embeddings


def workers(rank, mode, dataset, world_size, devices, model=None, mgr=None, port=args.port, ip='localhost'):
    os.environ['MASTER_ADDR'] = ip
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if mode == 'train':
        train_worker(dataset, devices[rank], rank, world_size)
    elif mode == 'eval':
        assert model is not None and mgr is not None
        eval_worker(model, dataset, devices[rank], rank, world_size, mgr)


def train(dataset):
    world_size = len(args.cuda_devices)
    if world_size == 1:
        train_worker(dataset, args.cuda_devices[0])
    else:
        set_manual_seed(args.seed)
        mp.spawn(workers, args=('train', dataset, world_size,
                                args.cuda_devices), nprocs=world_size)
    return


def eval(dataset):
    world_size = len(args.cuda_devices)
    if world_size == 1:
        embeddings = eval_worker(
            args.model_path, dataset, args.cuda_devices[0])
    else:
        mgr = mp.get_context('spawn').Manager().dict()
        mp.spawn(workers, args=('eval', dataset, world_size,
                                args.cuda_devices, args.model_path, mgr, args.port), nprocs=world_size)
        embeddings = None
        for i in range(world_size):
            if embeddings is None:
                embeddings = mgr[i]
            else:
                embeddings = embeddings.append(mgr[i], ignore_index=True)
        ranking_metric(embeddings, args.source, args.target)
    return


if __name__ == "__main__":
    if args.mode == 'train':
        train(args.data)
    else:
        eval(args.data)
