import os
import time
import pandas as pd
import torch
import numpy as np
import torchtext
from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from models import TransformerPool
from parser import parameter_parser
from dataset import padding
from dataset import EvalDataset
from utils import ranking_metric, set_manual_seed


args = parameter_parser()
set_manual_seed(args.seed)
GLOVE = torchtext.vocab.GloVe(name='6B', dim=300)


def eval_worker(model, dataset, device, id, column, rank=0, world_size=None, mgr=None):
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

    evalset = EvalDataset(dataset, id, column)
    if in_distributed_mode:
        datasampler = DistributedSampler(evalset, shuffle=False)
        dataloader = DataLoader(evalset, pin_memory=True,
                                num_workers=0, batch_size=1, sampler=datasampler)
    else:
        dataloader = DataLoader(evalset,
                                pin_memory=True, num_workers=4, batch_size=1)

    model.eval()
    embeddings = []
    with torch.no_grad():
        bar = tqdm(desc='Valid ' + column, total=len(dataloader),
                   leave=True) if rank == 0 else None
        for index, input in enumerate(dataloader):
            try:
                value, text = input
                text = text.cuda()
                output = model(text)
            except Exception:
                if id == 'description_id':
                    output = torch.rand(1, args.embedding_dim)
            finally:
                try:
                    output = output.squeeze(dim=0)
                    embeddings.append({
                        "embedding": output.detach().cpu().numpy(),
                        id: value})
                except Exception:
                    pass
                bar.update() if rank == 0 else None
        bar.close() if rank == 0 else None

    if in_distributed_mode:
        mgr[rank] = embeddings
    else:
        return embeddings


def workers(rank, mode, dataset, world_size, devices, id, column,
            model=None, mgr=None, port=args.port, ip='localhost'):
    os.environ['MASTER_ADDR'] = ip
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    assert model is not None and mgr is not None
    eval_worker(model, dataset, devices[rank], id,
                column, rank, world_size, mgr)


def eval(dataset, id, column):
    world_size = len(args.cuda_devices)
    if world_size == 1:
        embeddings = eval_worker(
            args.model_path, dataset, args.cuda_devices[0], id, column)
    else:
        mgr = mp.get_context('spawn').Manager().dict()
        mp.spawn(workers, args=('eval', dataset, world_size,
                                args.cuda_devices, id,
                                column,
                                args.model_path, mgr,
                                args.port), nprocs=world_size)
        embeddings = None
        for i in range(world_size):
            if embeddings is None:
                embeddings = mgr[i]
            else:
                embeddings.extend(mgr[i])
    return embeddings


def submit(valid_data, all_data, mapped_data):
    t1 = time.time()
    corpus = np.array(all_data['embedding'].values.tolist()).astype('float32')
    import faiss
    faiss.normalize_L2(corpus)
    index = faiss.IndexFlatIP(corpus.shape[1])
    index.train(corpus)
    index.add(corpus)
    query = np.array(valid_data['embedding'].values.tolist()).astype('float32')
    faiss.normalize_L2(query)
    D, I = index.search(query, len(corpus))
    res = []
    for i, d in enumerate(I):
        index_lst = I[i][:3]
        paper_id_lst = [all_data.loc[index, 'paper_id'][0]
                        for index in index_lst]
        description_id = valid_data.loc[i, 'description_id'][0]
        res.append({
            "description_id": description_id,
            "paper_id_lst": ",".join(paper_id_lst)
        })
    print("Time {:.02f}s".format(time.time()-t1))
    res = pd.DataFrame(res)
    res.to_csv("./result/submit.csv", index=0, header=0)


if __name__ == '__main__':
    all_data_handed_path = "./data/candidate_paper_for_wsdm2020_handed.csv"
    valid_data_handed_path = "./data/validation_handed.csv"
    mapped_data_path = "./data/result_mapped.csv"
    all_data = pd.read_csv(all_data_handed_path, low_memory=False)
    valid_data = pd.read_csv(valid_data_handed_path)
    all_data_embedding = eval(all_data, "paper_id", "title_abstract")
    all_data_df = pd.DataFrame(all_data_embedding, columns=[
                               'embedding', 'paper_id'])
    valid_data_embedding = eval(
        valid_data, "description_id", "description_text")
    valid_data_df = pd.DataFrame(valid_data_embedding, columns=[
                                 'embedding', 'description_id'])
    mapped_data = pd.read_csv(mapped_data_path)
    submit(valid_data_df, all_data_df, mapped_data)
