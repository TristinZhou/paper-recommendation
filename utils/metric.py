import pandas as pd
import numpy as np


def ranking_metric(embeddings, source, target):
    import faiss
    pd.set_option('display.width', 1000, 'display.max_rows', 1000)
    print("embeddings: {}".format(embeddings.shape))
    corpus = np.array(embeddings[target].values.tolist()).astype('float32')
    faiss.normalize_L2(corpus)
    index = faiss.IndexFlatIP(corpus.shape[1])
    index.train(corpus)
    index.add(corpus)
    query = np.array(embeddings[source].values.tolist()).astype('float32')
    faiss.normalize_L2(query)
    D, I = index.search(query, len(corpus))
    mmr = 0
    for i, d in enumerate(D):
        rank = np.where(i == I[i])[0]
        if rank <= 2:
            acc = 1.0/(rank+1)
            mmr += acc[0]
    mmr = mmr / query.shape[0]
    print(source, target, mmr)
