from pathlib import Path

import faiss
import numpy as np
from faiss.contrib.ondisk import merge_ondisk

from utils import fvecs_read

# create faiss directory if it doesn't exist
Path("../faiss").mkdir(parents=True, exist_ok=True)

print("loading input vectors...")
xb = fvecs_read("../gist/gist_base.fvecs")

index = faiss.index_factory(xb.shape[1], "IVF4000,Flat")

batch_size = 100000

print("training faiss index...")
index.train(xb[0:batch_size])
faiss.write_index(index, "../faiss/trained.index")

n_batches = xb.shape[0] // batch_size
for i in range(n_batches):
    index = faiss.read_index("../faiss/trained.index")
    index.add_with_ids(
        xb[i * batch_size : (i + 1) * batch_size],
        np.arange(i * batch_size, (i + 1) * batch_size),
    )
    print(f"writing block_{i}.index with {i*batch_size} as starting index")
    faiss.write_index(index, f"../faiss/block_{i}.index")

# construct the output index
index = faiss.read_index("../faiss/trained.index")
block_fnames = [f"../faiss/block_{b}.index" for b in range(n_batches)]

merge_ondisk(index, block_fnames, "../faiss/merged_index.ivfdata")

print("write populated.index")
faiss.write_index(index, "../faiss/populated.index")
