from pathlib import Path
import numpy as np
from faiss.contrib.ondisk import merge_ondisk
from tqdm import tqdm
import faiss
from utils import fvecs_read

print("loading input vectors...")
S_vecs = fvecs_read("../sift/sift_base.fvecs")
batch_size = 100000

# create faiss directory if it doesn't exist
Path("../faiss").mkdir(parents=True, exist_ok=True)

index = faiss.index_factory(S_vecs.shape[1], f"IVF100,Flat")
print("training faiss index...")

index.train(S_vecs[0:batch_size])
faiss.write_index(index, "../faiss/trained.index")

n_batch = 0
while n_batch*batch_size < S_vecs.shape[0]:
    index = faiss.read_index("../faiss/trained.index")
    index.add_with_ids(S_vecs[n_batch * batch_size:(n_batch + 1) * batch_size], np.arange(n_batch * batch_size, (n_batch + 1) * batch_size))
    print(
        f"write block_{n_batch}.index with {n_batch*batch_size} as starting index"
    )
    faiss.write_index(index, f"../faiss/block_{n_batch}.index")
    n_batch += 1


print("loading trained index")
# construct the output index
index = faiss.read_index("../faiss/trained.index")

block_fnames = [f"../faiss/block_{b}.index" for b in range(n_batch)]

merge_ondisk(index, block_fnames, "../faiss/merged_index.ivfdata")

print("write populated.index")
faiss.write_index(index, "../faiss/populated.index")
