import argparse
from pathlib import Path
import numpy as np
from faiss.contrib.ondisk import merge_ondisk
from tqdm import tqdm
import faiss

parser = argparse.ArgumentParser(description="Train a faiss model to perform nearest-neighbor search.")

parser.add_argument("--input", type=str, help="Input filename", default="vectors.csv")
parser.add_argument("--d", type=int, help="Vector dimensions", default=100)
parser.add_argument("--batch_size", type=int, help="Size of batch to train faiss with", default=10000)

args = parser.parse_args()

# create faiss directory if it doesn't exist
Path("faiss").mkdir(parents=True, exist_ok=True)

index = faiss.index_factory(args.d, f"IVF100,Flat")
print("training faiss index...")

first_vecs = np.loadtxt(args.input, delimiter=",", max_rows=args.batch_size).astype(np.float32)

index.train(first_vecs)
faiss.write_index(index, "faiss/trained.index")

n_batch = 0
while True:
    index = faiss.read_index("faiss/trained.index")
    batch = np.loadtxt(
        args.input,
        delimiter=",",
        skiprows=n_batch * args.batch_size,
        max_rows=args.batch_size,
    ).astype(np.float32)
    if len(batch) == 0:
        break
    else:
        index.add_with_ids(
            batch, np.arange(n_batch * args.batch_size, (n_batch + 1) * args.batch_size)
        )
        print(
            f"write block_{n_batch}.index with {n_batch*args.batch_size} as starting index"
        )
        faiss.write_index(index, f"faiss/block_{n_batch}.index")
        n_batch += 1

print("loading trained index")
# construct the output index
index = faiss.read_index("faiss/trained.index")

block_fnames = [f"faiss/block_{b}.index" for b in range(n_batch)]

merge_ondisk(index, block_fnames, "faiss/merged_index.ivfdata")

print("write populated.index")
faiss.write_index(index, "faiss/populated.index")
