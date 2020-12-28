import argparse
import sys

import numpy as np
from faiss.contrib.ondisk import merge_ondisk

import faiss

parser = argparse.ArgumentParser(description="Perform nearest-neighbor search with faiss.")

parser.add_argument("--d", type=int, help="Vector dimensions", default=100)
parser.add_argument("--iter", type=int, help="Inference cycles", default=100)

args = parser.parse_args()


def find_neighbors_faiss(xq):
    return index.search(xq, 5)


index = faiss.read_index("faiss/populated.index")
index.nprobe = 16
index.make_direct_map()

np.random.seed(42)
for i in range(args.iter):
    xq = np.random.random(args.d).reshape(1, -1).astype(np.float32)
    # xq = index.reconstruct(i).reshape(1,-1).astype(np.float32)
    D, I = find_neighbors_faiss(xq)
    print(I[0])
