import sys
from utils import fvecs_read
import numpy as np
from faiss.contrib.ondisk import merge_ondisk
import faiss

print("loading query vectors...")
q_vecs = fvecs_read("sift/sift_query.fvecs")

index = faiss.read_index("faiss/populated.index")
index.nprobe = 16
index.make_direct_map()

np.set_printoptions(threshold=q_vecs.shape[0])

print(f"getting nearest neighbors for {q_vecs.shape[0]} vectors...")
distances, indices = index.search(q_vecs, 5)
print(indices)
