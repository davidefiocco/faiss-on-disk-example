import faiss
from faiss.contrib.ondisk import merge_ondisk

from utils import fvecs_read

print("loading query vectors...")
xq = fvecs_read("../gist/gist_query.fvecs")

index = faiss.read_index("../faiss/populated.index")
index.nprobe = 120
k = 5

print(f"getting nearest neighbors for {xq.shape[0]} vectors...")
distances, indices = index.search(xq, k)

# Simple benchmark of the quality of the search
from utils import ivecs_read
import numpy as np

iqt = ivecs_read("../gist/gist_groundtruth.ivecs")

print("Top1 accuracy on the 1-NN search: ", np.mean(indices[:, 0] == iqt[:, 0]))
