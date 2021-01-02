import faiss
from faiss.contrib.ondisk import merge_ondisk

from utils import fvecs_read

print("loading query vectors...")
xq = fvecs_read("../gist/gist_query.fvecs")

index = faiss.read_index("../faiss/populated.index")
index.nprobe = 80
k = 5

print(f"getting nearest neighbors for {xq.shape[0]} vectors...")
distances, indices = index.search(xq, k)
print(indices)
