import numpy as np
from utils import fvecs_read

print("loading base vectors...")
S_vecs = fvecs_read("sift/sift_base.fvecs")
print("loading query vectors...")
q_vecs = fvecs_read("sift/sift_query.fvecs")

def find_neighbors(S_vecs, q_vec, k=5):
    distances = np.linalg.norm(S_vecs - q_vec, axis = 1)
    return np.argpartition(distances, range(0, k))[:k]

print(f"getting nearest neighbors for {q_vecs.shape[0]} vectors...")
for i in range(q_vecs.shape[0]):
    indices = find_neighbors(S_vecs, q_vecs[i])
    print(indices)
