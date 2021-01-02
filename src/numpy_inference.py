import numpy as np

from utils import fvecs_read

print("loading base vectors...")
xb = fvecs_read("../gist/gist_base.fvecs")
print("loading query vectors...")
xq = fvecs_read("../gist/gist_query.fvecs")


def find_neighbors(xb, xq, k=5):
    distances = np.linalg.norm(xb - xq, axis=1)
    return np.argpartition(distances, range(0, k))[:k]


print(f"getting nearest neighbors for {xq.shape[0]} vectors...")
for i in range(xq.shape[0]):
    indices = find_neighbors(xb, xq[i])
    print(indices)
