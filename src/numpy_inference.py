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
indices = np.zeros((xq.shape[0], 5))
for i in range(xq.shape[0]):
    indices[i, :] = find_neighbors(xb, xq[i])

# Simple benchmark of the quality of the search
from utils import ivecs_read
import numpy as np

iqt = ivecs_read("../gist/gist_groundtruth.ivecs")

print("Top1 accuracy on the 1-NN search: ", np.mean(indices[:, 0] == iqt[:, 0]))
