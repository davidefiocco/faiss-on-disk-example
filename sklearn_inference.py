import argparse

import numpy as np
from joblib import load
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser(description="Perform nearest-neighbor search with scikit-learn.")

parser.add_argument("--d", type=int, help="Vector dimensions", default=100)
parser.add_argument("--iter", type=int, help="Inference cycles", default=100)

args = parser.parse_args()

print("loading sklearn model...")
neigh = load("sklearn/nn")

np.random.seed(42)
print(f"getting nearest neighbors for {args.d} vectors...")
for i in range(args.iter):
    q_vec = np.random.random(args.d)
    result = neigh.kneighbors([q_vec])[1][0]
    print(result)
