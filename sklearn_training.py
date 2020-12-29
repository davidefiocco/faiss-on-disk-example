import argparse
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser(description="Train a scikit-learn model to perform nearest-neighbor search.")

parser.add_argument("--input", type=str, help="Input filename", default="vectors.csv")

args = parser.parse_args()

print("loading input vectors...")
vecs = np.loadtxt(args.input, delimiter=",").astype(np.float32)
neigh = NearestNeighbors(n_neighbors=5)

print("fitting model...")
neigh.fit(vecs)

print("dumping model...")
# create sklearn directory if it doesn't exist
Path("sklearn").mkdir(parents=True, exist_ok=True)
dump(neigh, "sklearn/nn")
