from pathlib import Path
from joblib import dump
from sklearn.neighbors import NearestNeighbors
import numpy as np
from utils import fvecs_read

print("loading base vectors...")
S_vecs = fvecs_read("sift/sift_base.fvecs")
k = 5

neigh = NearestNeighbors(n_neighbors=k)

print("fitting model...")
neigh.fit(S_vecs)

print("dumping model...")
# create sklearn directory if it doesn't exist
Path("sklearn").mkdir(parents=True, exist_ok=True)
dump(neigh, "sklearn/nn")
