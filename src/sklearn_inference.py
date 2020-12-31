from joblib import load
from sklearn.neighbors import NearestNeighbors
import numpy as np
from utils import fvecs_read

print("loading query vectors...")
q_vecs = fvecs_read("../sift/sift_query.fvecs")

print("loading sklearn model...")
neigh = load("../sklearn/nn")

print(f"getting nearest neighbors for {q_vecs.shape[0]} vectors...")
result = neigh.kneighbors(q_vecs, return_distance = False)
print(result)
