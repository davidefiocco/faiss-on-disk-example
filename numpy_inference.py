import argparse

import numpy as np

parser = argparse.ArgumentParser(
    description="Perform nearest-neighbor search with numpy."
)

parser.add_argument("--d", type=int, help="Vector dimensions", default=100)
parser.add_argument("--input", type=str, help="Input filename", default="vectors.csv")
parser.add_argument("--iter", type=int, help="Inference cycles", default=100)

args = parser.parse_args()

print("loading input vectors...")
batch = np.loadtxt(args.input, delimiter=",").astype(np.float32)


def find_neighbors(batch, xq, k=5):
    distances = np.linalg.norm(batch - xq, axis = 1)
    return np.argpartition(distances, range(0, k))[:k]


np.random.seed(42)
print(f"getting nearest neighbors for {args.d} vectors...")
for i in range(args.iter):
    xq = np.random.random(args.d)
    result = find_neighbors(batch, xq)
    print(result)
