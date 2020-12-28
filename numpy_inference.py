import argparse

import numpy as np

parser = argparse.ArgumentParser(description="Perform nearest-neighbor search with numpy.")

parser.add_argument("--d", type=int, help="Vector dimensions", default=100)
parser.add_argument("--input", type=str, help="Input filename", default="vectors.csv")
parser.add_argument("--iter", type=int, help="Inference cycles", default=100)

args = parser.parse_args()

print("loading input vectors...")
batch = np.loadtxt(args.input, delimiter=",").astype(np.float32)


def find_neighbors_dot(batch, xq, k=5):
    product = np.dot(batch, xq)
    return np.argpartition(product, range(-k, 0))[-k:][::-1]


np.random.seed(42)
print(f"getting nearest neighbors for {args.d} vectors...")
for i in range(args.iter):
    xq = np.random.random(args.d)
    xq_norm = xq / np.linalg.norm(xq)
    result = find_neighbors_dot(batch, xq_norm)
    print(result)
