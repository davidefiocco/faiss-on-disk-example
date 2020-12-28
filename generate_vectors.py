import argparse

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Write vectors to csv file on disk.")

parser.add_argument("--n", type=int, help="Number of vecs to generate", default=50000)
parser.add_argument("--d", type=int, help="Vector dimensions", default=100)
parser.add_argument("--output", type=str, help="Output filename", default="vectors.csv")

args = parser.parse_args()

np.random.seed(42)

print(f"generating {args.n} {args.d}-dimensional vectors...")

with open(args.output, "w") as f:
    for i in tqdm(range(0, args.n)):
        vec = np.random.random(args.d)
        vec_string = np.array2string(vec, separator=", ", max_line_width=np.inf)
        f.write(vec_string[1 : len(vec_string) - 1] + "\n")
