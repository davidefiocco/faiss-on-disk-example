# faiss-sklearn-numpy-NN-search

The code here allows to generate a set of vectors in csv format and then experiment with different ways to search for nearest neighbors using different libraries (`numpy`, `sklearn`, `faiss`), and experiment with differences between them. The `faiss` example is meant to work also with a number of vectors not fitting into RAM!

## Running the examples

To run the examples, on a machine running Docker, run:

```bash
docker build -t faisssklearnnumpynnsearch:latest .
docker run --name nn -d faisssklearnnumpynnsearch:latest
docker exec -it nn bash
cd workspace
```

and then generate the vectors in the container with 

```bash
# beware that the vector generation performed here is not very fast...
python generate_vectors.py --n 5000000
```

To perform nearest neighbors search with `numpy`, run:

```bash
python numpy_inference.py
```

To perform the same search with `scikit-learn`, run:

```bash
python sklearn_training.py
python sklearn_inference.py
```

To perform the same search with `faiss` (meant to scale to large numbers of vectors), run:

```bash
python faiss_training.py
python faiss_inference.py
```

when done with runs, `make clean` should clean up all files created on the way.

## Profiling

To monitor memory usage during script execution one can use `memory_profiler`:

```bash
# requires to have run python faiss_training.py before
mprof run faiss_inference.py 
mprof plot -o faiss_inference
```