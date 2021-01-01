# from-numpy-to-faiss-NN-search

The code here allows to search for nearest neighbors on a benchmark set using different libraries (`numpy`, `sklearn`, `faiss`), and experiment with differences between them. The `faiss` example is meant to work also with a number of vectors not fitting into RAM.

## Running the examples

To run the examples, on a machine running Docker, run:

```bash
docker build -t nnsearch:latest .
docker run --name nn -d nnsearch:latest
docker exec -it nn bash
cd workspace
```

and then get and inflate [1M SIFT vectors](http://corpus-texmex.irisa.fr/) (a benchmark for nearest-neighbors search) with:

```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzvf sift.tar.gz 
```

To perform nearest neighbors search with `numpy`, run:

```bash
cd src
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

when done with runs, `make clean` in the root folder should clean up all files created on the way.

## Profiling

To monitor memory usage during script execution one can use `memory_profiler`:

```bash
# requires to have run python faiss_training.py before
mprof run faiss_inference.py
# generate memory usage plot vs time
mprof plot -o faiss_inference
```