# faiss-on-disk-example

This repo contains code to run `faiss` to search for neighbors in a dataset not fitting into RAM.

## Running the examples

To run the examples, on a machine running Docker, run:

```bash
docker build -t nnsearch:latest .
docker run --name nn -d nnsearch:latest
docker exec -it nn bash
cd workspace
```

and then get and inflate [1M GIST vectors](http://corpus-texmex.irisa.fr/) (a benchmark for vector nearest-neighbors search) with:

```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzvf gist.tar.gz 
```

To perform nearest neighbors search with `numpy` (this can fail on machines not having 8+GB of RAM for the process), run:

```bash
cd src
python numpy_inference.py
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