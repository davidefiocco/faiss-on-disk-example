FROM continuumio/miniconda3

RUN apt-get install make

# install faiss for cpu use
RUN conda install faiss-cpu -c pytorch

RUN conda install scikit-learn isort black memory_profiler matplotlib

RUN mkdir /workspace
COPY . /workspace

ENTRYPOINT ["tail", "-f", "/dev/null"]