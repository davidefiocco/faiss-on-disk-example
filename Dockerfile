FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends make wget && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /workspace
COPY . /workspace
WORKDIR /workspace

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["tail", "-f", "/dev/null"]
