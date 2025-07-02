FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash
ENV PATH="/opt/conda/bin:$PATH"

COPY conda-linux-env.yml /opt/conda/envs/conda-linux-env.yml
RUN conda env create -f /opt/conda/envs/conda-linux-env.yml
ENV PATH="/opt/conda/envs/sres/bin:$PATH"
