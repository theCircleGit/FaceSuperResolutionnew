FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash
ENV PATH="/opt/conda/bin:$PATH"

RUN apt install -y cuda-nvcc-11-8 build-essential

RUN apt-get install -y libgl1 libglib2.0-0

COPY conda-linux-env.yml /opt/conda/envs/conda-linux-env.yml

# Create the environment, but WITHOUT installing pytorch and torchvision initially
RUN conda env create -f /opt/conda/envs/conda-linux-env.yml --no-default-packages

# Activate the environment
SHELL ["/bin/bash", "-c", "source /opt/conda/bin/activate sres && conda activate sres"]

# Install pytorch and torchvision AFTER activating the env, forcing the correct CUDA version
# RUN conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
RUN conda install -c pytorch torchvision==0.15.2 pytorch-cuda=11.8

# Verify the installation (optional, but recommended)
RUN python -c "import torch; print(torch.__version__); print(torch.version.cuda); import torchvision; print(torchvision.__version__)"

ENV PATH="/opt/conda/envs/sres/bin:$PATH"

RUN mkdir /app
COPY . /app/

WORKDIR /app
ENV BASE_DIR=/app
