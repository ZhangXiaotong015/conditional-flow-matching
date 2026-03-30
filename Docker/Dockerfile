FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

WORKDIR /scratch

RUN apt-get update && apt-get install -y \
    wget git bzip2 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

COPY requirements.txt .
RUN conda create -n torchcfm python=3.10 -y && \
    conda run -n torchcfm pip install --no-cache-dir -r requirements.txt

COPY . /scratch

CMD ["bash"]
