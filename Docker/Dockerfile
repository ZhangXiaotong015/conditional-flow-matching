FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

WORKDIR /scratch

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /scratch

CMD ["bash"]
