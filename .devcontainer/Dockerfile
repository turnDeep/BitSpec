# Dockerfile for Mass Spectrum Prediction Dev Container
# RTX 40/50シリーズ対応のCUDA 12.1環境

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# pipのアップグレード
RUN python3 -m pip install --upgrade pip setuptools wheel

# 作業ディレクトリの設定
WORKDIR /workspace

# requirements.txtをコピーして依存関係をインストール
COPY requirements.txt /tmp/requirements.txt

# PyTorchとその他の依存関係のインストール
# CUDA 12.1対応の安定版を使用（PyG拡張機能の互換性のため）
RUN pip3 install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# RDKitのインストール（化学構造処理）
RUN pip3 install --no-cache-dir rdkit

# その他の主要な依存関係
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    plotly \
    tqdm \
    pyyaml \
    h5py \
    jupyter \
    ipython

# PyTorch Geometricとその依存関係
# PyTorch 2.5.1とCUDA 12.1用の事前ビルド済みホイールを使用
RUN pip3 install --no-cache-dir torch-geometric && \
    pip3 install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# その他の化学計算ライブラリ
RUN pip3 install --no-cache-dir \
    mordred || echo "mordred installation failed, continuing..." && \
    pip3 install --no-cache-dir mol2vec || echo "mol2vec installation failed, continuing..."

# 開発用ツール
RUN pip3 install --no-cache-dir \
    pytest \
    black \
    flake8 \
    mypy

# 非rootユーザーの作成
RUN useradd -m -s /bin/bash devuser && \
    chown -R devuser:devuser /workspace

USER devuser

# デフォルトコマンド
CMD ["/bin/bash"]
