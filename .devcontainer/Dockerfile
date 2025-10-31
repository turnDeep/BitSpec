# Dockerfile for Mass Spectrum Prediction Dev Container
# RTX 50シリーズ対応のCUDA 12.8環境

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

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

# Node.js のインストール (Claude CLI に必要)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Claude CLI のインストール
RUN npm install -g @anthropic-ai/claude-code

# python と pip のシンボリックリンクを作成
# -f オプションで既存のファイルを上書き
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# pipのアップグレード
RUN python3 -m pip install --upgrade pip setuptools wheel

# 作業ディレクトリの設定
WORKDIR /workspace

# requirements.txtをコピーして依存関係をインストール
COPY requirements.txt /tmp/requirements.txt

# PyTorchとその他の依存関係のインストール
# RTX 50対応のためにnightly buildを使用
RUN pip3 install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch>=2.7.0.dev torchvision>=0.22.0.dev torchaudio>=2.7.0.dev || \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

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
# ビルド済みホイールを使用してインストール (sm_120対応)
# --no-build-isolationを使用してビルド時にtorchを利用可能にする
RUN TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])") && \
    CUDA_VERSION=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))") && \
    echo "PyTorch version: ${TORCH_VERSION}, CUDA version: ${CUDA_VERSION}" && \
    pip3 install --no-cache-dir torch-geometric && \
    pip3 install --no-cache-dir ninja && \
    pip3 install --no-build-isolation --no-cache-dir \
    torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html || \
    (echo "Warning: Failed to install from PyG wheels with ${CUDA_VERSION}, trying with cu121..." && \
     pip3 install --no-build-isolation --no-cache-dir torch-scatter torch-sparse torch-cluster \
     -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu121.html) || \
    (echo "Warning: Failed to install from PyG wheels, building from source..." && \
     pip3 install --no-build-isolation --no-cache-dir torch-scatter torch-sparse torch-cluster)

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

# devuser の PATH 設定を追加
RUN echo 'export PATH="/opt/node22/bin:$PATH"' >> /home/devuser/.bashrc && \
    echo 'export PATH="/opt/node22/bin:$PATH"' >> /home/devuser/.profile

USER devuser

# デフォルトコマンド
CMD ["/bin/bash"]
