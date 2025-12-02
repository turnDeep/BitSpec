# QC-GN2oMS2-EI システム詳細技術仕様書 v4.2
## PyTorch統一環境・最小構成アプローチ（反復改善戦略）

**作成日**: 2025-12-02
**対象システム**: NExtIMS (NIST EI-MS Prediction System)
**基盤アーキテクチャ**: QC-GN2oMS2 (PNNL)
**ハードウェア**: NVIDIA GeForce RTX 5070 Ti (Blackwell sm_120)
**設計方針**: **Start Simple, Iterate Based on Evidence**

---

## 📋 目次

1. [主要変更点（v4.1 → v4.2）](#主要変更点v41--v42)
2. [設計哲学：最小構成アプローチ](#設計哲学最小構成アプローチ)
3. [システム概要](#システム概要)
4. [アーキテクチャ設計](#アーキテクチャ設計)
5. [Phase 0: BDE-db2環境構築](#phase-0-bde-db2環境構築)
6. [Phase 1: データ準備](#phase-1-データ準備)
7. [Phase 2: GNN学習](#phase-2-gnn学習)
8. [Phase 3: 評価と反復改善判断](#phase-3-評価と反復改善判断)
9. [Phase 4: 特徴量拡張（条件付き）](#phase-4-特徴量拡張条件付き)
10. [設定ファイル詳細](#設定ファイル詳細)
11. [開発環境構築](#開発環境構築)
12. [タイムライン](#タイムライン)
13. [参考文献](#参考文献)

---

## 主要変更点（v4.1 → v4.2）

### ✅ v4.2での大幅簡素化

| 項目 | v4.1 | v4.2 | 変更理由 |
|------|------|------|---------|
| **ノード特徴次元** | 128 (41+87) | **16 (16+0)** | QC-GN2oMS2実証済み設計に準拠 |
| **エッジ特徴次元** | 64 (12+52) | **3 (3+0)** | 最小限の特徴量（BDE+結合次数+環） |
| **予備次元** | 139 (87+52) | **0** | 実証主義アプローチ（必要性証明後に追加） |
| **メモリ使用量** | 約1.3GB | **約0.16GB** | **-88%削減** |
| **設計方針** | 拡張性重視 | **シンプルさ重視** | Start simple, iterate |

### 🎯 設計哲学の転換

**v4.1**: 「将来の拡張に備えて予備次元を大量に確保」
**v4.2**: 「最小構成で実装 → 性能評価 → 必要に応じて段階的に拡張」

**根拠**:
- QC-GN2oMS2がMS/MSで16次元ノード特徴、2次元エッジ特徴で**Cosine Similarity 0.88**を達成
- 過剰設計を避け、実証データに基づく意思決定
- 高速イテレーション（学習速度向上）によるアジャイル開発

---

## 設計哲学：最小構成アプローチ

### 基本原則

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: 最小構成で実装（v4.2）                          │
│   - ノード: 16次元（QC-GN2oMS2準拠）                     │
│   - エッジ: 3次元（BDE + 結合次数 + 環）                 │
│   - 目標: Cosine Similarity > 0.80                      │
└─────────────────────────────────────────────────────────┘
                    ↓
         ┌──────────────────────┐
         │ Phase 2: 性能評価    │
         │   - Cosine Sim       │
         │   - Top-K Recall     │
         │   - 汎化性能         │
         └──────────────────────┘
                    ↓
    ┌───────────────┴────────────────┐
    │ 判定                           │
    ├────────────┬──────────────────┤
    │ 十分       │ 不十分           │
    │ (>0.85)    │ (<0.85)          │
    ↓            ↓
┌─────────┐  ┌──────────────────────┐
│ 完了！  │  │ Phase 3: 特徴量分析  │
│ v4.2採用│  │   - Attention分析    │
└─────────┘  │   - Ablation study   │
             │   - 重要特徴の特定   │
             └──────────────────────┘
                    ↓
             ┌──────────────────────┐
             │ Phase 4: 段階的拡張  │
             │   - v4.3: 追加特徴   │
             │   - 再評価           │
             └──────────────────────┘
```

### QC-GN2oMS2の教訓

**彼らの成功事例**:
- MS/MSで16次元ノード、2次元エッジ
- Cosine Similarity 0.88達成
- 論文で実証済み

**我々の仮説**:
- EI-MSもシンプルな特徴量で十分な可能性
- 複雑さは段階的に追加すべき
- 最初から128次元は過剰設計の可能性大

---

## システム概要

### 目的

NIST 17 EI-MSデータベース（約280,000スペクトル、フィルタリング後）を用いて、**最小限の特徴量で高精度な**Graph Neural NetworkによるEI-MSスペクトル予測システムを構築する。

### アーキテクチャ概要図

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 0: BDE-db2環境構築（v4.1と同じ）                       │
├─────────────────────────────────────────────────────────────┤
│ 1. BDE-db2ダウンロード (531,244 reactions)                   │
│ 2. BonDNet再学習 (2-3日, RTX 5070 Ti)                        │
│ 3. 学習済みモデル検証 (MAE < 1.0 kcal/mol目標)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: データ準備（v4.1と同じ）                            │
├─────────────────────────────────────────────────────────────┤
│ 1.1 NIST 17読み込み (300,000 spectra)                       │
│ 1.2 データフィルタリング                                     │
│     - サポート元素チェック (C,H,O,N,F,S,P,Cl,Br,I)          │
│     - 分子量フィルタ (MW <= 1000 Da)                         │
│     → 280,000 spectra (93.3% retention)                     │
│ 1.3 BonDNet BDE計算 (70 min)                                │
│ 1.4 PyG Graph生成（16次元ノード、3次元エッジ）              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: GNN学習（最小構成）                                 │
├─────────────────────────────────────────────────────────────┤
│ 10-layer GATv2Conv + Residual Connections                   │
│ ノード: 16次元、エッジ: 3次元                               │
│ RTX 5070 Ti (16GB GDDR7) × 約40時間（高速化）                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: 評価と反復改善判断（NEW!）                         │
├─────────────────────────────────────────────────────────────┤
│ - Cosine Similarity評価                                     │
│ - Top-K Recall評価                                          │
│ - 汎化性能評価（未知化合物テスト）                          │
│ - Attention weights分析                                     │
│ → 判定: 十分 or 特徴量追加必要                              │
└─────────────────────────────────────────────────────────────┘
```

---

## アーキテクチャ設計

### BDE計算バックエンド: BonDNet (BDE-db2再学習版)

（v4.1と同じ - 変更なし）

---

### GNNアーキテクチャ: 10-layer GATv2Conv（最小構成版）

#### モデル構成

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class QCGN2oEI_Minimal(nn.Module):
    """
    QC-GN2oMS2 Architecture for EI-MS Prediction (Minimal Configuration)

    Key design:
    - Minimal feature dimensions (16 node, 3 edge)
    - Inspired by QC-GN2oMS2's proven approach
    - Iterate based on performance evaluation
    """

    def __init__(
        self,
        node_dim: int = 16,        # Minimal node feature dimension
        edge_dim: int = 3,         # Minimal edge feature dimension
        hidden_dim: int = 256,     # Hidden layer dimension
        num_layers: int = 10,      # GATv2Conv layers
        num_heads: int = 8,        # Attention heads
        output_dim: int = 1000,    # Output spectrum bins (m/z 50-1000)
        dropout: float = 0.1
    ):
        super().__init__()

        # Node embedding (16 → 256)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Edge embedding (3 → 256)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 10-layer GATv2Conv with residual connections
        self.gat_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()

        for i in range(num_layers):
            # GATv2Conv layer
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,  # Edge features
                    dropout=dropout,
                    concat=True,          # Concatenate heads
                    residual=True         # PyG 2.6.1+ feature
                )
            )

            # Residual connection projection
            self.residual_layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Global pooling + prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)  # Normalize to intensity distribution
        )

    def forward(self, data):
        """
        Args:
            data: PyG Data object
                - x: Node features [num_nodes, 16]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, 3]
                - batch: Batch assignment [num_nodes]

        Returns:
            spectrum: Predicted intensity [batch_size, 1000]
        """
        # Encode nodes and edges
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        # 10-layer GATv2Conv with residual connections
        for gat, residual in zip(self.gat_layers, self.residual_layers):
            x_res = residual(x)  # Residual projection
            x = gat(x, data.edge_index, edge_attr)
            x = x + x_res  # Residual addition
            x = torch.relu(x)

        # Global mean pooling
        x = global_mean_pool(x, data.batch)

        # Predict spectrum
        spectrum = self.prediction_head(x)

        return spectrum
```

#### ノード特徴量（16次元）- QC-GN2oMS2準拠

**設計方針**: QC-GN2oMS2が実証した最小限の特徴量セット

| カテゴリ | 次元 | 内容 | 理由 |
|---------|------|------|------|
| **原子種** | 10 | C, H, O, N, F, S, P, Cl, Br, I (one-hot) | 元素種は最重要特徴 |
| **芳香族性** | 1 | Binary (aromatic/aliphatic) | フラグメンテーション安定性 |
| **環構造** | 1 | Binary (in ring/not in ring) | 構造的安定性 |
| **ハイブリダイゼーション** | 3 | SP/SP2/SP3 (one-hot) | 結合の性質 |
| **部分電荷** | 1 | Gasteiger charge (continuous) | 電子分布 |
| **合計** | **16** | - | **予備なし** |

**削除された特徴（v4.1にあったもの）**:
- 形式電荷（3次元） → 部分電荷で代替可能
- 水素結合数（5次元） → 原子種+ハイブリダイゼーションから推測可能
- 次数（6次元） → グラフ構造から学習可能
- ラジカル電子（3次元） → EI-MSではあまり重要でない可能性
- キラリティ（3次元） → 立体化学は二次的
- 原子量・vdW半径・電気陰性度（3次元） → 原子種と相関

**実装例**:
```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

SUPPORTED_ELEMENTS = ['C', 'H', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Br', 'I']

def get_atom_features_minimal(atom: Chem.Atom) -> np.ndarray:
    """
    Extract 16-dimensional minimal atom features

    Inspired by QC-GN2oMS2's proven approach
    """

    # 1. Atom type (10-dim one-hot)
    atom_symbol = atom.GetSymbol()
    if atom_symbol not in SUPPORTED_ELEMENTS:
        raise ValueError(f"Unsupported element: {atom_symbol}")
    atom_type = one_hot(atom_symbol, SUPPORTED_ELEMENTS)  # 10 dims

    # 2. Aromatic (1-dim binary)
    aromatic = [int(atom.GetIsAromatic())]  # 1 dim

    # 3. In ring (1-dim binary)
    in_ring = [int(atom.IsInRing())]  # 1 dim

    # 4. Hybridization (3-dim one-hot: SP/SP2/SP3)
    hyb = atom.GetHybridization()
    if hyb == Chem.HybridizationType.SP:
        hybrid = [1, 0, 0]
    elif hyb == Chem.HybridizationType.SP2:
        hybrid = [0, 1, 0]
    elif hyb == Chem.HybridizationType.SP3:
        hybrid = [0, 0, 1]
    else:
        hybrid = [0, 0, 1]  # Default to SP3 for SP3D, etc.
    # 3 dims

    # 5. Partial charge (1-dim continuous)
    if atom.HasProp('_GasteigerCharge'):
        partial_charge = [atom.GetDoubleProp('_GasteigerCharge')]
    else:
        partial_charge = [0.0]
    # 1 dim

    # Concatenate: 10 + 1 + 1 + 3 + 1 = 16 dims
    features = np.concatenate([
        atom_type,      # 10
        aromatic,       # 1
        in_ring,        # 1
        hybrid,         # 3
        partial_charge  # 1
    ])

    assert len(features) == 16, f"Feature length mismatch: {len(features)}"

    return features

def one_hot(value, choices):
    """One-hot encoding"""
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding
```

#### エッジ特徴量（3次元）- 最小構成

**設計方針**: BDE + 結合情報の最小セット

| カテゴリ | 次元 | 内容 | 理由 |
|---------|------|------|------|
| **BDE（最重要）** | 1 | Bond Dissociation Energy from BonDNet (normalized) | フラグメンテーション確率の主要因子 |
| **結合次数** | 1 | Bond order (1.0, 2.0, 3.0, 1.5 for aromatic) | 結合の強さ |
| **環内結合** | 1 | Binary (in ring/not in ring) | 環の安定性 |
| **合計** | **3** | - | **予備なし** |

**削除された特徴（v4.1にあったもの）**:
- 結合次数one-hot（4次元） → 連続値1次元で代替
- 共役（1次元） → 芳香族性・環構造から推測可能
- 立体化学（3次元） → EI-MSではあまり重要でない
- 回転可能性（1次元） → フラグメンテーションへの影響小
- 結合距離（1次元） → 結合次数と相関

**実装例**:
```python
def get_bond_features_minimal(bond: Chem.Bond, bde_value: float) -> np.ndarray:
    """
    Extract 3-dimensional minimal bond features
    """

    # 1. BDE (normalized, 1-dim)
    bde_normalized = normalize_bde(bde_value)  # [0, 1] range

    # 2. Bond order (continuous, 1-dim)
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        bond_order = 1.0
    elif bond_type == Chem.BondType.DOUBLE:
        bond_order = 2.0
    elif bond_type == Chem.BondType.TRIPLE:
        bond_order = 3.0
    elif bond_type == Chem.BondType.AROMATIC:
        bond_order = 1.5
    else:
        bond_order = 1.0  # Default

    # 3. In ring (binary, 1-dim)
    in_ring = float(bond.IsInRing())

    # Concatenate: 1 + 1 + 1 = 3 dims
    features = np.array([bde_normalized, bond_order, in_ring])

    assert len(features) == 3, f"Feature length mismatch: {len(features)}"

    return features

def normalize_bde(bde_kcal_mol: float) -> float:
    """Normalize BDE to [0, 1] range"""
    return (bde_kcal_mol - 50.0) / 150.0  # 50-200 kcal/mol range
```

---

## Phase 0: BDE-db2環境構築

（v4.1と同じ内容 - 変更なし）

---

## Phase 1: データ準備

### 1.1 NIST 17データ読み込み

（v4.1と同じ内容 - 変更なし）

### 1.2 データフィルタリング

（v4.1と同じ内容 - 変更なし）

### 1.3 BDE前計算（BonDNet BDE-db2）

（v4.1と同じ内容 - 変更なし）

### 1.4 PyG Graph生成（最小構成版）

```python
# src/data/graph_generator.py
"""
PyTorch Geometric Graph Generator (Minimal Configuration)
"""

import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from typing import Dict, List
import h5py

class GraphGeneratorMinimal:
    """Generate PyTorch Geometric graphs with minimal features (16 node, 3 edge)"""

    def __init__(self, bde_cache_path: str = "data/processed/bde_cache.h5"):
        self.bde_cache = h5py.File(bde_cache_path, 'r')

    def smiles_to_graph(
        self,
        smiles: str,
        spectrum: np.ndarray,
        molecule_idx: int
    ) -> Data:
        """
        Convert SMILES to PyG Data object with minimal features

        Args:
            smiles: SMILES string
            spectrum: Target spectrum [1000]
            molecule_idx: Index for BDE cache lookup

        Returns:
            PyG Data object with 16-dim nodes and 3-dim edges
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogens for complete graph
        mol = Chem.AddHs(mol)

        # Compute Gasteiger charges (needed for partial charge feature)
        AllChem.ComputeGasteigerCharges(mol)

        # Get BDE values from cache
        bde_dict = {}
        if str(molecule_idx) in self.bde_cache:
            grp = self.bde_cache[str(molecule_idx)]
            for bond_idx in grp.keys():
                bde_dict[int(bond_idx)] = float(grp[bond_idx][()])

        # Node features (16 dims per atom)
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(get_atom_features_minimal(atom))

        x = torch.tensor(node_features, dtype=torch.float)

        # Edge features (3 dims per bond)
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_idx = bond.GetIdx()

            # Get BDE value
            bde_value = bde_dict.get(bond_idx, 100.0)  # Default if not in cache

            # Bidirectional edges
            edge_index.append([i, j])
            edge_index.append([j, i])

            bond_features = get_bond_features_minimal(bond, bde_value)
            edge_attr.append(bond_features)
            edge_attr.append(bond_features)  # Same features for both directions

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Target spectrum
        y = torch.tensor(spectrum, dtype=torch.float)

        # Create PyG Data
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            smiles=smiles
        )

        return data
```

---

## Phase 2: GNN学習

### 2.1 学習スクリプト（最小構成版）

```python
# scripts/train_gnn_minimal.py
"""
Train QC-GN2oEI model (Minimal Configuration)
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal
import wandb
import yaml
from pathlib import Path
from tqdm import tqdm

def cosine_similarity_loss(pred, target):
    """Cosine Similarity Loss (same as QC-GN2oMS2)"""
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    cosine_sim = (pred_norm * target_norm).sum(dim=1)
    return (1 - cosine_sim).mean()

def train_qcgn2oei_minimal(config_path: str = "config/training_minimal.yml"):
    """Train QC-GN2oEI model with minimal features"""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(project="qcgn2oei-minimal", config=config)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loaders
    train_data = torch.load("data/processed/nist17_train.pt")
    val_data = torch.load("data/processed/nist17_val.pt")

    train_loader = DataLoader(
        train_data,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Model (minimal configuration)
    model = QCGN2oEI_Minimal(
        node_dim=16,      # Minimal
        edge_dim=3,       # Minimal
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Node features: 16 dims (minimal)")
    print(f"Edge features: 3 dims (minimal)")

    # Optimizer (RAdam from QC-GN2oMS2)
    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=1e-6
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config['training']['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            batch = batch.to(device)

            optimizer.zero_grad()
            pred = model(batch)
            loss = cosine_similarity_loss(pred, batch.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                batch = batch.to(device)
                pred = model(batch)
                loss = cosine_similarity_loss(pred, batch.y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Scheduler step
        scheduler.step()

        # Logging
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, "models/qcgn2oei_minimal_best.pth")
            print(f"✅ Best model saved (Val Loss: {val_loss:.4f})")

    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_qcgn2oei_minimal()
```

### 2.2 設定ファイル（最小構成版）

```yaml
# config/training_minimal.yml

model:
  node_dim: 16      # Minimal (QC-GN2oMS2-inspired)
  edge_dim: 3       # Minimal (BDE + bond order + in ring)
  hidden_dim: 256
  num_layers: 10
  num_heads: 8
  output_dim: 1000
  dropout: 0.1

training:
  num_epochs: 300
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 1e-5
  early_stopping_patience: 50

data:
  bde_cache: "data/processed/bde_cache.h5"
  train_data: "data/processed/nist17_train.pt"
  val_data: "data/processed/nist17_val.pt"
  test_data: "data/processed/nist17_test.pt"
```

### 2.3 学習時間見積もり（高速化）

**パラメータ数の比較**:

| 項目 | v4.1 (128/64) | v4.2 (16/3) | 削減率 |
|------|--------------|------------|--------|
| Node encoder | 128×256 = 32,768 | 16×256 = 4,096 | **-87.5%** |
| Edge encoder | 64×256 = 16,384 | 3×256 = 768 | **-95.3%** |
| Encoder合計 | 49,152 | 4,864 | **-90.1%** |

**1エポックの時間（推定）**:
```
224,000 samples (train) ÷ 32 batch_size = 7,000 iterations
7,000 iterations × 0.7 sec/iter = 4,900 sec = 1.36 hours
（v4.1: 1.56時間 → v4.2: 1.36時間、約13%高速化）
```

**合計学習時間（推定）**:
```
300 epochs × 1.36 hours = 408 hours
→ early stoppingで約40時間（30エポック程度で収束想定）
（v4.1: 48時間 → v4.2: 40時間、約17%高速化）
```

---

## Phase 3: 評価と反復改善判断

### 3.1 評価メトリクス

```python
# scripts/evaluate_minimal.py
"""
Comprehensive evaluation for minimal configuration model
"""

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def cosine_similarity_metric(pred, target):
    """Calculate cosine similarity"""
    pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    return (pred_norm * target_norm).sum(axis=1).mean()

def top_k_recall(pred, target, k=10):
    """Top-K Recall"""
    recalls = []
    for p, t in zip(pred, target):
        true_top_k = set(np.argsort(t)[-k:])
        pred_top_k = set(np.argsort(p)[-k:])
        recall = len(true_top_k & pred_top_k) / k
        recalls.append(recall)
    return np.mean(recalls)

def evaluate_model(
    model_path: str = "models/qcgn2oei_minimal_best.pth",
    test_data_path: str = "data/processed/nist17_test.pt"
):
    """Comprehensive model evaluation"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(model_path)
    model = QCGN2oEI_Minimal(
        node_dim=16,
        edge_dim=3,
        hidden_dim=256,
        num_layers=10,
        num_heads=8,
        output_dim=1000,
        dropout=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load test data
    test_data = torch.load(test_data_path)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Inference
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Metrics
    cosine_sim = cosine_similarity_metric(predictions, targets)
    top5_recall = top_k_recall(predictions, targets, k=5)
    top10_recall = top_k_recall(predictions, targets, k=10)
    top20_recall = top_k_recall(predictions, targets, k=20)

    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)

    print("=" * 60)
    print("QC-GN2oEI Minimal Configuration Evaluation")
    print("=" * 60)
    print(f"Node features: 16 dims")
    print(f"Edge features: 3 dims")
    print("-" * 60)
    print(f"Cosine Similarity: {cosine_sim:.4f}")
    print(f"Top-5 Recall:      {top5_recall:.4f}")
    print(f"Top-10 Recall:     {top10_recall:.4f}")
    print(f"Top-20 Recall:     {top20_recall:.4f}")
    print(f"MSE:               {mse:.6f}")
    print(f"RMSE:              {rmse:.6f}")
    print("=" * 60)

    # Decision logic
    print("\n" + "=" * 60)
    print("Performance Assessment")
    print("=" * 60)

    if cosine_sim >= 0.85:
        print("✅ EXCELLENT: Cosine Similarity >= 0.85")
        print("   Recommendation: Adopt v4.2 minimal configuration!")
        print("   No feature expansion needed.")
    elif cosine_sim >= 0.80:
        print("⚠️  GOOD: Cosine Similarity 0.80-0.85")
        print("   Recommendation: Consider minor feature additions")
        print("   Proceed to Phase 4 for targeted feature expansion")
    else:
        print("❌ INSUFFICIENT: Cosine Similarity < 0.80")
        print("   Recommendation: Feature expansion required")
        print("   Proceed to Phase 4 for systematic feature addition")

    return {
        'cosine_similarity': cosine_sim,
        'top5_recall': top5_recall,
        'top10_recall': top10_recall,
        'top20_recall': top20_recall,
        'mse': mse,
        'rmse': rmse
    }

if __name__ == "__main__":
    results = evaluate_model()
```

### 3.2 判定基準とアクションプラン

| Cosine Similarity | 判定 | アクション |
|------------------|------|-----------|
| **≥ 0.85** | ✅ 優秀 | **v4.2採用完了！** 特徴量拡張不要 |
| **0.80 - 0.85** | ⚠️ 良好 | 軽微な改善検討 → Phase 4へ |
| **0.75 - 0.80** | ⚠️ 要改善 | 特徴量追加必須 → Phase 4へ |
| **< 0.75** | ❌ 不十分 | 中間構成(64/32)検討 → Phase 4へ |

---

## Phase 4: 特徴量拡張（条件付き）

### 4.1 特徴量重要度分析

**Phase 3でCosine Sim < 0.85の場合のみ実施**

```python
# scripts/analyze_feature_importance.py
"""
Analyze which features should be added next
"""

import torch
from src.models.qcgn2oei_minimal import QCGN2oEI_Minimal
import numpy as np

def analyze_attention_weights(
    model_path: str = "models/qcgn2oei_minimal_best.pth"
):
    """
    Analyze GATv2 attention weights to understand feature importance
    """

    # Load model
    model = QCGN2oEI_Minimal(...)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Hook to capture attention weights
    attention_weights = []

    def hook_fn(module, input, output):
        # GATv2Conv returns (output, attention_weights)
        if isinstance(output, tuple):
            attention_weights.append(output[1].detach().cpu().numpy())

    # Register hooks
    for layer in model.gat_layers:
        layer.register_forward_hook(hook_fn)

    # Run inference on validation set
    # ... (collect attention weights)

    # Analyze patterns
    print("Attention Weight Analysis")
    print("=" * 60)
    # TODO: Implement analysis

    return attention_weights

def propose_feature_additions(cosine_sim: float):
    """
    Propose which features to add based on performance
    """

    print("\n" + "=" * 60)
    print("Feature Addition Recommendations")
    print("=" * 60)

    if cosine_sim >= 0.80 and cosine_sim < 0.85:
        print("Performance: GOOD (0.80-0.85)")
        print("\nRecommended additions (Priority 1):")
        print("  1. Formal charge (3 dims) - for ionic fragments")
        print("  2. Degree (6 dims) - for branching patterns")
        print("Total: +9 dims → 16+9 = 25 node dims")

    elif cosine_sim >= 0.75 and cosine_sim < 0.80:
        print("Performance: MODERATE (0.75-0.80)")
        print("\nRecommended additions (Priority 1+2):")
        print("  Priority 1:")
        print("    - Formal charge (3 dims)")
        print("    - Degree (6 dims)")
        print("  Priority 2:")
        print("    - Hydrogen count (5 dims)")
        print("    - Conjugated bonds (1 edge dim)")
        print("Total: +14 node dims, +1 edge dim")
        print("  → 16+14 = 30 node dims, 3+1 = 4 edge dims")

    else:  # < 0.75
        print("Performance: INSUFFICIENT (<0.75)")
        print("\nRecommended: Move to intermediate configuration")
        print("  Node: 64 dims (41 used + 23 reserved)")
        print("  Edge: 32 dims (12 used + 20 reserved)")
        print("  See v4.3 specification for details")

if __name__ == "__main__":
    # Run after Phase 3 evaluation
    results = evaluate_model()
    cosine_sim = results['cosine_similarity']

    analyze_attention_weights()
    propose_feature_additions(cosine_sim)
```

### 4.2 段階的拡張フローチャート

```
Phase 3評価結果
     ↓
┌────────────────────────────────────────┐
│ Cosine Similarity = ?                  │
└────────────────────────────────────────┘
     ↓
     ├─ ≥ 0.85 → ✅ 完了！v4.2採用
     │
     ├─ 0.80-0.85 → v4.3 (軽微拡張)
     │                ├─ ノード: 16 → 25 (+9)
     │                └─ 再評価 → 完了 or さらに拡張
     │
     ├─ 0.75-0.80 → v4.3 (中度拡張)
     │                ├─ ノード: 16 → 30 (+14)
     │                ├─ エッジ: 3 → 4 (+1)
     │                └─ 再評価 → 完了 or さらに拡張
     │
     └─ < 0.75 → v4.3 (中間構成)
                    ├─ ノード: 16 → 64 (+48)
                    ├─ エッジ: 3 → 32 (+29)
                    └─ 再評価
```

---

## 設定ファイル詳細

### config.yml（メイン設定、v4.2最小構成版）

```yaml
# config.yml - Main Configuration (v4.2 Minimal)

project:
  name: "QC-GN2oEI"
  version: "2.2-minimal"
  description: "Minimal configuration approach with iterative refinement"
  design_philosophy: "Start simple, iterate based on evidence"

# BDE Configuration
bde:
  backend: "bondnet"
  bondnet:
    model_type: "bde-db2"
    model_path: "models/bondnet_bde_db2_best.pth"
    dataset_path: "data/external/bde-db2"
    device: "cuda"
    batch_size: 256

# Data paths
data:
  nist17_path: "data/external/nist17/mainlib"
  bde_cache: "data/processed/bde_cache.h5"
  train_data: "data/processed/nist17_train.pt"
  val_data: "data/processed/nist17_val.pt"
  test_data: "data/processed/nist17_test.pt"

  # Data filtering
  filtering:
    supported_elements: ['C', 'H', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Br', 'I']
    min_molecular_weight: 50.0
    max_molecular_weight: 1000.0
    validate_smiles: true

# Model architecture (MINIMAL CONFIGURATION)
model:
  type: "QCGN2oEI_Minimal"

  # Minimal feature dimensions (QC-GN2oMS2-inspired)
  node_dim: 16   # No reserved dims
  edge_dim: 3    # No reserved dims

  # GNN layers
  hidden_dim: 256
  num_layers: 10
  num_heads: 8

  # Output
  output_dim: 1000

  # Regularization
  dropout: 0.1

  # Advanced features
  use_residual: true
  use_edge_features: true
  global_pooling: "mean"

# Training
training:
  num_epochs: 300
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 1e-5

  optimizer: "RAdam"
  scheduler: "CosineAnnealingLR"
  scheduler_params:
    T_max: 300
    eta_min: 1e-6

  loss: "cosine_similarity"
  early_stopping_patience: 50

  checkpoint_dir: "checkpoints"
  save_every: 10

# Evaluation & Iteration
evaluation:
  metrics:
    - "cosine_similarity"
    - "top_k_recall"
    - "mse"
    - "rmse"

  top_k_values: [5, 10, 20, 50]

  # Performance thresholds for iteration decision
  performance_thresholds:
    excellent: 0.85      # No feature expansion needed
    good: 0.80           # Minor additions considered
    moderate: 0.75       # Feature additions recommended
    insufficient: 0.0    # Significant expansion required

  # Feature expansion plan (conditional)
  feature_expansion:
    enabled: true
    analyze_attention: true
    ablation_study: true

# Hardware
hardware:
  device: "cuda"
  gpu_id: 0
  num_workers: 4
  pin_memory: true
  use_amp: true
  amp_dtype: "float16"

# Logging
logging:
  use_wandb: true
  wandb_project: "qcgn2oei-minimal"
  wandb_entity: null
  log_every: 10
  save_predictions: true

# Reproducibility
seed: 42
deterministic: true
```

---

## 開発環境構築

（v4.1と同じ内容 - 変更なし）

---

## タイムライン

### 全体スケジュール（v4.2更新版）

| フェーズ | タスク | 推定時間 | 累積時間 |
|---------|--------|---------|---------|
| **Phase 0** | BDE-db2ダウンロード | 30分 | 30分 |
| **Phase 0** | データ前処理 | 3時間 | 3.5時間 |
| **Phase 0** | BonDNet再学習 | 48-72時間 | 51.5-75.5時間 |
| **Phase 0** | モデル検証 | 1時間 | 52.5-76.5時間 |
| **Phase 1** | NIST読み込み | 30分 | 53-77時間 |
| **Phase 1** | データフィルタリング | 10分 | 53.17-77.17時間 |
| **Phase 1** | BDE計算（280K） | 70分 | 54.33-78.33時間 |
| **Phase 1** | PyG Graph生成（16/3次元） | 60分 | 55.33-79.33時間 |
| **Phase 2** | GNN学習（最小構成） | **40時間** | **95.33-119.33時間** |
| **Phase 3** | 評価・判定 | 2時間 | 97.33-121.33時間 |
| **Phase 4** | 特徴量拡張（条件付き） | 0-24時間 | 97.33-145.33時間 |
| **合計** | - | **97-145時間** | **4.0-6.0日** |

**v4.2での変更**:
- GNN学習時間: 48時間 → 40時間（-17%、高速化）
- Phase 4追加（条件付き）: 性能不足時のみ実施

---

## 参考文献

（v4.1と同じ内容 - 変更なし）

---

## まとめ

### v4.2の主要な改善点

1. **最小構成アプローチ**: QC-GN2oMS2の実証済み設計に準拠
2. **メモリ効率**: v4.1比で88%削減（1.3GB → 0.16GB）
3. **学習高速化**: v4.1比で17%高速化（48時間 → 40時間）
4. **反復改善戦略**: 性能評価 → 必要に応じて段階的拡張
5. **実証主義**: "Start simple, iterate based on evidence"

### 期待される成果

**ベストケース（Cosine Sim ≥ 0.85）**:
- ✅ v4.2採用完了（最小構成で十分）
- メモリ効率・学習速度の大幅改善
- QC-GN2oMS2の成功を再現

**中間ケース（Cosine Sim 0.80-0.85）**:
- v4.3で軽微な特徴追加（+9-14次元）
- 合理的なトレードオフ

**最悪ケース（Cosine Sim < 0.75）**:
- 中間構成（64/32次元）に拡張
- それでもv4.1より効率的

---

**Document Version**: 4.2
**Last Updated**: 2025-12-02
**Status**: Ready for Implementation (Minimal Configuration with Iterative Refinement)
**Design Philosophy**: Start Simple, Iterate Based on Evidence
