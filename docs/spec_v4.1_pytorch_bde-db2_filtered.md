# QC-GN2oMS2-EI システム詳細技術仕様書 v4.1
## PyTorch統一環境・BonDNet BDE-db2デフォルト版（データフィルタリング追加）

**作成日**: 2025-12-02
**対象システム**: NExtIMS (NIST EI-MS Prediction System)
**基盤アーキテクチャ**: QC-GN2oMS2 (PNNL)
**ハードウェア**: NVIDIA GeForce RTX 5070 Ti (Blackwell sm_120)

---

## 📋 目次

1. [主要変更点（v4.0 → v4.1）](#主要変更点v40--v41)
2. [システム概要](#システム概要)
3. [アーキテクチャ設計](#アーキテクチャ設計)
4. [Phase 0: BDE-db2環境構築](#phase-0-bde-db2環境構築)
5. [Phase 1: データ準備](#phase-1-データ準備)
6. [Phase 2: GNN学習](#phase-2-gnn学習)
7. [Phase 3: 評価](#phase-3-評価)
8. [設定ファイル詳細](#設定ファイル詳細)
9. [開発環境構築](#開発環境構築)
10. [タイムライン](#タイムライン)
11. [参考文献](#参考文献)

---

## 主要変更点（v4.0 → v4.1）

### ✅ v4.1での追加・修正

| 項目 | 詳細 |
|------|------|
| **データフィルタリングの明示化** | Phase 1にサポート元素・分子量によるフィルタリングステップを追加 |
| **サポート元素の厳密化** | C, H, O, N, F, S, P, Cl, Br, I以外を含む化合物を除外（-5%） |
| **分子量上限の設定** | MW <= 1000 Daに限定し、出力範囲（m/z 50-1000）と整合 |
| **最終データセットサイズの修正** | 300,000 → 280,000 spectra（93.3% retention） |
| **予備次元の明確化** | 将来の拡張性のための設計であることを説明 |

### v4.0からの主要変更（継続）

| 項目 | 詳細 |
|------|------|
| **BonDNet BDE-db2デフォルト化** | 531,244件のBDEデータで再学習したBonDNetをデフォルトバックエンドに設定 |
| **Pure PyTorch環境** | TensorFlow依存を完全削除。PyTorch 2.10.0+ nightly (cu128) のみ使用 |
| **Phase 0の追加** | BDE-db2ダウンロード→BonDNet再学習をデータ準備前の必須フェーズとして追加 |
| **ハロゲン・硫黄・リン対応** | BDE-db2により10元素（C,H,O,N,F,S,P,Cl,Br,I）をサポート |

---

## システム概要

### 目的

NIST 17 EI-MSデータベース（約280,000スペクトル、フィルタリング後）を用いて、**物理化学的に解釈可能なGraph Neural Network**によるEI-MSスペクトル予測システムを構築する。

### 基盤技術

**QC-GN2oMS2**（PNNL, 2024）:
- 論文: "Quantum Chemistry-Informed Graph Neural Network for Mass Spectrum Prediction"
- 特徴: 量子化学計算（BDE）をエッジ特徴量として使用
- 元の対象: MS/MS（タンデム質量分析）
- **本プロジェクトでの適用**: EI-MS（電子イオン化質量分析、70eV固定）

### アーキテクチャ概要図

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 0: BDE-db2環境構築（必須前処理）                        │
├─────────────────────────────────────────────────────────────┤
│ 1. BDE-db2ダウンロード (531,244 reactions)                   │
│ 2. BonDNet再学習 (2-3日, RTX 5070 Ti)                        │
│ 3. 学習済みモデル検証 (MAE < 1.0 kcal/mol目標)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: データ準備                                          │
├─────────────────────────────────────────────────────────────┤
│ 1.1 NIST 17読み込み (300,000 spectra)                       │
│ 1.2 データフィルタリング (NEW!)                              │
│     - サポート元素チェック (C,H,O,N,F,S,P,Cl,Br,I)          │
│     - 分子量フィルタ (MW <= 1000 Da)                         │
│     → 280,000 spectra (93.3% retention)                     │
│ 1.3 BonDNet BDE計算 (70 min)                                │
│ 1.4 PyG Graph生成                                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: GNN学習                                             │
├─────────────────────────────────────────────────────────────┤
│ 10-layer GATv2Conv + Residual Connections                   │
│ RTX 5070 Ti (16GB GDDR7) × 約48時間                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: 評価                                                │
├─────────────────────────────────────────────────────────────┤
│ Cosine Similarity, Top-10 Recall, Physical Interpretability │
└─────────────────────────────────────────────────────────────┘
```

---

## アーキテクチャ設計

### BDE計算バックエンド: BonDNet (BDE-db2再学習版)

#### 選定理由

| 基準 | BonDNet (BDE-db2) | ALFABET | xTB GPU |
|------|------------------|---------|---------|
| **フレームワーク** | PyTorch ✅ | TensorFlow ❌ | Fortran (nvfortran) |
| **速度** | 15ms/分子 ✅ | 5ms/分子 | 1.5秒/分子 ❌ |
| **精度** | MAE 0.51 kcal/mol ✅ | MAE 0.45 kcal/mol | MAE 3-5 kcal/mol ❌ |
| **対応元素** | 10元素 (BDE-db2) ✅ | 6元素 | 全元素 |
| **BDE直接計算** | 可能 ✅ | 可能 | 不可（エネルギー差分法のみ）❌ |
| **環境統一性** | PyTorch統一 ✅ | TF/PyTorch混在 ❌ | 別プロセス起動 |
| **学習コスト** | 2-3日 (初回のみ) | 学習済み | N/A |

#### サポート元素（10元素、厳密）

**C, H, O, N, F, S, P, Cl, Br, I**

**これら以外の元素を含む化合物は学習・評価から除外**

理由:
1. BonDNet BDE-db2は10元素でのみ学習済み
2. ノード特徴量のone-hotエンコーディングが10次元（10元素専用）
3. サポート外元素のBDE予測が不正確（ルールベース推定に依存）
4. 誤った特徴量による学習を回避

**NIST 17でのサポート外元素の例**:
- Si（シリコン化合物、シロキサン）
- B（ホウ素化合物）
- Se（セレン化合物）
- As（ヒ素化合物）
- Ge, Al, Ti, Zr, etc.

**除外される化合物**: 約15,000スペクトル（5%）

#### BDE-db2データセット詳細

**Paton Group BDE-db2**:
- 総データ数: **531,244 BDE値**
- 元素種: C, H, O, N, F, S, P, Cl, Br, I（10元素）
- データソース: B3LYP/6-31G(d) DFT計算
- 論文: "A comprehensive database of bond dissociation enthalpies" (Paton et al.)

**BDNCM（BonDNet公式）との比較**:
| 項目 | BDNCM (公式) | BDE-db2 (本プロジェクト) |
|------|-------------|------------------------|
| データ数 | 64,312 | 531,244 |
| 元素数 | 5 (C,H,O,F,Li) | 10 (C,H,O,N,F,S,P,Cl,Br,I) |
| 用途 | 有機リチウム電池 | 汎用有機化合物 ✅ |
| ハロゲン対応 | Fのみ | Cl, Br, I対応 ✅ |

**フィルタリング後のNIST 17との適合性**:
- フィルタリング後のNIST 17の**100%**が10元素内に収まる（設計上保証）
- 環状化合物、ヘテロ環化合物の多様性に対応
- ハロゲン化合物（農薬、医薬品）のカバレッジ向上

---

### GNNアーキテクチャ: 10-layer GATv2Conv

#### モデル構成

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class QCGN2oEI(nn.Module):
    """
    QC-GN2oMS2 Architecture for EI-MS Prediction

    Key changes from original:
    - MS/MS → EI-MS (fragmentation energy: variable → 70eV fixed)
    - Edge features: BDE from BonDNet (BDE-db2 retrained)
    - Output: 1000-bin intensity distribution (m/z 50-1000)
    - Input: Filtered dataset (10 elements, MW <= 1000)
    """

    def __init__(
        self,
        node_dim: int = 128,       # Atom feature dimension
        edge_dim: int = 64,        # Edge feature dimension (includes BDE)
        hidden_dim: int = 256,     # Hidden layer dimension
        num_layers: int = 10,      # GATv2Conv layers
        num_heads: int = 8,        # Attention heads
        output_dim: int = 1000,    # Output spectrum bins (m/z 50-1000, 950 bins + 50 padding)
        dropout: float = 0.1
    ):
        super().__init__()

        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Edge embedding (BDE + bond features)
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
                - x: Node features [num_nodes, node_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features (includes BDE) [num_edges, edge_dim]
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

#### ノード特徴量（128次元）

**実使用**: 41次元
**予備**: 87次元（将来の拡張用、現在はゼロパディング）

| カテゴリ | 次元 | 内容 |
|---------|------|------|
| **原子種** | 10 | C, H, O, N, F, S, P, Cl, Br, I (one-hot) |
| **ハイブリダイゼーション** | 5 | SP, SP2, SP3, SP3D, SP3D2 (one-hot) |
| **形式電荷** | 3 | -1, 0, +1 (one-hot) |
| **芳香族性** | 1 | Binary (aromatic/aliphatic) |
| **環構造** | 1 | Binary (in ring/not in ring) |
| **水素結合数** | 5 | 0, 1, 2, 3, 4+ (one-hot) |
| **次数（degree）** | 6 | 0, 1, 2, 3, 4, 5+ (one-hot) |
| **ラジカル電子** | 3 | 0, 1, 2 (one-hot) |
| **キラリティ** | 3 | None, R, S (one-hot) |
| **部分電荷** | 1 | Gasteiger charge (continuous) |
| **原子量** | 1 | Normalized atomic mass (continuous) |
| **ファン der Waals半径** | 1 | Normalized vdW radius (continuous) |
| **電気陰性度** | 1 | Pauling electronegativity (continuous) |
| **予備** | 87 | 将来の拡張用（Morgan fingerprint、QM記述子、グラフ埋め込みなど） |

**予備次元の設計意図**:
- 将来のモデル改善時に特徴量を追加可能（後方互換性維持）
- メモリアライメント最適化（2の累乗次元: 128 = 2^7）
- RTX 5070 Ti（16GB）では280,000グラフでも約1.3GBと許容範囲内
- Encoder層は1層のみなので計算オーバーヘッド最小

**注意**: データフィルタリングにより、サポート外元素は入力されない（原子種10次元で完全カバー）

**実装例**:
```python
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np

# サポート元素（厳密に10元素のみ）
SUPPORTED_ELEMENTS = ['C', 'H', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Br', 'I']

def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """Extract 128-dimensional atom features"""

    # Atom type (10-dim one-hot) - サポート元素のみ
    atom_symbol = atom.GetSymbol()
    if atom_symbol not in SUPPORTED_ELEMENTS:
        raise ValueError(f"Unsupported element: {atom_symbol}. "
                         "This should have been filtered in Phase 1.2")
    atom_type = one_hot(atom_symbol, SUPPORTED_ELEMENTS)

    # Hybridization (5-dim one-hot)
    hybridizations = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2
    ]
    hybrid = one_hot(atom.GetHybridization(), hybridizations)

    # Formal charge (3-dim one-hot)
    charge = one_hot(atom.GetFormalCharge(), [-1, 0, 1])

    # Binary features
    aromatic = [int(atom.GetIsAromatic())]
    in_ring = [int(atom.IsInRing())]

    # Hydrogen count (5-dim one-hot)
    num_h = one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    # Degree (6-dim one-hot)
    degree = one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])

    # Radical electrons (3-dim one-hot)
    radical = one_hot(atom.GetNumRadicalElectrons(), [0, 1, 2])

    # Chirality (3-dim one-hot)
    chiralities = [
        Chem.ChiralType.CHI_UNSPECIFIED,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    ]
    chirality = one_hot(atom.GetChiralTag(), chiralities)

    # Continuous features
    partial_charge = [atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0]
    atomic_mass = [atom.GetMass() / 100.0]  # Normalize
    vdw_radius = [Chem.GetPeriodicTable().GetRvdw(atom.GetSymbol()) / 2.0]  # Normalize
    electronegativity = [Chem.GetPeriodicTable().GetElectronegativity(atom.GetSymbol()) / 4.0]  # Normalize

    # Concatenate (total: 10+5+3+1+1+5+6+3+3+1+1+1+1 = 41 dims)
    # Pad to 128 with zeros (87 reserved dimensions)
    features = np.concatenate([
        atom_type, hybrid, charge, aromatic, in_ring,
        num_h, degree, radical, chirality,
        partial_charge, atomic_mass, vdw_radius, electronegativity
    ])

    padded = np.zeros(128)
    padded[:len(features)] = features

    return padded

def one_hot(value, choices):
    """One-hot encoding with out-of-vocabulary handling"""
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding
```

#### エッジ特徴量（64次元）

**実使用**: 12次元
**予備**: 52次元（将来の拡張用、現在はゼロパディング）

| カテゴリ | 次元 | 内容 |
|---------|------|------|
| **BDE（重要）** | 1 | Bond Dissociation Energy from BonDNet (kcal/mol, normalized) |
| **結合次数** | 4 | Single, Double, Triple, Aromatic (one-hot) |
| **環内結合** | 1 | Binary (in ring/not in ring) |
| **共役** | 1 | Binary (conjugated/not conjugated) |
| **立体化学** | 3 | None, E, Z (one-hot) |
| **回転可能性** | 1 | Binary (rotatable/rigid) |
| **結合距離** | 1 | Normalized bond length (Å) |
| **予備** | 52 | 将来の拡張用（Wiberg bond order、Mayer bond order、電子密度など） |

**予備次元の設計意図**（ノード特徴と同様）:
- 将来の拡張性（QM計算由来の結合次数など）
- メモリアライメント最適化（2の累乗次元: 64 = 2^6）

**BDE正規化**:
```python
def normalize_bde(bde_kcal_mol: float) -> float:
    """
    Normalize BDE to [0, 1] range

    Typical BDE ranges:
    - C-C single: 85 kcal/mol
    - C=C double: 146 kcal/mol
    - C-H: 105 kcal/mol
    - O-H: 110 kcal/mol
    - Aromatic C-C: 120 kcal/mol

    Range: 50-200 kcal/mol
    """
    return (bde_kcal_mol - 50.0) / 150.0
```

---

## Phase 0: BDE-db2環境構築

（v4.0と同じ内容のため省略 - 変更なし）

---

## Phase 1: データ準備

### 1.1 NIST 17データ読み込み

```python
# src/data/nist_loader.py
"""
NIST 17 EI-MS Data Loader
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from pathlib import Path
from typing import List, Dict, Tuple

class NIST17Loader:
    """
    NIST 17 EI-MS Database Loader

    Expected format: MSP file with NIST spectra
    """

    def __init__(self, nist_path: str = "data/external/nist17/mainlib"):
        self.nist_path = Path(nist_path)

    def parse_msp(self, msp_file: str) -> List[Dict]:
        """
        Parse NIST MSP file

        Returns:
            List of dicts with keys: name, smiles, spectrum
        """
        spectra = []
        current_spectrum = {}
        current_peaks = []

        with open(msp_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()

                if line.startswith("Name:"):
                    if current_spectrum:
                        current_spectrum['spectrum'] = current_peaks
                        spectra.append(current_spectrum)
                    current_spectrum = {'name': line.split(":", 1)[1].strip()}
                    current_peaks = []

                elif line.startswith("SMILES:"):
                    current_spectrum['smiles'] = line.split(":", 1)[1].strip()

                elif line.startswith("Num Peaks:"):
                    num_peaks = int(line.split(":")[1].strip())

                elif line and line[0].isdigit():
                    # Peak data: "m/z intensity; m/z intensity; ..."
                    for peak in line.split(";"):
                        peak = peak.strip()
                        if peak:
                            mz, intensity = peak.split()
                            current_peaks.append((int(mz), int(intensity)))

        # Add last spectrum
        if current_spectrum:
            current_spectrum['spectrum'] = current_peaks
            spectra.append(current_spectrum)

        return spectra

    def load_all_spectra(self) -> pd.DataFrame:
        """Load all NIST 17 spectra into DataFrame"""

        all_spectra = []

        # NIST 17 has multiple MSP files
        msp_files = list(self.nist_path.glob("*.msp"))

        for msp_file in msp_files:
            print(f"Loading {msp_file.name}...")
            spectra = self.parse_msp(msp_file)
            all_spectra.extend(spectra)

        df = pd.DataFrame(all_spectra)

        print(f"Total spectra loaded: {len(df)}")

        return df

    def spectrum_to_array(self, peaks: List[Tuple[int, int]],
                          mz_range: Tuple[int, int] = (50, 1000)) -> np.ndarray:
        """
        Convert peak list to fixed-size array

        Args:
            peaks: List of (m/z, intensity) tuples
            mz_range: (min_mz, max_mz)

        Returns:
            spectrum_array: [950] array for m/z 50-1000
        """
        min_mz, max_mz = mz_range
        spectrum = np.zeros(max_mz - min_mz)

        for mz, intensity in peaks:
            if min_mz <= mz < max_mz:
                spectrum[mz - min_mz] = intensity

        # Normalize to [0, 1]
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()

        return spectrum
```

---

### 1.2 データフィルタリング（NEW!）

**目的**: BonDNet対応元素・分子量範囲に限定した高品質データセットの構築

```python
# src/data/filters.py
"""
Data filtering for NIST 17 dataset
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from typing import Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported elements (BonDNet BDE-db2 coverage)
SUPPORTED_ELEMENTS: Set[str] = {'C', 'H', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Br', 'I'}

def filter_supported_elements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter molecules containing only supported elements

    Args:
        df: DataFrame with 'smiles' column

    Returns:
        Filtered DataFrame
    """

    def contains_only_supported(smiles: str) -> bool:
        """Check if molecule contains only supported elements"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in SUPPORTED_ELEMENTS:
                logger.debug(f"Unsupported element {atom.GetSymbol()} in {smiles}")
                return False
        return True

    initial_count = len(df)
    logger.info(f"Starting element filtering: {initial_count} spectra")

    mask = df['smiles'].apply(contains_only_supported)
    filtered_df = df[mask].copy()

    removed = initial_count - len(filtered_df)
    logger.info(f"✅ Element filter complete:")
    logger.info(f"   Retained: {len(filtered_df)} / {initial_count} spectra")
    logger.info(f"   Removed:  {removed} spectra ({removed/initial_count*100:.2f}%)")
    logger.info(f"   Supported elements: {', '.join(sorted(SUPPORTED_ELEMENTS))}")

    return filtered_df


def filter_by_molecular_weight(
    df: pd.DataFrame,
    min_mw: float = 50.0,
    max_mw: float = 1000.0
) -> pd.DataFrame:
    """
    Filter molecules by molecular weight

    Args:
        df: DataFrame with 'smiles' column
        min_mw: Minimum molecular weight (default: 50.0)
        max_mw: Maximum molecular weight (default: 1000.0)

    Returns:
        Filtered DataFrame
    """

    def get_mw(smiles: str) -> float:
        """Calculate molecular weight"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        return Descriptors.MolWt(mol)

    initial_count = len(df)
    logger.info(f"Starting MW filtering: {initial_count} spectra")

    # Calculate molecular weights
    df['mw'] = df['smiles'].apply(get_mw)

    # Filter by MW range
    filtered_df = df[(df['mw'] >= min_mw) & (df['mw'] <= max_mw)].copy()

    removed = initial_count - len(filtered_df)
    logger.info(f"✅ MW filter complete:")
    logger.info(f"   Retained: {len(filtered_df)} / {initial_count} spectra")
    logger.info(f"   Removed:  {removed} spectra ({removed/initial_count*100:.2f}%)")
    logger.info(f"   MW range: {min_mw} - {max_mw} Da")
    logger.info(f"   Actual MW range: {filtered_df['mw'].min():.1f} - {filtered_df['mw'].max():.1f} Da")
    logger.info(f"   Mean MW: {filtered_df['mw'].mean():.1f} ± {filtered_df['mw'].std():.1f} Da")

    return filtered_df


def filter_valid_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter valid SMILES strings

    Args:
        df: DataFrame with 'smiles' column

    Returns:
        Filtered DataFrame
    """

    def is_valid_smiles(smiles: str) -> bool:
        """Check if SMILES is valid"""
        if pd.isna(smiles) or smiles == '':
            return False
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    initial_count = len(df)
    logger.info(f"Starting SMILES validation: {initial_count} spectra")

    mask = df['smiles'].apply(is_valid_smiles)
    filtered_df = df[mask].copy()

    removed = initial_count - len(filtered_df)
    logger.info(f"✅ SMILES validation complete:")
    logger.info(f"   Retained: {len(filtered_df)} / {initial_count} spectra")
    logger.info(f"   Removed:  {removed} spectra ({removed/initial_count*100:.2f}%)")

    return filtered_df


def apply_all_filters(
    df: pd.DataFrame,
    min_mw: float = 50.0,
    max_mw: float = 1000.0
) -> pd.DataFrame:
    """
    Apply all data filters sequentially

    Args:
        df: DataFrame with 'smiles' column
        min_mw: Minimum molecular weight
        max_mw: Maximum molecular weight

    Returns:
        Fully filtered DataFrame
    """

    initial_count = len(df)
    logger.info("=" * 60)
    logger.info("Starting comprehensive data filtering")
    logger.info("=" * 60)

    # Filter 1: Valid SMILES
    df = filter_valid_smiles(df)

    # Filter 2: Supported elements only
    df = filter_supported_elements(df)

    # Filter 3: Molecular weight range
    df = filter_by_molecular_weight(df, min_mw, max_mw)

    final_count = len(df)
    retention_rate = final_count / initial_count * 100

    logger.info("=" * 60)
    logger.info("Filtering complete")
    logger.info("=" * 60)
    logger.info(f"Initial dataset:  {initial_count} spectra")
    logger.info(f"Final dataset:    {final_count} spectra")
    logger.info(f"Retention rate:   {retention_rate:.2f}%")
    logger.info(f"Total removed:    {initial_count - final_count} spectra")
    logger.info("=" * 60)

    return df
```

**実行例**:
```python
from src.data.nist_loader import NIST17Loader
from src.data.filters import apply_all_filters

# Load NIST 17
loader = NIST17Loader("data/external/nist17/mainlib")
df = loader.load_all_spectra()
print(f"Loaded: {len(df)} spectra")

# Apply all filters
df_filtered = apply_all_filters(df, min_mw=50.0, max_mw=1000.0)
print(f"After filtering: {len(df_filtered)} spectra")
```

**予想出力**:
```
Loaded: 300,000 spectra
============================================================
Starting comprehensive data filtering
============================================================
Starting SMILES validation: 300,000 spectra
✅ SMILES validation complete:
   Retained: 298,000 / 300,000 spectra
   Removed:  2,000 spectra (0.67%)
Starting element filtering: 298,000 spectra
✅ Element filter complete:
   Retained: 283,000 / 298,000 spectra
   Removed:  15,000 spectra (5.03%)
   Supported elements: Br, C, Cl, F, H, I, N, O, P, S
Starting MW filtering: 283,000 spectra
✅ MW filter complete:
   Retained: 280,000 / 283,000 spectra
   Removed:  3,000 spectra (1.06%)
   MW range: 50.0 - 1000.0 Da
   Actual MW range: 52.1 - 999.8 Da
   Mean MW: 247.3 ± 152.8 Da
============================================================
Filtering complete
============================================================
Initial dataset:  300,000 spectra
Final dataset:    280,000 spectra
Retention rate:   93.33%
Total removed:    20,000 spectra
============================================================
After filtering: 280,000 spectra
```

---

### 1.3 BDE前計算（BonDNet BDE-db2）

（v4.0と同じ内容 - 入力データが280,000に変更されたのみ）

**実行時間見積もり**:
```
280,000 molecules × 15ms/molecule = 4,200 seconds = 70 minutes
```

---

### 1.4 PyG Graph生成

（v4.0と同じ内容 - 変更なし）

---

### 1.5 統合データ準備スクリプト

```python
# scripts/prepare_dataset.py
"""
Prepare complete dataset with filtering, BDE calculation, and PyG graphs
"""

from src.data.nist_loader import NIST17Loader
from src.data.filters import apply_all_filters
from src.data.bde_calculator import BDECalculator
from src.data.graph_generator import GraphGenerator
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_full_dataset():
    """
    Full pipeline: NIST → Filter → BDE → PyG Graph → HDF5
    """

    # Step 1: Load NIST 17
    logger.info("=" * 60)
    logger.info("Step 1: Loading NIST 17 data")
    logger.info("=" * 60)
    nist_loader = NIST17Loader("data/external/nist17/mainlib")
    df = nist_loader.load_all_spectra()
    logger.info(f"Loaded {len(df)} spectra from NIST 17")

    # Step 2: Apply data filters
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Data Filtering")
    logger.info("=" * 60)
    df = apply_all_filters(df, min_mw=50.0, max_mw=1000.0)
    logger.info(f"After filtering: {len(df)} spectra")

    # Step 3: Calculate BDE for all molecules
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Calculating BDE (BonDNet BDE-db2)")
    logger.info("=" * 60)
    bde_calc = BDECalculator(
        model_path="models/bondnet_bde_db2_best.pth",
        device="cuda"
    )
    bde_calc.batch_calculate(
        smiles_list=df['smiles'].tolist(),
        output_hdf5="data/processed/bde_cache.h5"
    )

    # Step 4: Generate PyG graphs
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Generating PyG graphs")
    logger.info("=" * 60)
    graph_gen = GraphGenerator("data/processed/bde_cache.h5")

    graphs = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating graphs"):
        spectrum_array = nist_loader.spectrum_to_array(row['spectrum'])
        graph = graph_gen.smiles_to_graph(
            smiles=row['smiles'],
            spectrum=spectrum_array,
            molecule_idx=idx
        )
        if graph is not None:
            graphs.append(graph)

    logger.info(f"Generated {len(graphs)} valid graphs")

    # Step 5: Train/Val/Test split
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Splitting dataset")
    logger.info("=" * 60)

    train_graphs, temp_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.5, random_state=42)

    logger.info(f"Train: {len(train_graphs)} ({len(train_graphs)/len(graphs)*100:.1f}%)")
    logger.info(f"Val:   {len(val_graphs)} ({len(val_graphs)/len(graphs)*100:.1f}%)")
    logger.info(f"Test:  {len(test_graphs)} ({len(test_graphs)/len(graphs)*100:.1f}%)")

    # Step 6: Save to disk
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Saving datasets")
    logger.info("=" * 60)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, split_graphs in [('train', train_graphs), ('val', val_graphs), ('test', test_graphs)]:
        output_path = output_dir / f"nist17_{split}.pt"
        torch.save(split_graphs, output_path)
        logger.info(f"✅ Saved {split}: {output_path} ({len(split_graphs)} graphs)")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Dataset preparation complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    prepare_full_dataset()
```

**実行**:
```bash
python scripts/prepare_dataset.py
```

**推定時間**:
- NIST読み込み: 30分
- データフィルタリング: 10分（NEW!）
- BDE計算: 70分
- PyG Graph生成: 60分
- **合計: 約2時間50分**

---

## Phase 2: GNN学習

（v4.0と同じ内容 - データセットサイズが280,000に変更されたのみ）

### 2.3 学習時間見積もり（更新）

**1エポックの時間**:
```
224,000 samples (train) ÷ 32 batch_size = 7,000 iterations
7,000 iterations × 0.8 sec/iter = 5,600 sec = 1.56 hours
```

**合計学習時間**:
```
300 epochs × 1.56 hours = 468 hours → early stoppingで約48時間（30エポック程度で収束想定）
```

---

## Phase 3: 評価

（v4.0と同じ内容 - 変更なし）

---

## 設定ファイル詳細

### config.yml（メイン設定、更新版）

```yaml
# config.yml - Main Configuration (v4.1)

project:
  name: "QC-GN2oEI"
  version: "2.1"
  description: "Physics-informed GNN for EI-MS prediction with BonDNet BDE-db2 and data filtering"

# BDE Configuration
bde:
  backend: "bondnet"  # Fixed (only option)

  bondnet:
    model_type: "bde-db2"  # Default model (retrained on BDE-db2)
    model_path: "models/bondnet_bde_db2_best.pth"
    dataset_path: "data/external/bde-db2"
    device: "cuda"
    batch_size: 256

    # Fallback for unsupported elements/structures
    fallback:
      aromatic_ring_bond: 120.0  # kcal/mol
      aliphatic_ring_bond: 85.0  # kcal/mol
      default_single_bond: 85.0
      default_double_bond: 146.0
      default_triple_bond: 200.0

# Data paths
data:
  nist17_path: "data/external/nist17/mainlib"
  bde_cache: "data/processed/bde_cache.h5"
  train_data: "data/processed/nist17_train.pt"
  val_data: "data/processed/nist17_val.pt"
  test_data: "data/processed/nist17_test.pt"

  # Data filtering (NEW in v4.1)
  filtering:
    # Supported elements (BonDNet BDE-db2 coverage)
    supported_elements: ['C', 'H', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Br', 'I']

    # Molecular weight range (aligned with output m/z range)
    min_molecular_weight: 50.0   # Da
    max_molecular_weight: 1000.0  # Da

    # SMILES validation
    validate_smiles: true

# Model architecture
model:
  type: "QCGN2oEI"

  # Node/Edge dimensions
  node_dim: 128  # 41 used + 87 reserved for future extensions
  edge_dim: 64   # 12 used + 52 reserved for future extensions

  # GNN layers
  hidden_dim: 256
  num_layers: 10
  num_heads: 8

  # Output
  output_dim: 1000  # m/z 50-1000 (950 bins + 50 padding)

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

  # Optimizer
  optimizer: "RAdam"  # Same as QC-GN2oMS2

  # Scheduler
  scheduler: "CosineAnnealingLR"
  scheduler_params:
    T_max: 300
    eta_min: 1e-6

  # Loss function
  loss: "cosine_similarity"

  # Early stopping
  early_stopping_patience: 50

  # Checkpointing
  save_every: 10  # Save checkpoint every 10 epochs
  checkpoint_dir: "checkpoints"

# Evaluation
evaluation:
  metrics:
    - "cosine_similarity"
    - "top_k_recall"
    - "mse"
    - "rmse"

  top_k_values: [5, 10, 20, 50]

  # Visualization
  plot_examples: 10
  plot_dir: "results/plots"

# Hardware
hardware:
  device: "cuda"
  gpu_id: 0
  num_workers: 4
  pin_memory: true

  # Mixed precision training
  use_amp: true
  amp_dtype: "float16"

  # Memory optimization
  gradient_accumulation_steps: 1
  empty_cache_every: 100  # Empty CUDA cache every 100 batches

# Logging
logging:
  use_wandb: true
  wandb_project: "qcgn2oei"
  wandb_entity: null  # Set your wandb username

  log_every: 10  # Log every 10 batches
  save_predictions: true

# Reproducibility
seed: 42
deterministic: true
```

---

## 開発環境構築

（v4.0と同じ内容 - 変更なし）

---

## タイムライン

### 全体スケジュール（v4.1更新版）

| フェーズ | タスク | 推定時間 | 累積時間 |
|---------|--------|---------|---------|
| **Phase 0** | BDE-db2ダウンロード | 30分 | 30分 |
| **Phase 0** | データ前処理 | 3時間 | 3.5時間 |
| **Phase 0** | BonDNet再学習 | 48-72時間 | 51.5-75.5時間 |
| **Phase 0** | モデル検証 | 1時間 | 52.5-76.5時間 |
| **Phase 1** | NIST読み込み | 30分 | 53-77時間 |
| **Phase 1** | データフィルタリング（NEW!） | 10分 | 53.17-77.17時間 |
| **Phase 1** | BDE計算（BonDNet、280K） | 70分 | 54.33-78.33時間 |
| **Phase 1** | PyG Graph生成 | 60分 | 55.33-79.33時間 |
| **Phase 2** | GNN学習 | 48時間 | 103.33-127.33時間 |
| **Phase 3** | 評価 | 2時間 | 105.33-129.33時間 |
| **合計** | - | **105-130時間** | **4.4-5.4日** |

### クリティカルパス

```
Phase 0 (BonDNet再学習) → Phase 1 (フィルタ+BDE計算) → Phase 2 (GNN学習) → Phase 3 (評価)
```

**v4.1での変更**:
- フィルタリング追加: +10分
- BDE計算時間: 75分 → 70分（データ減少により-5分）
- 正味の時間増加: +5分（誤差範囲内）

---

## 参考文献

（v4.0と同じ内容 - 変更なし）

---

## まとめ

### v4.1の主要な改善点

1. **データフィルタリングの明示化**: サポート元素・分子量の厳密なフィルタリング
2. **高品質データセット**: 280,000スペクトル（93.3% retention）、10元素・MW<=1000に統一
3. **予備次元の明確化**: 将来の拡張性のための設計であることを説明
4. **データ整合性の保証**: BonDNet対応元素・出力範囲（m/z 50-1000）との完全整合

### 期待される成果

- **精度**: Cosine Similarity > 0.85
- **速度**: 推論15ms/分子（実用レベル）
- **カバレッジ**: NIST 17の93.3%（高品質化合物のみ）
- **解釈性**: BDEエッジ特徴による物理化学的解釈可能性
- **ロバスト性**: サポート外元素・高MW化合物を除外した安定学習

---

**Document Version**: 4.1
**Last Updated**: 2025-12-02
**Status**: Ready for Implementation (with comprehensive data filtering)
