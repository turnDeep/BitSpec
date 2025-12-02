# 仕様書v4.0 問題点分析

## 発見された問題

### 1. サポート外元素の取り扱い

**問題**:
- 仕様書では「NIST 17の95%以上が10元素内に収まる」と記載
- しかし、残り5%（約15,000化合物）のフィルタリング処理が明記されていない
- サポート外元素（Si, B, Se, As, Ge, Al, Ti, etc.）を含む化合物の取り扱いが不明確

**影響**:
- BonDNet BDE-db2はC, H, O, N, F, S, P, Cl, Br, Iの10元素でのみ学習
- サポート外元素のBDE予測が不正確（ルールベースのフォールバック値に依存）
- ノード特徴量のone-hotエンコーディング（10次元）で対応不可
- 学習時に誤った特徴量（ゼロベクトル）が入力され、モデルが誤学習する可能性

**対策**: Phase 1.1でサポート元素のみを含む化合物にフィルタリング

---

### 2. 予備次元の過剰性

**問題**:
- ノード特徴量: 41次元使用 + 87次元予備 = 128次元
- エッジ特徴量: 12次元使用 + 52次元予備 = 64次元
- 予備次元が実使用次元の2倍以上

**現状分析**:

| 項目 | 実使用 | 予備 | 合計 | 使用率 |
|------|--------|------|------|--------|
| ノード特徴 | 41 | 87 | 128 | 32% |
| エッジ特徴 | 12 | 52 | 64 | 19% |

**影響**:
- メモリ使用量: 約3倍（41→128次元）
- 計算コスト: Encoder層での無駄な行列演算
- 280,000グラフ × 平均30原子 × 128次元 × 4 bytes = 約1.3GB（許容範囲内）

**結論**: RTX 5070 Ti（16GB）では問題なし。ただし、メモリ効率を重視する場合は以下に削減可能:
- ノード特徴: 64次元（41 + 23予備）
- エッジ特徴: 32次元（12 + 20予備）

**推奨**: 現状維持（将来の拡張性を優先）

---

### 3. 分子量1000超の化合物への対応

**問題**:
- 現在の仕様: `output_dim: 1000`（m/z 50-1000、実質950bins）
- NIST 17の実態: 最大分子量 ~2000 Da
- MW > 1000の化合物: 約1-3%（3,000-9,000スペクトル）

**NIST 17の分子量分布（推定）**:
```
MW範囲        | 化合物数  | 割合
-------------|----------|------
< 200 Da     | 150,000  | 50%
200-500 Da   | 120,000  | 40%
500-1000 Da  | 27,000   | 9%
> 1000 Da    | 3,000    | 1%
合計         | 300,000  | 100%
```

**影響**:
1. **データロス**: MW > 1000の化合物でm/z > 1000のピークが切り捨て
2. **スペクトル不完全**: 分子イオンピーク（M+）が記録されない
3. **評価指標の歪み**: 高MW化合物のCosine Similarityが不当に低く算出
4. **実用性の低下**: 医薬品、天然物、ペプチドなど高MW化合物に対応不可

**解決策の比較**:

#### 案A: フィルタリング（推奨）
MW <= 1000 Daの化合物のみ使用

**利点**:
- ✅ モデルサイズ変更不要
- ✅ メモリ使用量維持
- ✅ 学習速度維持
- ✅ データの一貫性

**欠点**:
- ❌ 約3,000スペクトル（1%）が除外
- ❌ 高MW化合物に非対応

#### 案B: 出力範囲拡張
m/z 50-2000（output_dim: 1950）

**利点**:
- ✅ 全NIST 17対応
- ✅ 汎用性向上

**欠点**:
- ❌ メモリ使用量1.95倍
- ❌ 出力層パラメータ増加（256→1950 = 7.6倍）
- ❌ 学習時間増加
- ❌ ほとんどのスペクトルで1000-2000範囲は空（計算の無駄）

**パラメータ数比較**:
```
案A (m/z 50-1000):
  prediction_head: 256 → 512 → 256 → 1000
  パラメータ数: 256×512 + 512×256 + 256×1000 = 518,144

案B (m/z 50-2000):
  prediction_head: 256 → 512 → 256 → 1950
  パラメータ数: 256×512 + 512×256 + 256×1950 = 761,344 (+47%)
```

**推奨**: 案A（フィルタリング）

---

## 修正方針

### Phase 1.1: データ準備の修正

```python
def prepare_full_dataset():
    """
    Full pipeline with filtering
    """

    # Step 1: Load NIST 17
    nist_loader = NIST17Loader("data/external/nist17/mainlib")
    df = nist_loader.load_all_spectra()
    print(f"Initial: {len(df)} spectra")

    # Step 1.5: Data Filtering
    print("\n=== Data Filtering ===")

    # Filter 1: Supported elements only
    df = filter_supported_elements(df)

    # Filter 2: Molecular weight <= 1000
    df = filter_by_molecular_weight(df, max_mw=1000.0)

    # Filter 3: Valid SMILES
    df = filter_valid_smiles(df)

    print(f"\nFinal dataset: {len(df)} spectra")

    # Continue with BDE calculation...
```

### 実装: フィルタリング関数

```python
# src/data/filters.py

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from typing import Set

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
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in SUPPORTED_ELEMENTS:
                return False
        return True

    initial_count = len(df)
    mask = df['smiles'].apply(contains_only_supported)
    filtered_df = df[mask].copy()

    removed = initial_count - len(filtered_df)
    print(f"Element filter: {len(filtered_df)} / {initial_count} retained "
          f"({removed} removed, {removed/initial_count*100:.2f}%)")

    return filtered_df


def filter_by_molecular_weight(df: pd.DataFrame, max_mw: float = 1000.0) -> pd.DataFrame:
    """
    Filter molecules by molecular weight

    Args:
        df: DataFrame with 'smiles' column
        max_mw: Maximum molecular weight (default: 1000.0)

    Returns:
        Filtered DataFrame
    """

    def get_mw(smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        return Descriptors.MolWt(mol)

    initial_count = len(df)
    df['mw'] = df['smiles'].apply(get_mw)
    filtered_df = df[df['mw'] <= max_mw].copy()

    removed = initial_count - len(filtered_df)
    print(f"MW filter (≤{max_mw}): {len(filtered_df)} / {initial_count} retained "
          f"({removed} removed, {removed/initial_count*100:.2f}%)")

    # MW statistics
    print(f"  MW range: {filtered_df['mw'].min():.1f} - {filtered_df['mw'].max():.1f}")
    print(f"  MW mean: {filtered_df['mw'].mean():.1f} ± {filtered_df['mw'].std():.1f}")

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
        if pd.isna(smiles) or smiles == '':
            return False
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    initial_count = len(df)
    mask = df['smiles'].apply(is_valid_smiles)
    filtered_df = df[mask].copy()

    removed = initial_count - len(filtered_df)
    print(f"SMILES filter: {len(filtered_df)} / {initial_count} retained "
          f"({removed} removed, {removed/initial_count*100:.2f}%)")

    return filtered_df
```

---

## 予想される最終データセット

```
NIST 17 初期データ:           300,000 spectra (100%)
  ↓
  フィルタ1: サポート元素のみ
  → 285,000 spectra (95.0%, -15,000)
  ↓
  フィルタ2: MW <= 1000
  → 282,000 spectra (94.0%, -3,000)
  ↓
  フィルタ3: 有効なSMILES
  → 280,000 spectra (93.3%, -2,000)

最終データセット:            280,000 spectra (93.3% retention)
```

---

## v4.1での変更点まとめ

### 1. Phase 1.1の更新
- データフィルタリングステップの明示的な追加
- フィルタリング関数の実装（`src/data/filters.py`）

### 2. 予想データセットサイズの修正
- 300,000 → 280,000 spectra

### 3. タイムラインへの影響
- フィルタリング処理: +10分
- BDE計算時間: 280,000 × 15ms = 70分（-5分）

### 4. config.ymlへの追加
```yaml
data:
  # Data filtering
  filtering:
    supported_elements: ['C', 'H', 'O', 'N', 'F', 'S', 'P', 'Cl', 'Br', 'I']
    max_molecular_weight: 1000.0  # Da
    min_molecular_weight: 50.0    # Da
```

### 5. 予備次元の説明明確化
- 将来の拡張性のための設計であることを明記
- メモリ使用量が許容範囲内であることを説明

---

## 結論

| 問題 | 対策 | 影響 |
|------|------|------|
| サポート外元素 | Phase 1.5でフィルタリング | -15,000 spectra (-5%) |
| 予備次元の多さ | 現状維持 | 影響なし |
| MW > 1000 | MW <= 1000でフィルタリング | -3,000 spectra (-1%) |

**最終データセット**: 280,000 spectra（93.3% retention）
**品質**: BonDNet対応元素のみ、m/z範囲整合、高品質データセット
