# BDE Generator Modifications Summary

## 修正日時
2025-11-27

## 修正の背景

### 問題点
- 従来の実装では芳香族環状結合のBDE値が95 kcal/molに設定されていた
- 実際の芳香族C-C結合のBDE（~124 kcal/mol）との乖離が大きかった
- 環状結合の統計的分布（70-80%が芳香族）を考慮すると、95 kcal/molは過小評価

### EI-MS実験事実
- **70eV電子衝撃下では芳香環は切れにくい**（prominent molecular ion生成）
- フラグメンテーションは芳香環の**β位**（隣接結合）で優先的に発生
- QC-GN2oMS2（MS/MS用）が環状結合を0とするのは、CIDで芳香環が切れないため

## 修正内容

### 1. BDE値の変更

| 結合タイプ | 修正前 | **修正後** | 理由 |
|----------|-------|----------|------|
| **芳香族環状結合** | 95 kcal/mol | **120 kcal/mol** | ベンゼンC-C実測値~124に最接近、EI-MSでの最大安定性を表現 |
| 非芳香族環状結合 | 85 kcal/mol | **85 kcal/mol** | シクロヘキサン~83に一致、変更なし |
| 非環状C-C結合 | 85 kcal/mol | **85 kcal/mol** | 標準値、変更なし |

### 2. 正規化後の値

```python
# 正規化範囲: 50-120 kcal/mol

normalize(85.0)  = (85-50)/(120-50) = 0.5   # 非芳香族環・非環状
normalize(120.0) = (120-50)/(120-50) = 1.0  # 芳香族環（境界値）
```

**GNNへの学習シグナル:**
- 芳香族環: 正規化値 **1.0** → 「最大の安定性、切れない」
- その他: 正規化値 **0.5** → 「中程度、切れやすい」

### 3. コード変更詳細

#### ファイル: `src/data/bde_generator.py`

**a) クラスdocstring更新**
- ALFABET制限（環状結合を予測しない）を明記
- 芳香族/非芳香族環の扱いを明確化
- QC-GN2oMS2との違いを詳細に記載

**b) `__init__` パラメータコメント更新**
```python
fallback_bde: float = 85.0  # Default BDE for acyclic C-C & aliphatic rings
```
- 用途を明確化（非環状・非芳香族環用）
- 芳香族環は別途120使用することを注記

**c) `predict_bde()` メソッド改良**
```python
# Priority 4: 環状結合の補完ロジック追加
if len(bde_dict) < num_bonds:
    for bond_idx, bond in enumerate(mol.GetBonds()):
        if bond_idx not in bde_dict:
            if bond.IsInRing():
                if bond.GetIsAromatic():
                    bde_dict[bond_idx] = 120.0  # ← 新規
                else:
                    bde_dict[bond_idx] = 85.0
```

**主要な改善点:**
1. ALFABETが部分的に予測した場合（環状結合を除く）にも対応
2. 環状結合を芳香族/非芳香族で明確に区別
3. 詳細なコメントで設計意図を記録

**d) `_rule_based_bde()` メソッド更新**
```python
elif bond.GetIsAromatic():  # Aromatic
    # Aromatic C-C bonds: ~124-147 kcal/mol (experimental)
    # Use 120 (bde_max) to signal maximum stability to GNN
    # EI-MS (70eV): aromatic rings rarely break, produce prominent M+
    base_bde = 120.0  # 95.0 → 120.0
```

**改善点:**
- 実験値（124-147 kcal/mol）を参照文献とともに記載
- EI-MS実験事実を明記
- GNNへの学習意図を明示

**e) 参照文献追加**
```python
# References:
# - Blanksby & Ellison, Acc. Chem. Res. 2003, 36, 255-263
# - Aromatic ring stability in EI-MS: fragmentation occurs at β-bond to ring
```

## Phase 1事前学習への影響

### BDE分布の変化

**PCQM4Mv2データセット（3.74M分子、推定）:**
```
芳香族環状結合: 15.6M結合 (17.3%)
  修正前: BDE正規化値 = 0.643 (95 kcal/mol)
  修正後: BDE正規化値 = 1.0   (120 kcal/mol) ← +35%増加

非芳香族環状: 3.4M結合 (3.8%)
  BDE正規化値 = 0.5 (85 kcal/mol) ← 変更なし

非環状結合: 71M結合 (78.9%)
  BDE正規化値 = 0.4-0.6 (ALFABET予測) ← 変更なし
```

### GNN学習への効果

**修正前（芳香族=95）:**
```python
芳香族環: 0.643 (中程度の安定性)
その他:   0.5   (標準的な安定性)
→ 差が小さい（14%差）、区別が曖昧
```

**修正後（芳香族=120）:**
```python
芳香族環: 1.0   (最大の安定性) ← 境界値
その他:   0.5   (標準的な安定性)
→ 差が大きい（100%差）、明確な区別
```

### 期待される改善

**Phase 1（BDE事前学習）:**
- GNNが芳香族の「切れない」特性を明確に学習
- 環状/非環状、芳香族/非芳香族の構造パターンを正確に獲得
- BDE予測MAE: ~1.0-1.2 kcal/mol（目標: <1.5）

**Phase 2（NIST17ファインチューニング）:**
- 芳香族化合物（NIST17の60-70%）での予測精度向上
- フラグメント位置の正確な予測（β位切断）
- **Recall@10: 95.5% → 96.0-96.5%**（+0.5-1.0%改善）

## テスト項目

### 1. ベンゼン (c1ccccc1)
- 期待: 全6結合 = 120 kcal/mol (正規化: 1.0)
- 検証: `bde_gen.predict_bde(mol)` でBDE辞書確認

### 2. シクロヘキサン (C1CCCCC1)
- 期待: 全6結合 = 85 kcal/mol (正規化: 0.5)
- 検証: 非芳香族環の扱い確認

### 3. ヘキサン (CCCCCC)
- 期待: 全5結合 = 85 kcal/mol (正規化: 0.5)
- 検証: 非環状結合の扱い確認

### 4. トルエン (Cc1ccccc1)
- 期待: 芳香環6結合=120、メチル-フェニル1結合=85
- 検証: 混合分子での正しい区別

### 5. ピリジン (c1ccncc1)
- 期待: 全結合=120（芳香族複素環）
- 検証: ヘテロ原子を含む芳香族の扱い

## 実装上の注意点

### 1. 境界値（1.0）の数値安定性
- Sigmoid出力層のため、目標値1.0は問題なし
- GNNは0.95-0.99程度に収束（完全な1.0は困難だが不要）

### 2. Phase 0（BDE事前計算）への影響
- `scripts/precompute_bde.py`は自動的に新ロジックを使用
- 既存のHDF5キャッシュは再生成推奨（環状結合の値が変更）

### 3. 後方互換性
- 既存のpickleキャッシュは自動的に新ロジックで補完される
- HDF5キャッシュが優先されるため、Phase 0再実行を推奨

## QC-GN2oMS2との比較

### QC-GN2oMS2（MS/MS用）
```python
edge_features = [bond_type, ALFABET_BDE if acyclic else 0.0]
# 環状結合 = 0（MS/MSで切れないため無視）
```

### NExtIMS v2.0（EI-MS用、修正後）
```python
# 学習タスクとして使用
bde_target = {
    'aromatic': 120.0,  # ← 修正（95→120）
    'aliphatic_ring': 85.0,
    'acyclic': ALFABET_prediction
}
# → GNNが構造からBDEを予測する能力を獲得
# → 推論時にALFABET不要
```

## まとめ

### 主要な改善点
1. ✅ **芳香族環BDE: 95 → 120 kcal/mol**（実験値124に最接近）
2. ✅ **正規化値: 芳香族=1.0、その他=0.5**（明確な区別）
3. ✅ **環状結合補完ロジック実装**（ALFABET制限を克服）
4. ✅ **詳細なコメント・文献追加**（保守性向上）
5. ✅ **EI-MS実験事実の反映**（理論的正当性）

### 期待される効果
- Phase 1: BDE予測精度向上、芳香族認識強化
- Phase 2: EI-MSスペクトル予測精度 **+0.5-1.0%** 改善
- 実用性: 医薬品・天然物（85%が芳香環含有）での高精度化

### 次のステップ
1. Phase 0再実行（HDF5キャッシュ再生成）
2. Phase 1訓練で学習曲線確認
3. Phase 2でRecall@10性能評価
4. 芳香族化合物での詳細分析
