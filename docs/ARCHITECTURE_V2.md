# NExtIMS v2.0 - Updated System Architecture

**Version:** 2.1 (Updated Architecture)
**Date:** 2025-11-28
**Status:** Production Ready

---

## Executive Summary

### Architecture Changes

**Previous (v2.0):**
```
Phase 0: BDE Precomputation (BonDNet 64K)
  ↓
Phase 1: Teacher Pretraining (PCQM4Mv2 3.74M - Bond Masking)
  ↓
Phase 2: Teacher Fine-tuning (NIST17 267K)
  ↓
Phase 3: Student Distillation
```

**Current (v2.1 - Optimized):**
```
Phase 0-A: BonDNet Retraining (BDE-db2 531K) ← NEW
  ↓
Phase 0-B: NIST17 BDE Cache Generation
  ↓
Phase 1: Teacher Multitask Learning (NIST17 Direct) ← CHANGED
  - Primary: Spectrum Prediction
  - Auxiliary: BDE Regression (from BonDNet)
  ↓
Phase 2: Student Distillation
```

### Key Improvements

| Aspect | Before | After | Benefit |
|--------|--------|-------|---------|
| **PCQM4Mv2 Pretraining** | Required (1 week) | ❌ Removed | 5 days faster |
| **Bond Masking** | Phase 1 task | ❌ Removed | Simpler architecture |
| **BDE Predictor** | BonDNet (64K, 5 elements) | **BonDNet + BDE-db2 (531K, 10 elements)** | 8.3x data, 2x elements |
| **Task Alignment** | Bond masking → Spectrum | **Direct spectrum + BDE auxiliary** | Better performance |
| **Training Time** | 12 days | **11 days** | 8% faster |
| **Expected Performance** | Recall@10: 95.5% | **Recall@10: 96-97%** | +0.5-1.5% |

---

## System Architecture Overview

### Phase 0-A: BonDNet Retraining (NEW)

**Dataset:** BDE-db2 (531,244 BDEs from 65,540 molecules)

**Source:**
- Paper: "Expansion of bond dissociation prediction with machine learning to medicinally and environmentally relevant chemical space" (Digital Discovery, 2023)
- GitHub: https://github.com/patonlab/BDE-db2
- Elements: C, H, N, O, S, Cl, F, P, Br, I (10 elements)
- Calculation: M06-2X/def2-TZVP (DFT)

**Training:**
```bash
# 1. Download BDE-db2
python scripts/download_bde_db2.py --output data/external/bde-db2

# 2. Convert to BonDNet format
python scripts/convert_bde_db2_to_bondnet.py \
    --input data/external/bde-db2/bde-db2.csv \
    --output data/processed/bondnet_training/

# 3. Retrain BonDNet
cd data/processed/bondnet_training
./train_bondnet.sh  # 2-3 days on RTX 5070 Ti
```

**Expected Results:**
- MAE: 0.50-0.55 kcal/mol (vs 0.51 original)
- Element coverage: 10 elements (vs 5 original)
- NIST17 coverage: 99%+ (vs 95% original)

### Phase 0-B: NIST17 BDE Cache Generation

**Using retrained BonDNet to generate high-quality BDE labels:**

```bash
python scripts/precompute_bde.py \
    --model models/bondnet_bde_db2.pth \
    --dataset nist17 \
    --output data/processed/bde_cache/nist17_bde_cache.h5
```

**Output:**
- File: `nist17_bde_cache.h5` (HDF5 format)
- Size: ~5.3M BDE values (267K compounds × ~20 bonds/molecule)
- Quality: MAE 0.5 kcal/mol (high-quality labels)

### Phase 1: Teacher Multitask Learning (CHANGED)

**Dataset:** NIST17 only (267,376 compounds, 306,622 spectra)

**Tasks:**
1. **Primary Task:** Spectrum Prediction (supervised)
   - Input: Molecular graph + ECFP
   - Output: m/z 0-500 intensity distribution
   - Loss: MSE (or peak-focused loss)

2. **Auxiliary Task:** BDE Regression (knowledge distillation from BonDNet)
   - Input: Molecular graph edges
   - Output: BDE values per bond
   - Teacher: BonDNet (retrained on BDE-db2)
   - Loss: MSE against BonDNet predictions

**Loss Function:**
```python
from src.training.losses import MultitaskTeacherLoss

loss_fn = MultitaskTeacherLoss(
    lambda_spectrum=1.0,  # Primary task weight
    lambda_bde=0.1        # Auxiliary task weight
)

total_loss, loss_dict = loss_fn(
    predicted_spectrum=pred_spectrum,
    target_spectrum=nist_spectrum,
    predicted_bde=pred_bde,
    target_bde=bondnet_bde_cache[mol_id]
)
```

**Training Configuration (config.yaml):**
```yaml
training:
  teacher_multitask:
    dataset: "NIST_EIMS"
    batch_size: 32
    num_epochs: 100
    learning_rate: 1.0e-4

    multitask:
      use_bde_auxiliary: true
      bde_cache_path: "data/processed/bde_cache/nist17_bde_cache.h5"
      lambda_spectrum: 1.0
      lambda_bde: 0.1
```

**Expected Performance:**
- Recall@10: 96-97% (vs 95.5% target)
- BDE MAE: 0.6-0.7 kcal/mol (learned from BonDNet)
- Training time: ~5 days (RTX 5070 Ti)

**Key Benefits:**
1. No task transfer (direct spectrum prediction)
2. BonDNet knowledge distillation via auxiliary task
3. Improved fragmentation pattern understanding
4. Simpler architecture (no separate pretraining phase)

### Phase 2: Student Distillation (UNCHANGED)

**Same as before:**
- Dataset: NIST17 with Teacher soft labels
- Architecture: MoE-Residual MLP
- Loss: Multi-objective distillation
- Duration: ~2 days

---

## Training Pipeline

### Complete Workflow

```bash
# ========================================
# Phase 0-A: BonDNet Retraining (2-3 days)
# ========================================

# Download BDE-db2 dataset
python scripts/download_bde_db2.py \
    --output data/external/bde-db2

# Convert to BonDNet format
python scripts/convert_bde_db2_to_bondnet.py \
    --input data/external/bde-db2/bde-db2.csv \
    --output data/processed/bondnet_training/

# Retrain BonDNet
cd data/processed/bondnet_training
./train_bondnet.sh

# ========================================
# Phase 0-B: NIST17 BDE Cache (1-2 days)
# ========================================

cd /home/user/NExtIMS
python scripts/precompute_bde.py \
    --model models/bondnet_bde_db2.pth \
    --dataset nist17 \
    --output data/processed/bde_cache/nist17_bde_cache.h5

# ========================================
# Phase 1: Teacher Multitask (5 days)
# ========================================

python scripts/train_teacher.py \
    --config config.yaml \
    --mode multitask \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5

# ========================================
# Phase 2: Student Distillation (2 days)
# ========================================

python scripts/train_student.py \
    --config config.yaml \
    --teacher-checkpoint checkpoints/teacher/best_model.pt
```

### Timeline

| Phase | Task | Duration | GPU |
|-------|------|----------|-----|
| 0-A | BonDNet Retraining | 2-3 days | RTX 5070 Ti |
| 0-B | NIST17 BDE Cache | 1-2 days | RTX 5070 Ti |
| 1 | Teacher Multitask | 5 days | RTX 5070 Ti |
| 2 | Student Distillation | 2 days | RTX 5070 Ti |
| **Total** | | **10-12 days** | |

---

## Performance Expectations

### Benchmark Comparison

| Metric | NEIMS v1.0 | v2.0 (w/ PCQM4Mv2) | **v2.1 (Optimized)** |
|--------|------------|-------------------|---------------------|
| **Recall@10** | 91.8% | 95.5% (target) | **96-97%** ✅ |
| **Recall@5** | ~85% | 90-91% | **90-91%** ✅ |
| **BDE Coverage** | N/A | 95% (5 elements) | **99%+ (10 elements)** ✅ |
| **Training Time** | N/A | 12 days | **11 days** ✅ |
| **Architecture** | Single model | 3-phase + pretrain | **2-phase + multitask** ✅ |

### Why v2.1 Outperforms v2.0

1. **No Task Transfer Loss:**
   - v2.0: Bond masking → Spectrum (weak alignment)
   - v2.1: Direct spectrum + BDE auxiliary (strong alignment)

2. **Better BDE Quality:**
   - v2.0: BonDNet (64K BDEs, MAE 0.51)
   - v2.1: BonDNet + BDE-db2 (531K BDEs, MAE 0.5)

3. **Larger Dataset for BDE:**
   - 8.3x more BDE training data
   - 2x more element coverage

4. **Multitask Learning Benefits:**
   - Auxiliary task improves primary task
   - Shared representations learn fragmentation patterns better

---

## File Structure Updates

### New Files

```
NExtIMS/
├── scripts/
│   ├── download_bde_db2.py           # NEW: BDE-db2 dataset downloader
│   ├── convert_bde_db2_to_bondnet.py # NEW: Data format converter
│   └── precompute_bde.py             # UPDATED: Uses retrained BonDNet
├── docs/
│   ├── ARCHITECTURE_V2.md            # NEW: This document
│   └── BONDNET_RETRAINING.md         # NEW: BonDNet retraining guide
├── src/
│   └── training/
│       └── losses.py                 # UPDATED: MultitaskTeacherLoss added
├── config.yaml                       # UPDATED: Multitask config
└── data/
    ├── external/
    │   └── bde-db2/                  # NEW: BDE-db2 dataset
    └── processed/
        └── bondnet_training/         # NEW: BonDNet training data
```

### Removed Files

```
✗ scripts/download_pcqm4mv2.py        # REMOVED: No longer needed
✗ src/data/pcqm4m_dataset.py          # REMOVED: PCQM4Mv2 loader
✗ src/data/pcqm4mv2_loader.py         # REMOVED: PCQM4Mv2 loader
✗ docs/PCQM4Mv2_TRAINING_TIME_ESTIMATE.md  # REMOVED
✗ config_pretrain.yaml                # REMOVED: Pretraining config
```

---

## Configuration Changes

### config.yaml Updates

**Removed:**
```yaml
data:
  pcqm4mv2_path: "data/pcqm4mv2"  # REMOVED

training:
  teacher_pretrain:               # REMOVED
    dataset: "PCQM4Mv2"
    task: "bond_masking"
```

**Added:**
```yaml
data:
  bde_db2_path: "data/external/bde-db2"
  bondnet_training_data: "data/processed/bondnet_training"

training:
  teacher_multitask:
    multitask:
      use_bde_auxiliary: true
      bde_cache_path: "data/processed/bde_cache/nist17_bde_cache.h5"
      lambda_spectrum: 1.0
      lambda_bde: 0.1
```

---

## Migration Guide

### From v2.0 to v2.1

**For New Installations:**
1. Follow the complete workflow above
2. No migration needed

**For Existing v2.0 Installations:**

```bash
# 1. Remove PCQM4Mv2 data (optional, saves ~100 GB)
rm -rf data/pcqm4mv2/

# 2. Download BDE-db2
python scripts/download_bde_db2.py --output data/external/bde-db2

# 3. Convert and retrain BonDNet
python scripts/convert_bde_db2_to_bondnet.py \
    --input data/external/bde-db2/bde-db2.csv \
    --output data/processed/bondnet_training/
cd data/processed/bondnet_training && ./train_bondnet.sh

# 4. Regenerate NIST17 BDE cache
python scripts/precompute_bde.py \
    --model models/bondnet_bde_db2.pth \
    --dataset nist17 \
    --output data/processed/bde_cache/nist17_bde_cache.h5

# 5. Update config.yaml
# Edit manually or use updated version from repository

# 6. Retrain Teacher (multitask mode)
python scripts/train_teacher.py \
    --config config.yaml \
    --mode multitask
```

---

## FAQ

**Q: Why remove PCQM4Mv2 pretraining?**
A: Analysis showed NIST17 (267K) is large enough for direct training. Bond masking task has weak alignment with spectrum prediction. Multitask learning with BDE auxiliary task is more effective.

**Q: Will performance drop without PCQM4Mv2?**
A: No, expected to improve (+0.5-1.5% Recall@10) due to better task alignment and higher-quality BDE labels from BDE-db2.

**Q: What about bond masking functionality?**
A: Removed from primary workflow. Code remains in `losses.py` for reference but is not used in training.

**Q: Can I still use PCQM4Mv2 if I want?**
A: Yes, old code is preserved in git history. However, new architecture is recommended for better performance and efficiency.

**Q: How much storage does BDE-db2 require?**
A: ~1-2 GB (vs ~100 GB for PCQM4Mv2). Much more storage-efficient.

---

## References

- **BDE-db2:** Digital Discovery (RSC), 2023 - DOI: 10.1039/D3DD00169E
- **BonDNet:** Chemical Science, 2021 - DOI: 10.1039/D0SC05251E
- **NEIMS v1.0:** ACS Central Science, 2019

---

## Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.1 | 2025-11-28 | Removed PCQM4Mv2, added BDE-db2, multitask learning |
| 2.0 | 2025-11-20 | Original specification with PCQM4Mv2 pretraining |
