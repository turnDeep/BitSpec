# BonDNet Retraining Guide

Complete guide for retraining BonDNet on the BDE-db2 dataset (531,244 BDEs).

## Overview

**Goal:** Retrain BonDNet with BDE-db2 to improve:
- BDE prediction accuracy: 0.51 → 0.5 kcal/mol
- Element coverage: 5 → 10 elements (C,H,N,O,S,Cl,F,P,Br,I)
- NIST17 coverage: 95% → 99%+

**Estimated Time:**
- Data preparation: 2-4 hours
- Training (RTX 5070 Ti): 2-3 days
- Total: ~3 days

---

## Prerequisites

### System Requirements

**Minimum:**
- GPU: NVIDIA RTX 3090 (24GB) or better
- RAM: 32 GB
- Storage: 50 GB free space
- OS: Ubuntu 20.04+

**Recommended:**
- GPU: NVIDIA RTX 5070 Ti (16GB) or A100 (40GB)
- RAM: 64 GB
- Storage: 100 GB SSD
- CUDA: 12.0+

### Software Dependencies

```bash
# Python 3.8+
python --version  # Should be >= 3.8

# CUDA toolkit
nvcc --version  # Should be >= 11.0

# PyTorch with CUDA support
python -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

---

## Step-by-Step Guide

### Step 1: Download BDE-db2 Dataset

```bash
# From NExtIMS root directory
cd /home/user/NExtIMS

# Download dataset (automatic)
python scripts/download_bde_db2.py \
    --output data/external/bde-db2

# Expected output:
# - data/external/bde-db2/bde-db2.csv.gz (~100-200 MB compressed)
# - data/external/bde-db2/bde-db2.csv (~500-800 MB extracted)
# - data/external/bde-db2/BDE-db2-repo/ (GitHub repository)
```

**Manual Download (if automatic fails):**
1. Visit: https://figshare.com/articles/dataset/bde-db2_csv_gz/19367051
2. Download `bde-db2.csv.gz`
3. Place in `data/external/bde-db2/`
4. Extract: `gunzip data/external/bde-db2/bde-db2.csv.gz`

### Step 2: Convert to BonDNet Format

```bash
# Convert BDE-db2 CSV to BonDNet training format
python scripts/convert_bde_db2_to_bondnet.py \
    --input data/external/bde-db2/bde-db2.csv \
    --output data/processed/bondnet_training/

# This creates:
# - molecules.sdf (65,540 molecules with 3D coordinates)
# - molecule_attributes.yaml (molecular properties)
# - reactions.yaml (531,244 BDE reactions)
# - train_bondnet.sh (training script)

# Expected time: 1-2 hours
```

**Subset Testing (optional):**
```bash
# Convert only 1,000 molecules for testing
python scripts/convert_bde_db2_to_bondnet.py \
    --input data/external/bde-db2/bde-db2.csv \
    --output data/processed/bondnet_training_test/ \
    --max-molecules 1000
```

### Step 3: Install BonDNet

```bash
# Clone BonDNet repository
cd ~
git clone https://github.com/mjwen/bondnet.git
cd bondnet

# Install in editable mode
pip install -e .

# Verify installation
python -c "from bondnet.prediction.predictor import predict_single_molecule; print('OK')"
```

**Dependencies:**
```bash
# Install additional dependencies if needed
conda install "pytorch>=1.10.0" -c pytorch
conda install "dgl>=0.5.0" -c dglteam
conda install "pymatgen>=2022.01.08" "rdkit>=2020.03.5" "openbabel>=3.1.1" -c conda-forge
```

### Step 4: Configure Training

**Edit training script:**
```bash
cd /home/user/NExtIMS/data/processed/bondnet_training
nano train_bondnet.sh
```

**Key parameters to adjust:**
```bash
# Batch size (adjust based on GPU memory)
BATCH_SIZE=128  # RTX 5070 Ti (16GB)
# BATCH_SIZE=64   # RTX 3090 (24GB) - more stable
# BATCH_SIZE=256  # A100 (40GB) - faster

# Epochs
EPOCHS=200  # Full training
# EPOCHS=50   # Quick test

# Learning rate
LEARNING_RATE=0.001  # Default
# LEARNING_RATE=0.0005  # More conservative
```

### Step 5: Start Training

```bash
cd /home/user/NExtIMS/data/processed/bondnet_training

# Start training (runs in background)
nohup ./train_bondnet.sh > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Or use screen/tmux
screen -S bondnet
./train_bondnet.sh
# Detach: Ctrl+A, D
# Reattach: screen -r bondnet
```

**Expected output:**
```
Epoch 1/200 | Loss: 2.456 | MAE: 1.23 kcal/mol | Time: 15m
Epoch 2/200 | Loss: 1.987 | MAE: 0.98 kcal/mol | Time: 15m
...
Epoch 200/200 | Loss: 0.234 | MAE: 0.52 kcal/mol | Time: 15m

Training complete!
Model saved to: ../../models/bondnet_bde_db2.pth
```

### Step 6: Validate Model

```bash
# Test prediction on a single molecule
python ~/bondnet/bondnet/scripts/predict_bde.py \
    --model /home/user/NExtIMS/models/bondnet_bde_db2.pth \
    --smiles "CCO"  # Ethanol

# Expected output:
# Bond 0 (C-C): BDE = 85.2 kcal/mol
# Bond 1 (C-O): BDE = 91.5 kcal/mol
# Bond 2 (O-H): BDE = 104.3 kcal/mol
```

### Step 7: Benchmark Against Test Set

```bash
# Evaluate on BDE-db2 test set
python scripts/evaluate_bondnet.py \
    --model models/bondnet_bde_db2.pth \
    --test-set data/external/bde-db2/test_set.csv

# Expected results:
# MAE: 0.50-0.55 kcal/mol (vs 0.51 original)
# R²: 0.98-0.99
```

---

## Troubleshooting

### GPU Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size:
   ```bash
   BATCH_SIZE=64  # or even 32
   ```

2. Enable gradient accumulation (edit training script):
   ```python
   --gradient-accumulation-steps 2
   ```

3. Use mixed precision training:
   ```python
   --fp16
   ```

### Training Divergence

**Symptom:** Loss increases or NaN

**Solutions:**
1. Reduce learning rate:
   ```bash
   LEARNING_RATE=0.0005  # or 0.0001
   ```

2. Add gradient clipping:
   ```python
   --max-grad-norm 1.0
   ```

3. Check data quality:
   ```bash
   python scripts/validate_bondnet_data.py \
       --data-dir data/processed/bondnet_training/
   ```

### Slow Training

**Expected speed:** ~15-20 minutes per epoch (RTX 5070 Ti)

**If slower:**
1. Check GPU utilization:
   ```bash
   nvidia-smi -l 1
   # GPU util should be 95-100%
   ```

2. Increase num_workers:
   ```python
   --num-workers 8  # Ryzen 7700: 8 cores
   ```

3. Use SSD for dataset (not HDD)

---

## Performance Benchmarks

### Training Time (Full BDE-db2)

| GPU | Batch Size | Time/Epoch | Total Time | Cost |
|-----|------------|------------|------------|------|
| RTX 5070 Ti (16GB) | 128 | 15 min | 50 hours | ~2-3 days |
| RTX 4090 (24GB) | 256 | 10 min | 33 hours | ~1.5 days |
| A100 (40GB) | 512 | 5 min | 17 hours | ~1 day |

### Expected Performance

| Metric | BonDNet Original | BonDNet + BDE-db2 |
|--------|------------------|-------------------|
| **MAE** | 0.51 kcal/mol | **0.50-0.55 kcal/mol** |
| **R²** | 0.98 | **0.98-0.99** |
| **Elements** | C,H,O,F,Li (5) | **C,H,N,O,S,Cl,F,P,Br,I (10)** |
| **Dataset** | 64K BDEs | **531K BDEs** |
| **NIST17 Coverage** | 95% | **99%+** |

---

## Next Steps

After successful training:

### 1. Generate NIST17 BDE Cache

```bash
python scripts/precompute_bde.py \
    --model models/bondnet_bde_db2.pth \
    --dataset nist17 \
    --output data/processed/bde_cache/nist17_bde_cache.h5 \
    --max-samples 0

# Expected time: 1-2 days
# Output: ~5.3M BDE values for 267K NIST17 compounds
```

### 2. Train NEIMS Teacher (Multitask)

```bash
python scripts/train_teacher.py \
    --config configs/teacher_nist17_multitask.yaml \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    --epochs 100

# Expected time: 5 days
# Expected performance: Recall@10 = 96-97%
```

### 3. Distill to Student

```bash
python scripts/train_student.py \
    --config configs/student_distill.yaml \
    --teacher-checkpoint checkpoints/teacher_best.ckpt \
    --epochs 150

# Expected time: 2 days
# Expected performance: Recall@10 = 95.5-96%
```

---

## Advanced Configuration

### Custom Training Script

```python
# custom_train_bondnet.py
import torch
from bondnet.model.gnn import BondNet
from bondnet.data.dataloader import DataLoaderReactionNetwork

# Load data
train_loader = DataLoaderReactionNetwork(
    sdf_file="molecules.sdf",
    attributes_file="molecule_attributes.yaml",
    reactions_file="reactions.yaml",
    batch_size=128,
    num_workers=8
)

# Initialize model
model = BondNet(
    in_feats=64,
    gnn_layers=4,
    fc_layers=3,
    dropout=0.1
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(200):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        predictions = model(batch.graph, batch.features)
        loss = criterion(predictions, batch.labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/200 | Loss: {total_loss/len(train_loader):.4f}")

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

# Save final model
torch.save(model.state_dict(), "bondnet_bde_db2_final.pth")
```

---

## FAQ

**Q: Can I train on CPU?**
A: Not recommended. Training would take 2-3 weeks. Use cloud GPU (AWS, GCP, Colab Pro) if local GPU unavailable.

**Q: How much GPU memory is needed?**
A: Minimum 12 GB (RTX 3060 Ti). Recommended 16 GB+ (RTX 5070 Ti, RTX 4090).

**Q: Can I resume training from checkpoint?**
A: Yes, use `--resume-from checkpoint_epoch_N.pth` in training script.

**Q: How to verify model quality?**
A: Run benchmark on test set. MAE should be 0.50-0.55 kcal/mol, R² > 0.98.

**Q: What if training fails?**
A: Check `training.log` for errors. Common issues: OOM (reduce batch size), NaN loss (reduce LR), data corruption (re-convert dataset).

---

## References

- **BDE-db2 Paper:** Digital Discovery (RSC), 2023 - DOI: 10.1039/D3DD00169E
- **BonDNet Paper:** Chemical Science, 2021 - DOI: 10.1039/D0SC05251E
- **BonDNet GitHub:** https://github.com/mjwen/bondnet
- **BDE-db2 GitHub:** https://github.com/patonlab/BDE-db2

---

## Support

For issues:
1. Check this document's Troubleshooting section
2. Review BonDNet GitHub issues
3. Examine training logs: `data/processed/bondnet_training/training.log`
4. Contact: NExtIMS development team
