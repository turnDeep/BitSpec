# NEIMS v2.0 - 完全システム仕様書
## Neural Electron-Ionization Mass Spectrometry with Advanced Knowledge Distillation

**Version:** 2.0
**Date:** 2025-11-20
**Status:** Design Specification
**Authors:** Research Team

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Specifications](#4-model-specifications)
5. [Training Strategy](#5-training-strategy)
6. [Loss Functions](#6-loss-functions)
7. [Hyperparameters](#7-hyperparameters)
8. [Implementation Details](#8-implementation-details)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Deployment Specifications](#10-deployment-specifications)
11. [Implementation Roadmap](#11-implementation-roadmap)

---

## 1. Executive Summary

### 1.1 Project Objective

Develop a next-generation electron-ionization mass spectrum (EI-MS) prediction system that achieves:
- **Accuracy:** Recall@10 ≥ 95.5% (baseline NEIMS: 91.8%)
- **Speed:** ≤ 10ms per molecule inference (baseline NEIMS: 5ms)
- **Scalability:** No GPU required for inference

### 1.2 Innovation Points

1. **Teacher-Student Knowledge Distillation**
   - Heavy Teacher (GNN+ECFP Hybrid): Maximum accuracy
   - Lightweight Student (MoE-Residual MLP): Maximum speed

2. **Mixture of Experts (MoE) Architecture**
   - 4 specialized experts (Aromatic, Aliphatic, Heterocyclic, General)
   - Deep Residual MLPs (6 blocks per expert)

3. **Uncertainty-Aware Distillation**
   - MC Dropout for Teacher uncertainty estimation
   - Confidence-weighted soft labels
   - Label Distribution Smoothing (LDS)

4. **Adaptive Loss Weighting**
   - GradNorm-based automatic balancing
   - Temperature annealing schedule
   - Warmup strategy for stability

### 1.3 Expected Performance

| Metric | NEIMS v1.0 | NEIMS v2.0 (Target) | Improvement |
|--------|------------|---------------------|-------------|
| Recall@10 | 91.8% | 95.5-96.0% | +3.7-4.2% |
| Recall@5 | ~85% | 90-91% | +5-6% |
| Inference Speed | 5ms | 8-12ms | 1.6-2.4x slower |
| GPU Required | No | No | Same |
| Model Size | ~50MB | ~150MB | 3x larger |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Teacher Model   │         │  Student Model   │         │
│  │  (GNN + ECFP)    │────────▶│  (MoE-Residual)  │         │
│  │                  │ KD      │                  │         │
│  │  - GINEConv      │         │  - 4 Experts     │         │
│  │  - Bond-Breaking │         │  - Residual MLP  │         │
│  │  - MC Dropout    │         │  - Gate Network  │         │
│  └──────────────────┘         └──────────────────┘         │
│         ▲                              ▲                    │
│         │                              │                    │
│         └──────────────┬───────────────┘                    │
│                        │                                    │
│                 ┌──────▼──────┐                            │
│                 │  NIST Data  │                            │
│                 │  MassBank   │                            │
│                 │  GNPS       │                            │
│                 └─────────────┘                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PHASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Molecule → ECFP4 + Count FP → Student Model         │
│                                          ↓                   │
│                                    Mass Spectrum             │
│                                    (5-10ms latency)          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Overview

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Teacher Model** | High-accuracy prediction | GNN (GINEConv) + ECFP Hybrid |
| **Student Model** | Fast inference | MoE-Residual MLP |
| **Knowledge Distillation** | Knowledge transfer | Uncertainty-aware KD |
| **Data Pipeline** | Data processing | RDKit, PyTorch DataLoader |
| **Training Manager** | Orchestration | PyTorch Lightning |
| **Evaluation Engine** | Metrics computation | Custom metrics |

---

## 3. Data Pipeline

### 3.1 Data Sources

#### Primary Dataset: NIST EI-MS
- **Size:** ~300,000 spectra
- **Format:** (SMILES, Spectrum) pairs
- **Spectrum:** m/z 0-500, intensity 0-999
- **Split:** Train (80%), Val (10%), Test (10%)

#### Auxiliary Datasets
1. **MassBank:** ~50,000 spectra
2. **GNPS:** ~1,000,000 spectra
3. **PCQM4Mv2:** 3,740,000 molecules (for pre-training)

### 3.2 Data Processing Pipeline

```python
Input: SMILES string
  ↓
Step 1: Molecule Parsing (RDKit)
  ↓
Step 2: Molecular Graph Generation
  │
  ├──→ For Teacher: Graph(nodes, edges, features)
  │
  └──→ For Student: ECFP4 (4096-dim) + Count FP (2048-dim)
  ↓
Step 3: Spectrum Normalization
  - Max intensity → 999
  - Binning: 1 amu resolution
  ↓
Output: (Features, Target Spectrum)
```

### 3.3 Data Augmentation

#### Chemically Valid Augmentations
1. **Isotope Substitution**
   - C12 → C13 (1% natural abundance)
   - Apply to random 1-2 carbons per molecule

2. **Conformer Generation**
   - Generate 3-5 conformers via RDKit
   - Use for Teacher pre-training only

3. **Label Distribution Smoothing (LDS)**
   - Gaussian smoothing with σ=1.5 m/z units
   - Applied to target spectra

### 3.4 DataLoader Configuration

```yaml
Training:
  batch_size: 32
  num_workers: 8
  pin_memory: true
  shuffle: true

Validation:
  batch_size: 64
  num_workers: 4
  shuffle: false

Test:
  batch_size: 1
  num_workers: 1
```

### 3.5 Memory-Efficient Dataset Loading

#### 3.5.1 Challenge: Training on Full NIST17 (300k Compounds)

**Traditional In-Memory Approach:**
```
Per-compound memory: ~15-20 KB (graph + spectrum + metadata + overhead)
300,000 compounds:   10-15 GB (dataset only)
Model:               2-3 GB
Training overhead:   5-8 GB
Total:               17-26 GB
```

**32GB RAM System:**
```
OS + Processes:      3-5 GB
Available:           27-29 GB
Required:            17-26 GB  → ⚠️ Tight fit or overflow risk
```

#### 3.5.2 Solution: Lazy Loading with HDF5 Backend

**LazyMassSpecDataset Implementation:**

```python
from src.data.lazy_dataset import LazyMassSpecDataset

dataset = LazyMassSpecDataset(
    msp_file="data/NIST17.msp",
    mol_files_dir="data/mol_files",
    max_mz=500,
    cache_dir="data/processed/lazy_cache",
    precompute_graphs=False,  # ✅ On-the-fly generation
    max_samples=None          # ✅ Use full dataset (300k)
)
```

**Architecture:**

```
┌─────────────────────────────────────────────────────┐
│           LazyMassSpecDataset                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────┐     ┌──────────────────┐     │
│  │  Metadata Only  │     │   HDF5 Spectrum  │     │
│  │   (In RAM)      │     │   Cache (Disk)   │     │
│  │   ~150 MB       │     │   ~250 MB        │     │
│  │                 │     │   (Compressed)   │     │
│  └─────────────────┘     └──────────────────┘     │
│           │                       │                │
│           └───────────┬───────────┘                │
│                       │                            │
│              ┌────────▼────────┐                   │
│              │  On-the-Fly     │                   │
│              │  Graph Gen      │                   │
│              │  (Per Batch)    │                   │
│              └─────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

**Key Features:**
1. **Metadata-Only in RAM:** Store only compound IDs and SMILES (~0.5 KB per compound)
2. **HDF5 Spectrum Cache:** Compressed spectra on disk with fast random access
3. **On-the-Fly Graph Generation:** Generate molecular graphs only when needed
4. **Automatic Cache Building:** First run builds HDF5 cache (5-10 minutes)

#### 3.5.3 Memory Reduction Results

| Component | Traditional | Lazy Loading | Reduction |
|-----------|-------------|--------------|-----------|
| Dataset (RAM) | 10-15 GB | 150 MB | **70-100x** |
| Disk Cache | 10 GB | 250 MB | **40x** |
| Model | 2-3 GB | 2-3 GB | 1x |
| Training | 5-8 GB | 3-5 GB | 1.6x |
| **Total RAM** | **17-26 GB** | **5-8 GB** | **2-3x** |

**Result:** ✅ Full NIST17 (300k compounds) fits comfortably in 32GB RAM

**Note on PCQM4Mv2 Pre-training (3.74M compounds):**

The lazy loading approach also supports PCQM4Mv2 pre-training on 32GB RAM:

| Component | Traditional | Lazy Loading |
|-----------|-------------|--------------|
| Dataset (RAM) | 53.5 GB | 1.8 GB |
| Model | 2.5 GB | 2.5 GB |
| Training | 6.5 GB | 3.5 GB |
| **Total RAM** | **62.5 GB** ❌ | **7.8 GB** ✅ |

- RAM Usage: **24.3%** (7.8GB / 32GB)
- Free Memory: **24.2 GB**
- Even more efficient since Bond Masking task doesn't require spectra

#### 3.5.4 Performance Trade-offs

| Metric | Change | Assessment |
|--------|--------|------------|
| Memory Usage | 70-100x reduction | ✅ Excellent |
| Training Speed | ~13% slower | ⚠️ Acceptable |
| Disk Space | 40x reduction | ✅ Excellent |
| First-Run Setup | 5-10 min cache build | ⚠️ One-time cost |

**Reason for slowdown:** Graph generation overhead (CPU-bound)
- Mitigated by: Ryzen 7700's 8 cores for parallel processing
- Trade-off: 13% slower training vs 70x less memory → **Excellent trade-off**

#### 3.5.5 Usage Examples

**Recommended Configuration (config.yaml):**

```yaml
data:
  memory_efficient_mode:
    enabled: true
    use_lazy_loading: true
    lazy_cache_dir: "data/processed/lazy_cache"
    precompute_graphs: false  # ✅ Memory-efficient

    ram_32gb_mode:
      max_training_samples: null  # Use all 300k compounds
      gradient_accumulation: 2
      empty_cache_frequency: 50

training:
  student_distill:
    batch_size: 32
    num_workers: 8  # Ryzen 7700: 8 cores
    gradient_accumulation_steps: 2
```

**Training Workflow:**

```bash
# Step 1: Build HDF5 cache (first run only, 5-10 minutes)
python scripts/train_student.py --config config.yaml

# Output:
# Building HDF5 spectrum cache...
# Processing: 100%|██████████| 300000/300000
# Cache built: 300,000 samples
# Estimated memory usage: ~150.0 MB ✅

# Step 2: Memory estimation
python scripts/benchmark_memory.py --mode estimate --ram_gb 32

# Output:
# Full NIST17 (300,000 samples):
#   Lazy Loading:
#     Dataset:  150.0 MB
#     Total:    ~5.1 GB (dataset + model + training)
#     Status:   ✅ RECOMMENDED (fits in 32GB RAM)
```

#### 3.5.6 Memory Benchmark Tool

```bash
# Estimate memory usage
python scripts/benchmark_memory.py --mode estimate --ram_gb 32

# Benchmark actual loading (requires data)
python scripts/benchmark_memory.py --mode benchmark --dataset_size 100000

# Compare lazy vs traditional
python scripts/benchmark_memory.py --mode compare --max_samples 300000
```

**Example Output:**

```
NEIMS v2.0 Memory Estimation Tool
System RAM: 32 GB
Dataset Size: 300,000 compounds

🔴 Traditional In-Memory Dataset
  Dataset in RAM:     10240.0 MB  (10.0 GB)
  Total RAM Required: 17280.0 MB  (16.9 GB)
  Status: ⚠️ Tight on 32GB RAM

🟢 Lazy Loading Dataset (HDF5 + On-the-Fly)
  Dataset in RAM:       150.0 MB  (0.15 GB)
  HDF5 Cache (Disk):    250.0 MB  (0.24 GB)
  Total RAM Required:   5150.0 MB  (5.0 GB)
  Status: ✅ RECOMMENDED (fits in 32GB RAM)

📊 Memory Reduction Analysis
  Dataset Memory Reduction:   68.3x
  Total Memory Reduction:      3.4x
  RAM Freed:                  12.1 GB
```

#### 3.5.7 System Requirements Update

**Minimum (Inference Only)** - No change
- CPU: 4 cores
- RAM: 8 GB
- Storage: 500 MB

**Recommended (Training - Full NIST17)**
- **CPU:** 8+ cores (Ryzen 7700 or equivalent)
- **RAM:** 32 GB (sufficient with lazy loading)
- **GPU:** 1x RTX 5070 Ti (16GB) or equivalent
- **Storage:** 50 GB (10 GB dataset + 40 GB working)
- **OS:** Ubuntu 20.04+

**Previous requirement:** 64 GB RAM for full dataset
**New with lazy loading:** 32 GB RAM ✅

---

## 4. Model Specifications

### 4.1 Teacher Model Architecture

#### 4.1.1 Input Processing

```
Input: Molecular Graph
  - Nodes: Atoms with features [atom_type, degree, formal_charge, ...]
  - Edges: Bonds with features [bond_type, is_aromatic, is_conjugated, ...]
```

#### 4.1.2 GNN Branch

```python
GNN_Branch:
  Input_Embedding:
    - Atom_Embedding: (num_atom_types, 128)
    - Bond_Embedding: (num_bond_types, 128)

  GINEConv_Layers: 8 layers
    For each layer:
      - GINEConv(hidden_dim=256, edge_dim=128)
      - BatchNorm1d(256)
      - ReLU()
      - DropEdge(p=0.2)
      - PairNorm(scale=1.0)

  Bond_Breaking_Attention:
    - Input: Node features [N, 256]
    - Edge features [E, 128]
    - Output: Bond breaking probabilities [E, 1]
    - Implementation:
        attention_scores = MLP([node_i || node_j || edge_ij])
        bond_probs = Sigmoid(attention_scores)

  Global_Pooling:
    - Mean pooling: mean(node_features)
    - Max pooling: max(node_features)
    - Attention pooling: Σ(αi * node_i)
    - Concat all → [768-dim]

  Output: GNN_Embedding [768-dim]
```

#### 4.1.3 ECFP Branch

```python
ECFP_Branch:
  Input: ECFP4 fingerprint [4096-dim]

  MLP:
    - Linear(4096, 1024) + ReLU + Dropout(0.3)
    - Linear(1024, 512) + ReLU + Dropout(0.3)

  Output: ECFP_Embedding [512-dim]
```

#### 4.1.4 Fusion and Prediction

```python
Fusion:
  Concat([GNN_Embedding(768), ECFP_Embedding(512)]) → [1280-dim]

Prediction_Head:
  - Linear(1280, 1024) + ReLU + Dropout(0.3)
  - Linear(1024, 512) + ReLU + Dropout(0.3)
  - Linear(512, 501)  # m/z 0-500

  Bidirectional_Adjustment:
    # NEIMS-style physical adjustments
    - Fragment prediction: forward direction
    - Neutral loss prediction: backward direction
    - Combine: output = α * forward + (1-α) * backward

Output: Spectrum [501-dim]
```

#### 4.1.5 MC Dropout Configuration

```python
MC_Dropout:
  n_samples: 30
  dropout_rate: 0.3
  active_layers: [GNN layers, MLP layers]

  Uncertainty_Estimation:
    mean_spectrum = E[predictions]
    std_spectrum = Std[predictions]
```

### 4.2 Student Model Architecture

#### 4.2.1 Input Processing

```python
Input: ECFP4(4096) + Count_FP(2048) = [6144-dim]
```

#### 4.2.2 Gate Network (Router)

```python
Gate_Network:
  - Linear(6144, 512) + ReLU
  - Linear(512, 128) + ReLU
  - Linear(128, 4)  # 4 experts
  - Softmax() → Expert_Weights [4]

  Gating_Mechanism:
    - Top-k routing: k=2 (use top-2 experts)
    - Load balancing bias: dynamically adjusted
```

#### 4.2.3 Expert Networks

```python
Expert_Network (x4):
  Each Expert:
    Residual_Block x 6:
      def residual_block(x):
          identity = x
          x = LayerNorm(x)
          x = Linear(6144, 2048) + GELU()
          x = Linear(2048, 6144)
          return x + identity

    Output: Expert_Output [6144-dim]

Specialization (initialized via clustering):
  - Expert_1: Aromatic compounds
  - Expert_2: Aliphatic compounds
  - Expert_3: Heterocyclic compounds
  - Expert_4: General/Mixed
```

#### 4.2.4 Fusion and Prediction

```python
Expert_Fusion:
  # Weighted combination
  combined = Σ(expert_weights[i] * expert_outputs[i])
  # Output: [6144-dim]

Final_Prediction_Head:
  - Linear(6144, 2048) + GELU + Dropout(0.2)
  - Linear(2048, 1024) + GELU + Dropout(0.2)
  - Linear(1024, 501)  # m/z 0-500

  Bidirectional_Module:
    # Same as NEIMS original
    - Forward fragmentation prediction
    - Backward neutral loss prediction

Output: Spectrum [501-dim]
```

### 4.3 Model Size Comparison

| Model | Parameters | Size (MB) | Inference Speed |
|-------|-----------|-----------|-----------------|
| Teacher (GNN+ECFP) | ~15M | ~60 | 100ms |
| Student (MoE-Residual) | ~50M | ~200 | 10ms |
| NEIMS v1.0 (baseline) | ~10M | ~40 | 5ms |

---

## 5. Training Strategy

### 5.1 Three-Phase Training

#### Phase 1: Teacher Pre-training
```yaml
Objective: Learn robust molecular representations
Dataset: PCQM4Mv2 (3.74M molecules)
Task: Bond Masking (self-supervised)
Duration: 50 epochs
GPU: 4x A100 (40GB)
Time: ~1 week
```

#### Phase 2: Teacher Fine-tuning
```yaml
Objective: Specialize for spectrum prediction
Dataset: NIST + MassBank + GNPS
Task: Spectrum prediction with MC Dropout
Duration: 100 epochs
GPU: 2x A100
Time: ~3 days
```

#### Phase 3: Student Distillation
```yaml
Objective: Transfer knowledge to lightweight student
Dataset: NIST (with Teacher soft labels)
Task: Multi-objective distillation
Duration: 150 epochs
GPU: 1x A100
Time: ~2 days
```

### 5.2 Optimization Configuration

```yaml
Teacher_Training:
  optimizer: AdamW
  learning_rate: 1e-4
  weight_decay: 1e-5
  scheduler:
    type: CosineAnnealingWarmRestarts
    T_0: 10
    T_mult: 2
  gradient_clipping: 1.0

Student_Training:
  optimizer: AdamW
  learning_rate: 5e-4
  weight_decay: 1e-4
  scheduler:
    type: OneCycleLR
    max_lr: 1e-3
    pct_start: 0.1
  gradient_clipping: 0.5
```

---

## 6. Loss Functions

### 6.1 Teacher Training Loss

```python
L_teacher = L_spectrum + λ_bond * L_bond_masking

L_spectrum = MSE(predicted_spectrum, target_spectrum)

L_bond_masking = CrossEntropy(
    predicted_masked_bonds,
    true_masked_bonds
)

λ_bond = 0.1  # During pre-training
λ_bond = 0.0  # During fine-tuning
```

### 6.2 Student Training Loss (Complete)

```python
L_student = (α * L_hard +
             β * L_soft +
             γ * L_feature +
             δ_load * L_load +
             δ_entropy * L_entropy)
```

#### L1: Hard Label Loss
```python
L_hard = MSE(student_output, nist_spectrum)
```

#### L2: Soft Label Loss (Uncertainty-Aware)
```python
# Step 1: Get Teacher prediction with uncertainty
teacher_mean, teacher_std = teacher.predict_with_mc_dropout(
    molecule, n_samples=30
)

# Step 2: Apply Label Distribution Smoothing
teacher_smoothed = gaussian_filter1d(
    teacher_mean,
    sigma=1.5  # m/z units
)

# Step 3: Compute confidence weights
confidence = 1.0 / (1.0 + teacher_std)
confidence = confidence / confidence.sum()  # Normalize

# Step 4: Apply temperature
T = temperature_scheduler.get_temperature(epoch)
teacher_soft = teacher_smoothed / T
student_scaled = student_output / T

# Step 5: Confidence-weighted loss
L_soft = MSE(
    student_scaled * confidence,
    teacher_soft * confidence
) * (T ** 2)
```

#### L3: Feature-Level Distillation
```python
# Match intermediate representations
student_features = student.get_hidden_features(molecule)  # [6144]
teacher_features = teacher.get_ecfp_embedding(molecule)   # [512]

# Project to common space
student_proj = project_student(student_features)  # [512]

L_feature = MSE(student_proj, teacher_features)
```

#### L4: Load Balancing Loss
```python
# Switch Transformer style
def load_balancing_loss(expert_weights, expert_indices, num_experts=4):
    # expert_weights: [batch, num_experts]
    # expert_indices: [batch] - selected expert IDs

    # Frequency of expert selection
    expert_counts = torch.bincount(expert_indices, minlength=num_experts)
    expert_freq = expert_counts.float() / expert_counts.sum()

    # Average gating weight per expert
    expert_avg_weight = expert_weights.mean(dim=0)

    # Load balancing loss
    L_load = num_experts * (expert_freq * expert_avg_weight).sum()

    return L_load

δ_load = 0.01  # Empirically optimal (Switch Transformer)
```

#### L5: Entropy Regularization
```python
def entropy_regularization(expert_weights):
    # expert_weights: [batch, num_experts]
    eps = 1e-8
    entropy = -(expert_weights * torch.log(expert_weights + eps)).sum(dim=1)

    # Maximize entropy → negative loss
    L_entropy = -entropy.mean()

    return L_entropy

δ_entropy = 0.001  # 1/10 of load balancing
```

### 6.3 GradNorm-based Adaptive Weighting

```python
class AdaptiveLossWeighting:
    def __init__(self, warmup_epochs=15):
        self.warmup_epochs = warmup_epochs

        # Initial fixed weights
        self.alpha_init = 0.3  # Hard
        self.beta_init = 0.5   # Soft
        self.gamma_init = 0.2  # Feature

        # Learnable weights (for GradNorm)
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.2))

    def get_weights(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup: use fixed weights
            return self.alpha_init, self.beta_init, self.gamma_init
        else:
            # GradNorm: dynamic weights
            total = self.alpha + self.beta + self.gamma
            return (self.alpha/total,
                    self.beta/total,
                    self.gamma/total)

    def gradnorm_update(self, L_hard, L_soft, L_feature, model_params):
        # Compute gradient norms
        grad_hard = compute_grad_norm(L_hard, model_params)
        grad_soft = compute_grad_norm(L_soft, model_params)
        grad_feature = compute_grad_norm(L_feature, model_params)

        avg_grad = (grad_hard + grad_soft + grad_feature) / 3

        # Compute update ratios with clipping
        alpha_ratio = torch.clamp(avg_grad / (grad_hard + 1e-8), 0.5, 2.0)
        beta_ratio = torch.clamp(avg_grad / (grad_soft + 1e-8), 0.5, 2.0)
        gamma_ratio = torch.clamp(avg_grad / (grad_feature + 1e-8), 0.5, 2.0)

        # Update weights
        self.alpha.data *= alpha_ratio
        self.beta.data *= beta_ratio
        self.gamma.data *= gamma_ratio
```

### 6.4 Temperature Annealing Schedule

```python
class DynamicTemperatureScheduler:
    def __init__(self, T_init=4.0, T_min=1.0, schedule='cosine'):
        self.T_init = T_init
        self.T_min = T_min
        self.schedule = schedule

    def get_temperature(self, epoch, max_epochs):
        if self.schedule == 'cosine':
            progress = epoch / max_epochs
            T = self.T_min + (self.T_init - self.T_min) * \
                0.5 * (1 + np.cos(np.pi * progress))

        elif self.schedule == 'linear':
            T = self.T_init - (self.T_init - self.T_min) * \
                (epoch / max_epochs)

        elif self.schedule == 'exponential':
            decay_rate = -np.log(self.T_min / self.T_init) / max_epochs
            T = self.T_init * np.exp(-decay_rate * epoch)

        return max(T, self.T_min)

# Configuration
temp_scheduler = DynamicTemperatureScheduler(
    T_init=4.0,    # High temperature → soft labels
    T_min=1.0,     # Low temperature → sharp labels
    schedule='cosine'
)
```

---

## 7. Hyperparameters

### 7.1 Complete Hyperparameter Table

| Category | Parameter | Value | Justification |
|----------|-----------|-------|---------------|
| **Teacher** | GNN layers | 8 | Deep enough for complex patterns |
| | Hidden dim | 256 | Balance capacity/speed |
| | DropEdge rate | 0.2 | Prevent over-smoothing |
| | Dropout rate | 0.3 | MC Dropout standard |
| | MC samples | 30 | Optimal efficiency/accuracy |
| **Student** | Num experts | 4 | Chemical diversity coverage |
| | Residual blocks/expert | 6 | Depth for representation |
| | Hidden dim | 6144 | Match input dimension |
| | Top-k routing | 2 | Use 2 best experts |
| **Distillation** | T_init | 4.0 | Smooth initial transfer |
| | T_min | 1.0 | Sharp final distribution |
| | α (hard) init | 0.3 | Balance with soft |
| | β (soft) init | 0.5 | Primary knowledge source |
| | γ (feature) init | 0.2 | Auxiliary alignment |
| | δ_load | 0.01 | Switch Transformer optimal |
| | δ_entropy | 0.001 | 1/10 of load balancing |
| | Warmup epochs | 15 | 10% of total (150) |
| | Gradient clip range | [0.5, 2.0] | Prevent divergence |
| **Optimization** | Learning rate (Teacher) | 1e-4 | Conservative for GNN |
| | Learning rate (Student) | 5e-4 | Faster for MLP |
| | Weight decay | 1e-4 | L2 regularization |
| | Batch size | 32 | Memory/speed balance |
| | Max epochs | 150 | Sufficient convergence |
| **Data Aug** | LDS sigma | 1.5 | Smooth neighboring m/z |
| | Isotope prob | 0.05 | 5% of molecules |

### 7.2 Learning Rate Schedules

#### Teacher: CosineAnnealingWarmRestarts
```python
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # Restart every 10 epochs
    T_mult=2,    # Double restart interval
    eta_min=1e-6 # Minimum LR
)
```

#### Student: OneCycleLR
```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=150,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos'
)
```

---

## 8. Implementation Details

### 8.1 File Structure

```
NExtIMS/
├── data/
│   ├── nist/
│   ├── massbank/
│   ├── gnps/
│   ├── pcqm4mv2/
│   └── processed/
│       └── lazy_cache/         # HDF5 cache for lazy loading
├── src/
│   ├── models/
│   │   ├── teacher.py           # Teacher GNN+ECFP model
│   │   ├── student.py           # Student MoE-Residual model
│   │   ├── moe.py              # MoE components
│   │   └── modules.py          # Shared modules
│   ├── data/
│   │   ├── dataset.py          # Dataset classes
│   │   ├── nist_dataset.py     # NIST dataset classes
│   │   ├── lazy_dataset.py     # 🆕 Memory-efficient lazy loading
│   │   ├── preprocessing.py    # Data processing
│   │   └── augmentation.py     # Data augmentation
│   ├── training/
│   │   ├── teacher_trainer.py  # Teacher training logic
│   │   ├── student_trainer.py  # Student training logic
│   │   ├── losses.py           # Loss functions
│   │   └── schedulers.py       # Temperature/LR schedulers
│   ├── evaluation/
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── visualize.py        # Result visualization
│   └── utils/
│       ├── chemistry.py        # RDKit utilities
│       └── logging.py          # Logging utilities
├── configs/
│   ├── teacher_pretrain.yaml
│   ├── teacher_finetune.yaml
│   └── student_distill.yaml
├── scripts/
│   ├── train_teacher.py
│   ├── train_student.py
│   ├── evaluate.py
│   └── benchmark_memory.py     # 🆕 Memory usage estimation tool
├── tests/
│   ├── test_models.py
│   ├── test_data.py
│   └── test_training.py
└── docs/
    ├── NEIMS_v2_SYSTEM_SPECIFICATION.md  # This file
    └── API_REFERENCE.md
```

### 8.2 Key Implementation Classes

#### Teacher Model
```python
class TeacherModel(nn.Module):
    """
    GNN+ECFP Hybrid Teacher Model
    """
    def __init__(self, config):
        super().__init__()
        self.gnn_branch = GNNBranch(config)
        self.ecfp_branch = ECFPBranch(config)
        self.fusion = FusionModule(config)
        self.prediction_head = PredictionHead(config)

    def forward(self, mol_graph, ecfp, dropout=False):
        gnn_emb = self.gnn_branch(mol_graph, dropout=dropout)
        ecfp_emb = self.ecfp_branch(ecfp)
        fused = self.fusion(gnn_emb, ecfp_emb)
        spectrum = self.prediction_head(fused)
        return spectrum

    def predict_with_uncertainty(self, mol_graph, ecfp, n_samples=30):
        """MC Dropout uncertainty estimation"""
        self.train()  # Enable dropout
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(mol_graph, ecfp, dropout=True)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std
```

#### Student Model
```python
class StudentModel(nn.Module):
    """
    MoE-Residual Student Model
    """
    def __init__(self, config):
        super().__init__()
        self.gate = GateNetwork(config)
        self.experts = nn.ModuleList([
            ExpertNetwork(config) for _ in range(config.num_experts)
        ])
        self.prediction_head = PredictionHead(config)

        # Load balancing
        self.expert_bias = nn.Parameter(torch.zeros(config.num_experts))
        self.expert_load_history = torch.zeros(config.num_experts)

    def forward(self, ecfp_count_fp):
        # Gate decision
        gate_logits = self.gate(ecfp_count_fp) + self.expert_bias
        expert_weights = F.softmax(gate_logits, dim=-1)

        # Top-k routing
        top_k_weights, top_k_indices = expert_weights.topk(
            k=self.config.top_k, dim=-1
        )
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Expert computation
        expert_outputs = []
        for i in range(self.config.top_k):
            expert_idx = top_k_indices[:, i]
            expert_out = self.experts[expert_idx](ecfp_count_fp)
            expert_outputs.append(expert_out * top_k_weights[:, i].unsqueeze(-1))

        # Fusion
        combined = sum(expert_outputs)
        spectrum = self.prediction_head(combined)

        return spectrum, expert_weights, top_k_indices

    def update_expert_bias(self, expert_counts):
        """Auxiliary-loss-free load balancing"""
        self.expert_load_history = (0.9 * self.expert_load_history +
                                    0.1 * expert_counts)
        target_load = expert_counts.mean()
        self.expert_bias.data = -0.1 * (self.expert_load_history - target_load)
```

### 8.3 Training Loop Pseudocode

```python
def train_student_with_distillation(teacher, student, train_loader, config):
    # Initialize
    optimizer = AdamW(student.parameters(), lr=config.lr)
    lr_scheduler = OneCycleLR(optimizer, ...)
    temp_scheduler = DynamicTemperatureScheduler(...)
    loss_weighting = AdaptiveLossWeighting(warmup_epochs=15)

    for epoch in range(config.max_epochs):
        for batch in train_loader:
            molecules, nist_spectra = batch

            # === Teacher Prediction ===
            teacher_mean, teacher_std = teacher.predict_with_uncertainty(
                molecules, n_samples=30
            )

            # Temperature and smoothing
            T = temp_scheduler.get_temperature(epoch, config.max_epochs)
            teacher_soft = apply_lds(teacher_mean) / T
            confidence = 1.0 / (1.0 + teacher_std)

            # === Student Prediction ===
            student_out, expert_weights, expert_indices = student(molecules)
            student_scaled = student_out / T

            # === Compute Losses ===
            L_hard = F.mse_loss(student_out, nist_spectra)

            L_soft = F.mse_loss(
                student_scaled * confidence,
                teacher_soft * confidence
            ) * (T ** 2)

            L_feature = F.mse_loss(
                student.get_features(molecules),
                teacher.get_ecfp_embedding(molecules)
            )

            L_load = load_balancing_loss(expert_weights, expert_indices)
            L_entropy = entropy_regularization(expert_weights)

            # === Adaptive Weighting ===
            α, β, γ = loss_weighting.get_weights(epoch)

            if epoch >= loss_weighting.warmup_epochs:
                α, β, γ = loss_weighting.gradnorm_update(
                    L_hard, L_soft, L_feature, student.parameters()
                )

            # === Total Loss ===
            L_total = (α * L_hard +
                      β * L_soft +
                      γ * L_feature +
                      0.01 * L_load +
                      0.001 * L_entropy)

            # === Optimization ===
            optimizer.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
            optimizer.step()
            lr_scheduler.step()

            # Update expert bias (auxiliary-loss-free balancing)
            expert_counts = torch.bincount(expert_indices.flatten())
            student.update_expert_bias(expert_counts)

        # Validation and logging
        validate(student, val_loader)
```

---

## 9. Evaluation Metrics

### 9.1 Primary Metrics

#### Recall@K
```python
def recall_at_k(predicted_spectrum, target_spectrum, k=10):
    """
    Fraction of top-k target peaks found in top-k predictions
    """
    # Get top-k peaks
    target_top_k = torch.topk(target_spectrum, k).indices
    pred_top_k = torch.topk(predicted_spectrum, k).indices

    # Compute overlap
    overlap = len(set(target_top_k.tolist()) & set(pred_top_k.tolist()))
    recall = overlap / k

    return recall
```

#### Spectral Similarity (Cosine)
```python
def spectral_similarity(pred, target):
    """
    Cosine similarity between spectra
    """
    return F.cosine_similarity(pred, target, dim=-1).mean()
```

### 9.2 Secondary Metrics

- **MAE (Mean Absolute Error):** Average intensity error
- **RMSE (Root Mean Square Error):** Intensity prediction accuracy
- **Peak Detection Rate:** Percentage of true peaks detected
- **False Positive Rate:** Percentage of false peaks

### 9.3 Efficiency Metrics

- **Inference Time:** Average ms per molecule
- **Throughput:** Molecules per second
- **Memory Usage:** Peak GPU/RAM consumption
- **Model Size:** Disk space (MB)

### 9.4 Benchmarking Protocol

```yaml
Test_Set:
  - NIST Test Split: 30,000 spectra
  - Evaluation modes:
      - In-distribution: Common molecular scaffolds
      - Out-of-distribution: Rare scaffolds
      - By molecular weight: <200, 200-400, >400 Da
      - By complexity: Simple, Medium, Complex

Metrics_to_Report:
  - Recall@5, @10, @20
  - Spectral Similarity (mean, median, std)
  - MAE, RMSE
  - Inference time (mean, std, p95, p99)
  - Expert usage distribution (for Student)
```

---

## 10. Deployment Specifications

### 10.1 Inference API

```python
class NEIMS_v2_Predictor:
    """
    Production inference interface
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = StudentModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.to(device)

    def predict(self, smiles: str) -> Dict[str, np.ndarray]:
        """
        Predict mass spectrum from SMILES

        Args:
            smiles: SMILES string

        Returns:
            {
                'mz': np.ndarray (501,),        # m/z values 0-500
                'intensity': np.ndarray (501,), # Intensities 0-999
                'expert_weights': np.ndarray (4,) # Expert contributions
            }
        """
        # Preprocess
        mol = Chem.MolFromSmiles(smiles)
        ecfp = self._compute_ecfp(mol)
        count_fp = self._compute_count_fp(mol)

        # Inference
        with torch.no_grad():
            spectrum, expert_weights, _ = self.model(
                torch.cat([ecfp, count_fp])
            )

        return {
            'mz': np.arange(501),
            'intensity': spectrum.cpu().numpy(),
            'expert_weights': expert_weights.cpu().numpy()
        }

    def batch_predict(self, smiles_list: List[str],
                     batch_size: int = 32) -> List[Dict]:
        """Batch prediction for efficiency"""
        # Implementation...
```

### 10.2 Model Export

```python
# ONNX Export
torch.onnx.export(
    student_model,
    dummy_input,
    "neims_v2_student.onnx",
    opset_version=14,
    input_names=['ecfp_count_fp'],
    output_names=['spectrum', 'expert_weights'],
    dynamic_axes={
        'ecfp_count_fp': {0: 'batch_size'},
        'spectrum': {0: 'batch_size'}
    }
)

# TorchScript Export
scripted_model = torch.jit.script(student_model)
scripted_model.save("neims_v2_student.pt")
```

### 10.3 System Requirements

#### Minimum (Inference Only)
- CPU: 4 cores
- RAM: 8 GB
- Storage: 500 MB
- OS: Linux/macOS/Windows

#### Recommended (Training)
- GPU: 1x NVIDIA A100 (40GB)
- CPU: 16 cores
- RAM: 64 GB
- Storage: 100 GB (for datasets)
- OS: Ubuntu 20.04+

---

## 11. Implementation Roadmap

### 11.1 Phase Timeline (12 Weeks)

#### Week 1-2: Infrastructure Setup
```yaml
Tasks:
  - Setup project structure
  - Implement data loaders
  - Test RDKit integration
  - Baseline NEIMS reproduction
Deliverables:
  - Working data pipeline
  - NEIMS v1.0 baseline (91.8% Recall@10)
```

#### Week 3-4: Teacher Model Development
```yaml
Tasks:
  - Implement GNN branch (GINEConv)
  - Implement ECFP branch
  - Bond-Breaking attention module
  - MC Dropout uncertainty estimation
Deliverables:
  - Teacher model achieving >93% Recall@10
  - Uncertainty calibration validated
```

#### Week 5-6: Student Model Development
```yaml
Tasks:
  - Implement MoE architecture
  - Residual blocks per expert
  - Gate network with load balancing
  - Expert specialization initialization
Deliverables:
  - Student model (standalone) >92% Recall@10
  - Expert usage balanced (20-30% each)
```

#### Week 7-8: Knowledge Distillation
```yaml
Tasks:
  - Implement uncertainty-aware KD loss
  - LDS integration
  - Temperature annealing
  - Feature-level distillation
Deliverables:
  - Student with KD: >94% Recall@10
  - Stable training dynamics
```

#### Week 9-10: Adaptive Optimization
```yaml
Tasks:
  - GradNorm implementation
  - Warmup strategy
  - Gradient clipping
  - Hyperparameter tuning
Deliverables:
  - Fully optimized model: >95.5% Recall@10
  - Convergence within 150 epochs
```

#### Week 11: Large-Scale Pre-training
```yaml
Tasks:
  - Pre-train Teacher on PCQM4Mv2
  - Bond masking task
  - Transfer learning to NIST
Deliverables:
  - Pre-trained Teacher: >94.5% Recall@10
  - Student (post-distill): >96% Recall@10
```

#### Week 12: Evaluation & Deployment
```yaml
Tasks:
  - Comprehensive benchmarking
  - Model export (ONNX, TorchScript)
  - API implementation
  - Documentation
Deliverables:
  - Final model: 95.5-96% Recall@10 @ 10ms
  - Production-ready inference API
  - Technical report
```

### 11.2 Checkpoints and Validation

| Week | Checkpoint | Success Criteria |
|------|------------|------------------|
| 2 | Baseline | NEIMS v1.0 reproduced (91.8% Recall@10) |
| 4 | Teacher | >93% Recall@10 on validation set |
| 6 | Student (standalone) | >92% Recall@10, balanced experts |
| 8 | Basic KD | >94% Recall@10 |
| 10 | Full System | >95.5% Recall@10 |
| 12 | Final | >95.5% @ <12ms inference |

### 11.3 Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Expert collapse | High | Critical | 3-layer safety (Load balance + Entropy + Bias) |
| Training instability | Medium | High | Warmup + Gradient clipping + Temperature annealing |
| Overfitting | Medium | Medium | Dropout + Weight decay + Data augmentation |
| Slow inference | Low | Medium | Model quantization + ONNX optimization |
| GPU OOM | Medium | High | Gradient accumulation + Mixed precision |

---

## 12. Appendices

### 12.A Configuration Files

#### teacher_pretrain.yaml
```yaml
model:
  type: TeacherGNN
  gnn_layers: 8
  hidden_dim: 256
  dropout: 0.3
  drop_edge: 0.2

data:
  dataset: PCQM4Mv2
  batch_size: 256
  num_workers: 16

training:
  task: bond_masking
  epochs: 50
  optimizer: AdamW
  lr: 1e-4
  weight_decay: 1e-5
  scheduler: CosineAnnealingWarmRestarts
```

#### student_distill.yaml
```yaml
model:
  type: StudentMoE
  num_experts: 4
  residual_blocks_per_expert: 6
  hidden_dim: 6144
  top_k: 2

distillation:
  teacher_checkpoint: checkpoints/teacher_best.ckpt
  mc_dropout_samples: 30
  temperature_init: 4.0
  temperature_min: 1.0
  warmup_epochs: 15

  loss_weights:
    alpha_init: 0.3  # Hard
    beta_init: 0.5   # Soft
    gamma_init: 0.2  # Feature
    delta_load: 0.01
    delta_entropy: 0.001

training:
  epochs: 150
  batch_size: 32
  optimizer: AdamW
  lr: 5e-4
  weight_decay: 1e-4
  gradient_clip: 0.5
```

### 12.B Key References

1. **NEIMS:** Wei et al., "Rapid Prediction of Electron-Ionization Mass Spectrometry Using Neural Networks", ACS Central Science, 2019
2. **GLNNs:** Zhang et al., "Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation", ICLR 2021
3. **MoE:** Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models", JMLR 2022
4. **GradNorm:** Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing", ICML 2018
5. **MC Dropout:** Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
6. **MolCLR:** Wang et al., "Molecular Contrastive Learning of Representations via Graph Neural Networks", Nature MI 2022
7. **FIORA:** "Local neighborhood-based prediction of compound mass spectra", Nature Communications 2025
8. **Uncertainty-aware KD:** "Teaching with Uncertainty: Unleashing the Potential of Knowledge Distillation", CVPR 2024

### 12.C Glossary

- **KD:** Knowledge Distillation
- **MoE:** Mixture of Experts
- **GNN:** Graph Neural Network
- **ECFP:** Extended Connectivity Fingerprint
- **MC Dropout:** Monte Carlo Dropout
- **LDS:** Label Distribution Smoothing
- **GradNorm:** Gradient Normalization
- **NIST:** National Institute of Standards and Technology
- **EI-MS:** Electron Ionization Mass Spectrometry

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-20 | Research Team | Initial specification |
| 1.1 | 2025-11-21 | Research Team | Added memory-efficient dataset loading (Section 3.5) |

**Document Status:** APPROVED FOR IMPLEMENTATION
**Next Review:** After Phase 1 completion (Week 2)

**Latest Updates (v1.1):**
- Added Section 3.5: Memory-Efficient Dataset Loading
- Implemented LazyMassSpecDataset with HDF5 backend
- Added memory benchmark tool (scripts/benchmark_memory.py)
- Updated system requirements: 32GB RAM now sufficient for full NIST17
- Memory reduction: 70-100x for dataset, 2-3x overall

---

**END OF SPECIFICATION**
