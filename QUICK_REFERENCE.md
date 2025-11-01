# BitSpec: Quick Reference Guide

## What is BitSpec?

A **Graph Convolutional Network (GCN) for predicting EI-MS fragmentation patterns** from molecular structures.

- **Input**: MOL file or SMILES string
- **Output**: 1000-dimensional mass spectrum (m/z 0-999)
- **Training Data**: NIST EI-MS database
- **Hardware**: Optimized for RTX 50 GPU (sm_120)

## Architecture at a Glance

```
Molecule (MOL/SMILES)
  ↓
Feature Extraction (48D atoms + 6D bonds)
  ↓
Node Embedding (→ 256D)
  ↓
5 GCN Layers (residual + batch norm)
  ↓
Attention Pooling
  ↓
MLP Head
  ↓
Mass Spectrum (1000D output, sigmoid)
```

## Key Files (what to know)

| File | Purpose | Lines |
|------|---------|-------|
| `src/models/gcn_model.py` | Model definition | 340 |
| `src/data/features.py` | Feature extraction (48D+6D) | 300 |
| `src/training/loss.py` | Modified Cosine Loss | 120 |
| `src/data/dataset.py` | Dataset + dataloader | 220 |
| `scripts/train.py` | Training loop | 280 |
| `config.yaml` | All hyperparameters | - |
| `src/data/mol_parser.py` | MSP/MOL parsing | 240 |

## Data Processing Pipeline

```
NIST17.MSP (text)
  ↓
Parse MSP file
  ↓
Load corresponding MOL file
  ↓
Extract features (RDKit)
  ↓
Create PyG Data object
  ↓
Normalize spectrum (1000D)
  ↓
Batch multiple molecules
  ↓
Feed to model
```

## Feature Dimensions

### Node Features (per atom) - 48D
- Atomic number: 12D one-hot (H, C, N, O, F, Si, P, S, Cl, Br, I + unknown)
- Degree: 8D one-hot (0-6 + unknown)
- Formal charge: 8D one-hot (-3 to +3 + unknown)
- Chirality: 5D one-hot (tags 0-3 + unknown)
- Num hydrogens: 6D one-hot (0-4 + unknown)
- Hybridization: 7D one-hot (SP, SP2, SP3, etc. + unknown)
- Aromatic: 1D binary
- In ring: 1D binary

### Edge Features (per bond) - 6D
- Bond type: 4D one-hot (SINGLE, DOUBLE, TRIPLE, AROMATIC)
- Conjugated: 1D binary
- In ring: 1D binary

## Current Training Configuration

```yaml
Batch size:        32
Learning rate:     0.001 (AdamW)
Epochs:            200
Optimizer:         AdamW with CosineAnnealingWarmRestarts
Mixed precision:   Yes (FP16)
Loss function:     ModifiedCosineLoss
Activation:        ReLU
Pooling:           Attention
Residual:          Yes (GCN layers)
Early stopping:    Yes (patience=20)
Gradient clipping: 1.0
```

## Model Hyperparameters

```yaml
Node features:     48 (optimized from 157)
Edge features:     6 (optimized from 16)
Hidden dim:        256
Num layers:        5
Output spectrum:   1000 (m/z bins)
Dropout:           0.1
Conv type:         GCNConv
Pooling:           Attention
```

## Loss Function: ModifiedCosineLoss

```
Loss = 1 - [ (CosineSim + ShiftedMatching) / 2 ]

Where:
- CosineSim: Standard cosine similarity between pred and target spectra
- ShiftedMatching: Accounts for neutral losses (precursor m/z differences)
- Tolerance: 0.1 m/z units for peak matching
```

**Why this loss?**
- EI-MS fragmentations often lose neutral molecules (H2O, CO, etc.)
- Standard cosine ignores peak shifts
- ModifiedCosineLoss handles both global similarity AND shifted peaks

## Evaluation Metrics

```
Primary:       Cosine Similarity (main metric)
Secondary:     Pearson Correlation, MSE, MAE
Additional:    Top-K Accuracy (top 20 peaks)
```

## Data Format

### NIST MSP Format
```
Name: Aspirin
InChIKey: BSYNRYMUTXBXSQ-UHFFFAOYSA-N
Formula: C9H8O4
MW: 180
ID: 200001
Num peaks: 15
41 100.0          # m/z intensity pairs
55 50.0
69 25.0
...
180 999.0
$$$$
```

### MOL File
- Standard MOL V2000/V3000 format
- Filename: `ID{compound_id}.MOL`
- Must match ID from MSP file

## Spectrum Processing

1. **Binning**: m/z values discretized to 1.0 m/z bins (0-1000)
2. **Aggregation**: Max intensity per bin
3. **Normalization**: Divided by max intensity → [0, 1]
4. **Output**: 1000-dimensional float array

## Key Recent Changes

| Commit | Change | Impact |
|--------|--------|--------|
| 4406639 | Feature reduction (157→48 atoms, 16→6 bonds) | More efficient, similar performance |
| 96f8670 | Simplified to ModifiedCosineLoss only | Cleaner, faster training |
| b957899 | ID mapping MSP↔MOL | Data integrity guaranteed |

## Important Limitations

⚠️ **Current system does NOT support**:
- Pretraining on external datasets
- Transfer learning
- Pretrained models
- Backbone freezing
- Multi-task learning
- PCQM4Mv2 integration
- Massformer architecture

## What's Implemented

✓ End-to-end GCN training
✓ NIST data loading
✓ EI-MS aware loss function
✓ Mixed precision training (FP16)
✓ RTX 50 optimization
✓ Data caching/serialization
✓ Attention pooling
✓ Residual connections

## Running the System

```bash
# Test with 10 samples
python scripts/test_training.py

# Preprocess NIST data
python scripts/preprocess_data.py --input data/NIST17.MSP --output data/processed

# Train from scratch
python scripts/train.py --config config.yaml

# Make predictions
python scripts/predict.py --checkpoint checkpoints/best_model.pt \
                          --config config.yaml \
                          --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

## File Structure Quick Map

```
BitSpec/
├── config.yaml ..................... All hyperparameters
├── README.md ....................... Project overview
├── requirements.txt ................ Dependencies (PyTorch 2.7.0+)
├── setup.py ........................ Package setup
├── src/
│   ├── models/gcn_model.py ......... GCNMassSpecPredictor
│   ├── data/
│   │   ├── features.py ............ Feature extraction
│   │   ├── dataset.py ............ Dataset class
│   │   ├── mol_parser.py ........ MSP/MOL parsing
│   │   └── dataloader.py ....... Data loading
│   ├── training/loss.py ........... ModifiedCosineLoss
│   └── utils/
│       ├── metrics.py ........... calculate_metrics()
│       └── rtx50_compat.py ... GPU setup
└── scripts/
    ├── train.py ................... Training loop
    ├── predict.py ................. Inference
    ├── preprocess_data.py ......... Data splitting
    ├── test_training.py ........... 10-sample test
    └── test_data_loading.py ....... Data loading test
```

## GPU Support

- **Primary**: RTX 50 series (Blackwell, sm_120)
- **Fallback**: Any CUDA 12.8+ compatible GPU
- **Emulation**: Can emulate sm_90 (H100) if needed
- **Mixed Precision**: FP16 supported and enabled
- **torch.compile**: Yes, enabled for optimization

## Performance Notes

- Model size: ~1.2M parameters
- Typical molecule: 5-50 atoms
- Per batch time: 10-50ms on RTX 50
- Memory per sample: ~2-4 MB

## To Add Pretraining

You would need:
1. **Data loader** for PCQM4Mv2 (4.3M molecules)
2. **Property prediction head** (HOMO/LUMO/atomization energy)
3. **Pretraining loop** (new script)
4. **Transfer learning wrapper** (freeze/unfreeze utilities)
5. **Config extensions** (pretraining + transfer sections)

Current architecture supports this but components not implemented.

## Contact/References

**Referenced Systems** (not implemented):
- NEIMS: Neural EI-MS Prediction
- ICEBERG/SCARF: MIT approaches
- Massformer: Graph Transformer variant

**Databases**:
- Current: NIST 2017 (EI-MS)
- Could integrate: PCQM4Mv2, ChEMBL, PubChem

## Summary: One Sentence

**BitSpec is a production-ready GCN for EI-MS spectrum prediction with optimized 48D+6D features, Modified Cosine Loss for fragmentation awareness, RTX 50 support, but NO pretraining infrastructure.**
