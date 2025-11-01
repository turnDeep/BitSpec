# BitSpec Technical Summary: Code Architecture & Data Flow

## Quick Reference: File Organization

```
CRITICAL FILES:
├── src/models/gcn_model.py          [GCNMassSpecPredictor class, ~340 lines]
├── src/data/features.py             [MolecularFeaturizer, 48D node + 6D edge, ~300 lines]
├── src/training/loss.py             [ModifiedCosineLoss with neutral loss, ~120 lines]
├── src/data/dataset.py              [MassSpecDataset + NISTDataLoader, ~220 lines]
├── scripts/train.py                 [Trainer class, mixed precision, ~280 lines]
├── config.yaml                      [All hyperparameters, RTX 50 settings]
└── README.md                        [Project documentation, references]

DATA FLOW FILES:
├── src/data/mol_parser.py           [MSP/MOL file parsing, ~240 lines]
├── src/data/dataloader.py           [Alternative loader, ~200 lines]
└── scripts/preprocess_data.py       [Data splitting utility, ~200 lines]

UTILITY FILES:
├── src/utils/rtx50_compat.py        [CUDA/RTX50 setup, ~200 lines]
├── src/utils/metrics.py             [calculate_metrics(), ~60 lines]
└── scripts/test_training.py         [Validation script, ~280 lines]
```

## Data Processing Pipeline (Step by Step)

### Step 1: Parse NIST MSP File
```python
# src/data/mol_parser.py:NISTMSPParser.parse_file()
Input:  data/NIST17.MSP (text format)
        ├─ Name: Aspirin
        ├─ Formula: C9H8O4
        ├─ MW: 180
        ├─ ID: 200001
        ├─ Num peaks: 15
        ├─ 41 100.0
        └─ ...

Output: List[Dict] with keys:
        {'Name', 'Formula', 'MW', 'ID', 'Spectrum': [(mz, intensity), ...]}
```

### Step 2: Parse MOL File & Create RDKit Molecule
```python
# src/data/mol_parser.py:MOLParser.parse_file()
Input:  data/mol_files/ID200001.MOL (V2000/V3000 format)

Processing:
├─ Chem.MolFromMolFile() → RDKit Mol object
├─ If no 3D coords: AllChem.EmbedMolecule()
└─ If no 3D coords: AllChem.MMFFOptimizeMolecule()

Output: rdkit.Chem.Mol object with atomic/bond properties
```

### Step 3: Extract Molecular Features
```python
# src/data/features.py:MolecularFeaturizer.mol_to_graph()

NODE FEATURES (per atom):
Input:  Atom object from RDKit
Processing:
  ├─ Atomic number → 12D one-hot (H, C, N, O, F, Si, P, S, Cl, Br, I, unknown)
  ├─ Degree → 8D one-hot (0-6, unknown)
  ├─ Formal charge → 8D one-hot (-3 to +3, unknown)
  ├─ Chirality → 5D one-hot (0-3, unknown)
  ├─ # Hydrogens → 6D one-hot (0-4, unknown)
  ├─ Hybridization → 7D one-hot (SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED, unknown)
  ├─ Is Aromatic → 1D binary
  └─ Is in Ring → 1D binary
Output: 48D feature vector per atom

EDGE FEATURES (per bond):
Input:  Bond object from RDKit
Processing:
  ├─ Bond type → 4D one-hot (SINGLE, DOUBLE, TRIPLE, AROMATIC)
  ├─ Is conjugated → 1D binary
  └─ Is in ring → 1D binary
Output: 6D feature vector per bond

Graph Construction:
├─ x: [num_atoms, 48] ← stacked node features
├─ edge_index: [2, num_edges] ← undirected graph (both directions)
├─ edge_attr: [num_edges, 6] ← stacked edge features
└─ batch: [num_atoms] ← for batching multiple molecules
```

### Step 4: Normalize Mass Spectrum
```python
# src/data/mol_parser.py:NISTMSPParser.normalize_spectrum()

Input:  [(41, 100.0), (55, 50.0), (69, 25.0), ..., (180, 999.0)]

Processing:
1. Create 1000-bin array (one per m/z unit)
2. For each peak (mz, intensity):
   bin_idx = int(mz / 1.0)  # 1.0 = mz_bin_size
   spectrum_array[bin_idx] = max(spectrum_array[bin_idx], intensity)
3. Normalize: spectrum_array /= spectrum_array.max()

Output: [1000] float array, values in [0, 1]
```

### Step 5: Create PyG Data Object
```python
# src/data/dataset.py:MassSpecDataset.__getitem__()

Output: (graph_data, spectrum, metadata)
where graph_data is torch_geometric.data.Data:
{
  'x': [num_atoms, 48],          # Node features
  'edge_index': [2, num_edges],  # Edge connectivity
  'edge_attr': [num_edges, 6],   # Edge features
  'y': [1000],                   # Target spectrum
  'batch': [num_atoms],          # Batch indices
  'mol_weight': float,           # Molecular weight
  'num_atoms': int,
  'num_bonds': int
}
```

### Step 6: Batch Multiple Molecules
```python
# src/data/dataset.py:NISTDataLoader.collate_fn()

Input:  [(graph1, spectrum1, meta1), (graph2, spectrum2, meta2), ...]

Processing:
├─ Batch graphs: torch_geometric.data.Batch.from_data_list()
│  Creates:
│  ├─ x: [total_atoms_in_batch, 48]
│  ├─ edge_index: [2, total_edges_in_batch] (with node offsets)
│  ├─ edge_attr: [total_edges_in_batch, 6]
│  └─ batch: [total_atoms_in_batch] (graph assignment indices)
│
└─ Stack spectra: torch.stack() → [batch_size, 1000]

Output: (batched_graphs, batched_spectra, metadata_list)
```

## Model Architecture (Detailed)

### Forward Pass in GCNMassSpecPredictor
```python
# src/models/gcn_model.py:GCNMassSpecPredictor.forward()

Input: Data object from batched dataloader
  {x: [N, 48], edge_index: [2, E], edge_attr: [E, 6], batch: [N]}

Step 1: Node Embedding
  x = Linear(48 → 256)(x)
  x = BatchNorm1d(256)(x)
  x = ReLU(x)
  x = Dropout(0.1)(x)

Step 2: Edge Embedding (not used in current GCN, but defined)
  edge_attr = Linear(6 → 256)(edge_attr)
  edge_attr = BatchNorm1d(256)(edge_attr)
  edge_attr = ReLU(edge_attr)

Step 3: GCN Convolution Layers (5 layers)
  for i in range(5):
    # GraphConvBlock
    identity = x
    x = GCNConv(256, 256)(x, edge_index)
    x = BatchNorm1d(256)(x)
    x = ReLU(x)
    x = Dropout(0.1)(x)
    x = x + identity  # Residual connection

  Output: x still [N, 256]

Step 4: Attention Pooling (Graph-level aggregation)
  # AttentionalAggregation pools over all nodes
  # Learns which nodes are important for prediction
  gate_nn = Linear(256 → 128 → 1)  # Attention weights
  transform_nn = Linear(256 → 256)  # Feature transformation
  graph_features = AttentionalAggregation(x, batch)
  
  Output: [batch_size, 256]

Step 5: Spectrum Prediction Head
  hidden = Linear(256 → 512)(graph_features)
  hidden = BatchNorm1d(512)(hidden)
  hidden = ReLU(hidden)
  hidden = Dropout(0.1)(hidden)
  
  hidden = Linear(512 → 512)(hidden)
  hidden = BatchNorm1d(512)(hidden)
  hidden = ReLU(hidden)
  hidden = Dropout(0.1)(hidden)
  
  spectrum = Linear(512 → 1000)(hidden)
  spectrum = Sigmoid(spectrum)
  
  Output: [batch_size, 1000] in [0, 1]
```

## Training Loop (Key Features)

### Mixed Precision Training (scripts/train.py)
```python
if self.scaler:  # scaler = GradScaler('cuda') if use_amp
    with torch.amp.autocast('cuda'):
        pred_spectra = self.model(graphs)
        loss = self.criterion(pred_spectra, spectra)
    
    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    self.scaler.step(self.optimizer)
    self.scaler.update()
else:
    # Standard training
    pred_spectra = self.model(graphs)
    loss = self.criterion(pred_spectra, spectra)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    self.optimizer.step()
```

### Loss Function (src/training/loss.py)
```python
class ModifiedCosineLoss:
    def forward(pred, target, precursor_mz_diff=None):
        # Standard cosine similarity
        cosine_sim = F.cosine_similarity(pred, target, dim=1).mean()
        
        # Neutral loss consideration (if precursor_mz_diff provided)
        if precursor_mz_diff is not None:
            shifted_matching = self._compute_shifted_matching(
                pred, target, precursor_mz_diff, self.tolerance
            )
            cosine_sim = (cosine_sim + shifted_matching) / 2
        
        return 1.0 - cosine_sim  # Loss = 1 - similarity
```

## Configuration Structure (config.yaml)

### Critical Parameters
```yaml
# Model Architecture
model:
  node_features: 157  # Currently: 48 (optimized)
  edge_features: 16   # Currently: 6 (optimized)
  hidden_dim: 256
  num_layers: 5
  dropout: 0.1
  pooling: attention
  gcn:
    conv_type: GCNConv
    batch_norm: true
    residual: true

# Data
data:
  nist_msp_path: data/NIST17.MSP
  mol_files_dir: data/mol_files
  output_dir: data/processed
  max_mz: 1000
  mz_bin_size: 1.0

# Training
training:
  batch_size: 32
  num_epochs: 200
  learning_rate: 0.001
  use_amp: true           # Mixed precision
  loss_tolerance: 0.1     # For neutral loss matching
  early_stopping:
    patience: 20
    min_delta: 0.0001

# GPU
gpu:
  use_cuda: true
  mixed_precision: true
  compile: true
  rtx50:
    enable_compat: true
```

## Key Metrics (src/utils/metrics.py)

```python
def calculate_metrics(pred_spectra, true_spectra):
    # Cosine Similarity (PRIMARY METRIC)
    cosine_sim = dot(pred, true) / (norm(pred) * norm(true))
    
    # Pearson Correlation
    correlation = pearsonr(pred, true)
    
    # MSE (Mean Squared Error)
    mse = mean((pred - true) ** 2)
    
    # MAE (Mean Absolute Error)
    mae = mean(abs(pred - true))
    
    # Top-K Accuracy (top 20 peaks)
    top_k_accuracy = overlap(top_20_indices_pred, top_20_indices_true) / 20
    
    Returns: {'cosine_similarity', 'pearson_correlation', 'mse', 'mae', 'top_k_accuracy'}
```

## Important Notes on Recent Changes

### Feature Dimension Optimization (Commit 4406639)
```
OLD:  157D atoms (too many categorical combinations)
      16D bonds
NEW:  48D atoms (efficient, covers essentials)
      6D bonds

IMPACT: Smaller model, faster training, similar performance
```

### Loss Function Simplification (Commit 96f8670)
```
OLD:  Multiple loss functions (MSE, Cosine, Custom)
NEW:  ModifiedCosineLoss ONLY (cleaner, faster, EI-MS aware)

BENEFIT: Simplified training, unified metric
```

### ID Mapping Implementation (Commit b957899)
```python
# MSP ID (from NIST17.MSP) must match MOL file name
mol_file = mol_files_dir / f"ID{compound_id}.MOL"

Without this:
- Random MOL files picked → wrong spectra paired
- Training corrupted

With mapping:
- Guaranteed 1:1 correspondence
- Data integrity maintained
```

## Missing Pretraining Components

### What DOESN'T exist:
1. **src/pretraining/** - No pretraining module
2. **PCQM4Mv2 loader** - No OGB dataset integration
3. **Pretrain heads** - No property prediction heads
4. **Freeze utilities** - No layer freezing functions
5. **Transfer config** - No transfer_learning section in config.yaml
6. **Finetune scripts** - No finetune_from_pretrained.py

### What EXISTS but not integrated:
1. Model saving/loading ✓
2. Checkpoint system ✓
3. Flexible config ✓
4. Modular architecture ✓

### To add pretraining, you need:
```python
src/pretraining/
├── pcqm4mv2_loader.py    # Load OGB PCQM4Mv2
├── property_heads.py     # HOMO/LUMO/atomization heads
├── pretrain_trainer.py   # Pretraining loop
└── transfer_utils.py     # Freeze/unfreeze utilities

scripts/
├── pretrain_gcn.py       # Pretraining script
└── finetune_nist.py      # Finetuning script

config.yaml:
├── pretraining section
└── transfer_learning section
```

## References & Related Work

**NIST Database**:
- NIST 2017 Mass Spectral Library
- 900-50,000+ unique compounds (varies by subset)
- EI-MS fragmentation patterns for GC-MS

**Related Systems** (mentioned in README):
1. **NEIMS** - Neural EI-MS Prediction
2. **ICEBERG/SCARF** - MIT approaches
3. **Massformer** - Graph Transformer variant

**Datasets Used**:
- Current: NIST17.MSP (local)
- Potential: PCQM4Mv2 (4.3M molecules, DFT properties)
- Could add: ChEMBL, PubChem, DrugBank

## Performance Characteristics

**Memory Usage**:
- Model parameters: ~1.2M
- Per molecule (avg 20 atoms, 20 bonds):
  - Node features: 20 × 48 = 960 floats
  - Edge index: 2 × 20 = 40 ints
  - Edge attributes: 20 × 6 = 120 floats
  - Batched (32): ~32× memory

**Computation**:
- GCN forward: O(E × D²) where E = edges, D = hidden_dim (256)
- Pooling: O(N × D) where N = nodes
- MLP head: O(D²)
- Overall: ~10-50ms per batch on RTX 50

**Data Characteristics**:
- Typical molecule: 5-50 atoms
- Graph density: Sparse (covalent bonds only)
- Spectrum sparsity: Most m/z bins = 0
