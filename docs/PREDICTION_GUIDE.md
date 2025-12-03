# NExtIMS v4.2: Prediction Guide

Complete guide for using the prediction tools in NExtIMS v4.2.

---

## Overview

NExtIMS v4.2 provides two prediction modes:

1. **Single Molecule Prediction**: Predict EI-MS spectrum for one molecule
2. **Batch Prediction**: Predict spectra for multiple molecules from CSV

---

## Prerequisites

- Trained QCGN2oEI_Minimal model (`models/qcgn2oei_minimal_best.pth`)
- Optional: BDE cache for faster prediction (`data/processed/bde_cache/nist17_bde_cache.h5`)

---

## Single Molecule Prediction

### Basic Usage

```bash
python scripts/predict_single.py "CCO" \
    --model models/qcgn2oei_minimal_best.pth
```

**Output**:
```
======================================================================
NExtIMS v4.2: Single Molecule Prediction Results
======================================================================
SMILES: CCO
Formula: C2H6O
Molecular Weight: 46.07 Da
Atoms: 9 (3 heavy)
----------------------------------------------------------------------
Inference Time: 45.23 ms
Max Intensity: 0.9234
Number of Peaks (>1%): 15
----------------------------------------------------------------------

Top 10 Predicted Peaks:
----------------------------------------------------------------------
Rank   m/z      Intensity    Relative %
----------------------------------------------------------------------
1      46       0.9234        100.0%
2      31       0.7821         84.7%
3      45       0.5432         58.8%
4      27       0.3211         34.8%
5      29       0.2987         32.4%
...
======================================================================
```

### With Visualization

```bash
python scripts/predict_single.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \
    --model models/qcgn2oei_minimal_best.pth \
    --output caffeine_spectrum.png \
    --visualize
```

**Output**: `caffeine_spectrum.png` with annotated spectrum plot

### With BDE Cache (Faster)

```bash
python scripts/predict_single.py "CCO" \
    --model models/qcgn2oei_minimal_best.pth \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5
```

### Example: Common Molecules

```bash
# Ethanol
python scripts/predict_single.py "CCO" \
    --model models/qcgn2oei_minimal_best.pth \
    --visualize --output ethanol.png

# Caffeine
python scripts/predict_single.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \
    --model models/qcgn2oei_minimal_best.pth \
    --visualize --output caffeine.png

# Benzene
python scripts/predict_single.py "c1ccccc1" \
    --model models/qcgn2oei_minimal_best.pth \
    --visualize --output benzene.png

# Acetone
python scripts/predict_single.py "CC(=O)C" \
    --model models/qcgn2oei_minimal_best.pth \
    --visualize --output acetone.png
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `smiles` | SMILES string (required) | - |
| `--model` | Path to trained model | (required) |
| `--bde-cache` | Path to BDE cache HDF5 | None |
| `--device` | Device (cuda/cpu) | cuda |
| `--output` | Output plot path | predicted_spectrum.png |
| `--visualize` | Generate visualization | False |
| `--top-k` | Number of top peaks to show | 10 |
| `--min-mz` | Minimum m/z | 1 |
| `--max-mz` | Maximum m/z | 1000 |

---

## Batch Prediction

### Input CSV Format

Create a CSV file with at least a `smiles` column:

```csv
smiles,id,name
CCO,mol_001,ethanol
CC(C)O,mol_002,isopropanol
CC(=O)C,mol_003,acetone
c1ccccc1,mol_004,benzene
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,mol_005,caffeine
```

Optional columns:
- `id`: Molecule identifier (auto-generated if missing)
- `name`: Molecule name (defaults to SMILES if missing)

### Basic Usage

```bash
python scripts/predict_batch.py \
    --input molecules.csv \
    --output predictions.csv \
    --model models/qcgn2oei_minimal_best.pth
```

**Output**: `predictions.csv` with results

### Output CSV Format

```csv
id,smiles,name,prediction_status,error,inference_time_ms,base_peak_mz,base_peak_intensity,num_peaks,top_10_mz,top_10_intensity
mol_001,CCO,ethanol,success,,45.2,46,0.9234,15,"[46,31,45,27,29]","[0.923,0.782,0.543,0.321,0.299]"
mol_002,CC(C)O,isopropanol,success,,42.8,45,0.8123,18,"[45,43,27,59,41]","[0.812,0.654,0.432,0.234,0.198]"
...
```

### With BDE Cache (Recommended for Large Batches)

```bash
python scripts/predict_batch.py \
    --input molecules.csv \
    --output predictions.csv \
    --model models/qcgn2oei_minimal_best.pth \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5 \
    --batch-size 64
```

### Save Spectra to NPY File

```bash
python scripts/predict_batch.py \
    --input molecules.csv \
    --output predictions.csv \
    --model models/qcgn2oei_minimal_best.pth \
    --save-spectra predictions_spectra.npy
```

**Output**: `predictions_spectra.npy` (NumPy array, shape: [N, 1000])

### Limit Number of Samples

```bash
python scripts/predict_batch.py \
    --input molecules.csv \
    --output predictions.csv \
    --model models/qcgn2oei_minimal_best.pth \
    --max-samples 100
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input CSV file | (required) |
| `--output` | Output CSV file | (required) |
| `--model` | Path to trained model | (required) |
| `--bde-cache` | Path to BDE cache HDF5 | None |
| `--device` | Device (cuda/cpu) | cuda |
| `--batch-size` | Batch size for processing | 32 |
| `--save-spectra` | Save spectra to NPY file | None |
| `--max-samples` | Max samples to process | 0 (all) |

### Example Output Summary

```
======================================================================
NExtIMS v4.2: Batch Prediction Summary
======================================================================
Total molecules: 1,000
Successful predictions: 987 (98.7%)
Failed predictions: 13 (1.3%)
----------------------------------------------------------------------
Total time: 45.23 seconds
Average time per molecule: 45.23 ms
Throughput: 22.1 molecules/second
======================================================================
```

---

## Performance Optimization

### 1. Use BDE Cache

**Without cache**: ~80 ms/molecule (includes BDE calculation)
**With cache**: ~45 ms/molecule (BDE pre-computed)

```bash
# Pre-compute BDE cache (one time)
python scripts/precompute_bde.py \
    --nist-msp data/NIST17.MSP \
    --output data/processed/bde_cache/nist17_bde_cache.h5

# Use cache in prediction
python scripts/predict_batch.py \
    --input molecules.csv \
    --output predictions.csv \
    --model models/qcgn2oei_minimal_best.pth \
    --bde-cache data/processed/bde_cache/nist17_bde_cache.h5
```

### 2. Optimize Batch Size

| GPU | Recommended Batch Size | Throughput |
|-----|------------------------|------------|
| RTX 5070 Ti (16GB) | 64-128 | ~4,000-8,000 mol/sec |
| RTX 4090 (24GB) | 128-256 | ~10,000-15,000 mol/sec |
| RTX 3090 (24GB) | 128-256 | ~8,000-12,000 mol/sec |

```bash
# For RTX 5070 Ti
python scripts/predict_batch.py \
    --input molecules.csv \
    --output predictions.csv \
    --model models/qcgn2oei_minimal_best.pth \
    --batch-size 128
```

### 3. Use CPU for Small Batches

For < 100 molecules, CPU may be faster (no GPU transfer overhead):

```bash
python scripts/predict_single.py "CCO" \
    --model models/qcgn2oei_minimal_best.pth \
    --device cpu
```

---

## Supported Molecules

### Supported Elements

C, H, O, N, F, S, P, Cl, Br, I

### Molecular Weight Range

- **Optimal**: 50-1000 Da
- **Supported**: Any (but accuracy decreases for MW > 1000 Da)

### Unsupported Molecules

Molecules containing elements outside the supported set will be rejected:

```bash
python scripts/predict_single.py "C[Si](C)(C)C" \
    --model models/qcgn2oei_minimal_best.pth

# Output:
# ERROR: Unsupported element: Si. Supported: C, H, O, N, F, S, P, Cl, Br, I
```

---

## Error Handling

### Common Errors

**1. Invalid SMILES**
```
ERROR: Invalid SMILES: CCO(
```
**Solution**: Check SMILES syntax

**2. Model Not Found**
```
ERROR: Model file not found: models/qcgn2oei_minimal_best.pth
```
**Solution**: Train model first or check path

**3. CUDA Out of Memory**
```
ERROR: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU

**4. Unsupported Element**
```
ERROR: Unsupported element: Si
```
**Solution**: Remove molecules with unsupported elements

### Batch Prediction Error Handling

Failed predictions are marked in the output CSV:

```csv
id,smiles,name,prediction_status,error,inference_time_ms,...
mol_123,C[Si](C)(C)C,silane,failed,Unsupported element: Si,0,...
```

---

## Advanced Usage

### Python API

```python
from scripts.predict_single import predict_spectrum

# Predict spectrum
spectrum, metadata = predict_spectrum(
    smiles="CCO",
    model_path="models/qcgn2oei_minimal_best.pth",
    device="cuda"
)

print(f"Max intensity: {metadata['max_intensity']:.4f}")
print(f"Base peak m/z: {metadata['base_peak_mz']}")
```

### Batch Processing with Python

```python
from scripts.predict_batch import BatchPredictor

# Initialize
predictor = BatchPredictor(
    model_path="models/qcgn2oei_minimal_best.pth",
    bde_cache_path="data/processed/bde_cache/nist17_bde_cache.h5"
)

# Predict
smiles_list = ["CCO", "CC(C)O", "CC(=O)C"]
results = predictor.predict_batch(smiles_list, batch_size=32)

# Access results
for result in results:
    if result['status'] == 'success':
        print(f"{result['smiles']}: {result['base_peak_mz']} m/z")
```

---

## Visualization Examples

### Single Molecule

```bash
python scripts/predict_single.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \
    --model models/qcgn2oei_minimal_best.pth \
    --visualize \
    --output caffeine.png
```

**Output**: `caffeine.png` with:
- Stem plot of predicted spectrum
- Base peak annotation
- SMILES in title

### Multiple Molecules

```bash
# Create a list of molecules
echo "CCO" > molecules.txt
echo "CC(C)O" >> molecules.txt
echo "CC(=O)C" >> molecules.txt

# Predict each with visualization
while read smiles; do
    name=$(echo $smiles | md5sum | cut -c1-8)
    python scripts/predict_single.py "$smiles" \
        --model models/qcgn2oei_minimal_best.pth \
        --visualize \
        --output ${name}.png
done < molecules.txt
```

---

## Troubleshooting

### Issue: Slow Prediction

**Symptoms**: < 10 molecules/sec

**Solutions**:
1. Use BDE cache
2. Increase batch size
3. Check GPU utilization: `nvidia-smi`
4. Ensure CUDA is enabled

### Issue: Inconsistent Results

**Symptoms**: Different spectra for same SMILES

**Solutions**:
1. Check model is in eval mode (automatically handled)
2. Use same BDE cache consistently
3. Verify SMILES canonicalization

### Issue: Memory Errors

**Symptoms**: CUDA out of memory or RAM errors

**Solutions**:
1. Reduce batch size
2. Process in smaller chunks
3. Use CPU for very large molecules

---

## References

- v4.2 Specification: `docs/spec_v4.2_minimal_iterative.md`
- Model Architecture: `src/models/qcgn2oei_minimal.py`
- Graph Generator: `src/data/graph_generator_minimal.py`
- Evaluation: `scripts/evaluate_minimal.py`

---

**Last Updated**: 2025-12-03
**Version**: NExtIMS v4.2
**Status**: Ready for Use
