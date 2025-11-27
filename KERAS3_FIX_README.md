# Keras 3 Compatibility Fix for ALFABET

## Problem

The `precompute_bde.py` script fails when trying to use the ALFABET package because:

1. TensorFlow 2.15+ ships with Keras 3
2. Keras 3 no longer supports loading SavedModel format with `tf.keras.models.load_model()`
3. ALFABET uses this incompatible method to load its pre-trained BDE prediction model

### Error Message
```
ValueError: File format not supported: filepath=/home/devuser/.cache/pooch/model/output_model.
Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension).
Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3.
```

## Solution

A patch script has been created that modifies the ALFABET package to use `tf.keras.layers.TFSMLayer`, which is the Keras 3-compatible way to load SavedModel format models.

## Installation Steps

### 1. Install Phase 0 Requirements

```bash
# Install base dependencies
pip install -r requirements-phase0.txt

# Install ALFABET with --no-deps to avoid dependency conflicts
pip install --no-deps alfabet>=0.4.1
```

### 2. Apply the Keras 3 Compatibility Patch

```bash
# Run the patch script
python scripts/patch_alfabet_keras3.py
```

The script will:
- Locate the installed ALFABET package
- Create a backup of the original `prediction.py` file
- Apply the Keras 3 compatibility patch

### 3. Run BDE Pre-computation

```bash
# Test with a small subset (faster)
python scripts/precompute_bde.py --max-samples 1000

# Full dataset
python scripts/precompute_bde.py --max-samples 0
```

## Technical Details

The patch replaces this line in `alfabet/prediction.py`:
```python
model = tf.keras.models.load_model(os.path.dirname(model_files[0]))
```

With a Keras 3-compatible implementation:
```python
from tensorflow.keras.layers import TFSMLayer

# Load the SavedModel as a layer
tfsm_layer = TFSMLayer(model_dir, call_endpoint='serving_default')

# Create a wrapper that matches the expected API
class AlfabetModelWrapper:
    def __init__(self, tfsm_layer):
        self.tfsm_layer = tfsm_layer

    def __call__(self, inputs):
        result = self.tfsm_layer(inputs)
        if isinstance(result, dict):
            return list(result.values())[0]
        return result

    def predict(self, inputs):
        return self(inputs)

model = AlfabetModelWrapper(tfsm_layer)
```

## Verification

To verify the fix is working:
```bash
python -c "from alfabet import model as alfabet_model; print('ALFABET loaded successfully')"
```

## Rollback

If you need to revert the patch:
```bash
# The original file is backed up as prediction.py.keras2.bak
# Find the backup location
python -c "import alfabet; import os; print(os.path.dirname(alfabet.__file__))"

# Then manually restore from backup if needed
```

## Notes

- The patch creates a backup before making changes
- The patch is idempotent - running it multiple times won't cause issues
- CPU execution is recommended for ALFABET (GPU support is limited for sm_120)
- Expected runtime for full dataset: ~2-3 hours on CPU

## Troubleshooting

If the patch fails:
1. Ensure ALFABET is installed: `pip list | grep alfabet`
2. Check Python version: `python --version` (should be 3.11)
3. Verify TensorFlow version: `python -c "import tensorflow as tf; print(tf.__version__)"`
4. Check the patch script output for specific error messages

## References

- [Keras 3 Migration Guide](https://keras.io/guides/migrating_to_keras_3/)
- [TFSMLayer Documentation](https://keras.io/api/layers/core_layers/tf_sm_layer/)
- [ALFABET Paper](https://www.nature.com/articles/s41467-020-16201-z)
