#!/usr/bin/env python3
"""
Patch alfabet package for Keras 3 compatibility.

This script patches the alfabet package to work with Keras 3 by replacing
the incompatible `tf.keras.models.load_model()` call with `tf.keras.layers.TFSMLayer`
for loading SavedModel format models.

Issue: TensorFlow 2.15+ ships with Keras 3, which no longer supports loading
SavedModel format with `load_model()`. The alfabet package uses this method
to load its pre-trained model.

Solution: Use `tf.keras.layers.TFSMLayer` to wrap the SavedModel as a Keras layer,
then create a functional wrapper that matches the expected API.
"""

import os
import sys
import site

def find_alfabet_prediction_file():
    """Find the alfabet prediction.py file in site-packages."""
    # Get all site-packages directories
    site_packages = site.getsitepackages()

    # Also check user site-packages
    user_site = site.getusersitepackages()
    if user_site:
        site_packages.append(user_site)

    for sp in site_packages:
        alfabet_pred = os.path.join(sp, 'alfabet', 'prediction.py')
        if os.path.exists(alfabet_pred):
            return alfabet_pred

    return None

def backup_file(file_path):
    """Create a backup of the original file."""
    backup_path = file_path + '.keras2.bak'
    if not os.path.exists(backup_path):
        with open(file_path, 'r') as f:
            content = f.read()
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"Created backup: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

def patch_prediction_file(file_path):
    """Patch the alfabet prediction.py file for Keras 3 compatibility."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'KERAS3_COMPAT_PATCHED' in content:
        print("File already patched for Keras 3 compatibility")
        return False

    # Find the problematic line
    old_line = 'model = tf.keras.models.load_model(os.path.dirname(model_files[0]))'

    if old_line not in content:
        print("ERROR: Expected line not found in prediction.py")
        print("The alfabet package structure may have changed.")
        return False

    # Create the patched version
    # We need to wrap the SavedModel in a functional API model
    patch_code = '''# KERAS3_COMPAT_PATCHED: Use TFSMLayer for Keras 3 compatibility
# Keras 3 doesn't support SavedModel format with load_model()
# We use TFSMLayer to wrap the SavedModel and create a functional model
model_dir = os.path.dirname(model_files[0])
try:
    # Try loading with TFSMLayer (Keras 3)
    from tensorflow.keras.layers import TFSMLayer

    # Load the SavedModel as a layer
    tfsm_layer = TFSMLayer(model_dir, call_endpoint='serving_default')

    # Create a wrapper model that matches the expected API
    import tensorflow.keras as keras

    # Define inputs based on the model's expected signature
    # The alfabet model expects atom and bond features as inputs
    # We'll create a functional model that wraps the TFSMLayer
    class AlfabetModelWrapper:
        """Wrapper for alfabet TFSMLayer to match the original model API."""
        def __init__(self, tfsm_layer):
            self.tfsm_layer = tfsm_layer

        def __call__(self, inputs):
            """Call the underlying TFSMLayer."""
            result = self.tfsm_layer(inputs)
            # The result is a dictionary with output names as keys
            # Extract the main output (usually 'output_1' or similar)
            if isinstance(result, dict):
                # Get the first output value
                return list(result.values())[0]
            return result

        def predict(self, inputs):
            """Predict method to match tf.keras.Model API."""
            return self(inputs)

    model = AlfabetModelWrapper(tfsm_layer)

except (ImportError, AttributeError):
    # Fallback to old method for Keras 2
    model = tf.keras.models.load_model(model_dir)'''

    # Replace the problematic line
    content = content.replace(old_line, patch_code)

    # Write the patched content
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"Successfully patched {file_path}")
    return True

def main():
    print("=" * 60)
    print("Alfabet Keras 3 Compatibility Patcher")
    print("=" * 60)

    # Find the alfabet prediction.py file
    pred_file = find_alfabet_prediction_file()

    if not pred_file:
        print("ERROR: Could not find alfabet/prediction.py")
        print("Please ensure alfabet is installed:")
        print("  pip install --no-deps alfabet>=0.4.1")
        sys.exit(1)

    print(f"Found alfabet prediction file: {pred_file}")

    # Create backup
    backup_file(pred_file)

    # Apply patch
    if patch_prediction_file(pred_file):
        print("\n" + "=" * 60)
        print("Patch applied successfully!")
        print("alfabet should now work with Keras 3")
        print("=" * 60)
    else:
        print("\nNo changes made.")

if __name__ == '__main__':
    main()
