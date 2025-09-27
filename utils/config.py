import yaml
from pathlib import Path

def load_config(config_path='config.yaml'):
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

# Load the configuration once when the module is imported.
# This makes it accessible as a global variable `config` from anywhere.
try:
    config = load_config()
except (FileNotFoundError, ValueError) as e:
    print(f"CRITICAL: Could not load configuration. {e}")
    # Set config to None or an empty dict so the program doesn't crash on import
    # but fails gracefully later when a config value is accessed.
    config = {}


if __name__ == '__main__':
    # --- Example Usage & Test ---
    print("--- Testing Configuration Loader ---")

    if config:
        print("Configuration loaded successfully.")

        # Print some sample values to verify
        print("\n--- Sample Config Values ---")
        print(f"Device: {config.get('run', {}).get('device')}")
        print(f"Batch Size: {config.get('training', {}).get('batch_size')}")
        print(f"Encoder Input Dim: {config.get('model', {}).get('encoder', {}).get('input_dim')}")

        # Test nested access
        assert config['model']['encoder']['input_dim'] == 7
        assert config['training']['learning_rate'] == 0.001

        print("\nConfiguration values seem correct.")
    else:
        print("Failed to load configuration.")

    print("\n--- Config loader test completed. ---")