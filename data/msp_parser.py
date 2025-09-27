import numpy as np
import tqdm

def parse_msp_file(filepath):
    """
    Parses a NIST MSP file and yields spectrum data for each entry.

    Args:
        filepath (str): Path to the MSP file.

    Yields:
        dict: A dictionary containing the spectrum information for one entry.
              Example: {'name': '...', 'mw': '150', 'num_peaks': 20,
                        'peaks': array([[mz1, int1], [mz2, int2], ...])}
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        spectrum_info = {}
        for line in f:
            line = line.strip()

            # An empty line signifies the end of a record
            if not line:
                if spectrum_info:
                    # Post-process and yield the completed spectrum
                    if 'peaks' in spectrum_info and spectrum_info['peaks']:
                        peaks = np.array(spectrum_info['peaks'], dtype=np.float32)
                        # Normalize intensity to 100 if it's not already
                        if peaks.shape[0] > 0 and peaks[:, 1].max() > 0:
                            peaks[:, 1] = peaks[:, 1] / peaks[:, 1].max() * 100.0
                        spectrum_info['peaks'] = peaks
                    yield spectrum_info
                spectrum_info = {}
                continue

            # Lines with colons are metadata
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()

                if key == 'num_peaks':
                    try:
                        spectrum_info[key] = int(value)
                        spectrum_info['peaks'] = [] # Initialize list for peaks
                    except ValueError:
                        spectrum_info[key] = value # Keep as string if not an int
                else:
                    spectrum_info[key] = value

            # Lines without colons after 'Num Peaks' is declared are peak data
            elif 'peaks' in spectrum_info:
                # Peak data can be separated by spaces, tabs, or semicolons
                parts = line.replace(';', ' ').replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        spectrum_info['peaks'].append((mz, intensity))
                    except (ValueError, IndexError):
                        # Ignore lines that cannot be parsed as a peak pair
                        pass

        # Yield the last spectrum if the file doesn't end with a blank line
        if spectrum_info:
            if 'peaks' in spectrum_info and spectrum_info['peaks']:
                peaks = np.array(spectrum_info['peaks'], dtype=np.float32)
                if peaks.shape[0] > 0 and peaks[:, 1].max() > 0:
                    peaks[:, 1] = peaks[:, 1] / peaks[:, 1].max() * 100.0
                spectrum_info['peaks'] = peaks
            yield spectrum_info

def load_all_spectra(filepath):
    """
    Loads all spectra from an MSP file and returns them as a list.

    Args:
        filepath (str): Path to the MSP file.

    Returns:
        list[dict]: A list of all spectrum data.
    """
    print(f"Parsing spectra from {filepath}...")
    spectra = list(tqdm.tqdm(parse_msp_file(filepath), desc="Parsing MSP file"))
    print(f"Successfully parsed {len(spectra)} spectra.")
    return spectra

if __name__ == '__main__':
    # Example usage:
    # This script can be run directly to test the parser on the NIST17.MSP file.
    # Note: The full NIST17.MSP file is large and parsing may take some time.
    import os

    # Assuming the script is run from the root directory of the project
    msp_file_path = 'NIST17.MSP'

    if os.path.exists(msp_file_path):
        all_spectra = load_all_spectra(msp_file_path)

        if all_spectra:
            print("\nSuccessfully parsed all spectra.")
            print(f"Total spectra found: {len(all_spectra)}")

            # Print details of the first 3 spectra as a sample
            for i, spec in enumerate(all_spectra[:3]):
                print(f"\n--- Spectrum {i+1} ---")
                for key, value in spec.items():
                    if key != 'peaks':
                        print(f"{key}: {value}")
                if 'peaks' in spec:
                    print(f"peaks: {spec['peaks'].shape[0]} peaks, max intensity: {spec['peaks'][:, 1].max() if spec['peaks'].shape[0] > 0 else 0}")
                    print("Sample peaks:\n", spec['peaks'][:5])
        else:
            print("No spectra were parsed. The file might be empty or in an incorrect format.")
    else:
        print(f"Error: The file '{msp_file_path}' was not found in the current directory.")
        print("Please ensure the script is run from the project root and the MSP file is present.")