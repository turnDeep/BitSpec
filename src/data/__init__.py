# src/data/__init__.py
"""
NEIMS v2.0 Data Module

Datasets and utilities for NEIMS v2.0 training and evaluation.
"""

from .nist_dataset import (
    NISTDataset,
    collate_fn_teacher,
    collate_fn_student,
    parse_msp_file,
    peaks_to_spectrum,
    mol_to_graph,
    mol_to_ecfp,
    mol_to_count_fp
)

from .pcqm4m_dataset import (
    PCQM4Mv2Dataset,
    collate_fn_pretrain,
    download_pcqm4mv2,
    mol_to_graph_with_mask
)

from .preprocessing import (
    validate_smiles,
    canonicalize_smiles,
    filter_by_molecular_weight,
    filter_by_num_atoms,
    normalize_spectrum,
    remove_noise_peaks,
    peaks_to_spectrum_array,
    spectrum_to_peaks,
    compute_spectrum_statistics,
    compute_molecular_descriptors,
    split_dataset,
    batch_to_device
)

from .augmentation import (
    label_distribution_smoothing,
    isotope_substitution,
    generate_conformers
)

__all__ = [
    # NIST Dataset
    'NISTDataset',
    'collate_fn_teacher',
    'collate_fn_student',
    'parse_msp_file',
    'peaks_to_spectrum',
    'mol_to_graph',
    'mol_to_ecfp',
    'mol_to_count_fp',

    # PCQM4Mv2 Dataset
    'PCQM4Mv2Dataset',
    'collate_fn_pretrain',
    'download_pcqm4mv2',
    'mol_to_graph_with_mask',

    # Preprocessing
    'validate_smiles',
    'canonicalize_smiles',
    'filter_by_molecular_weight',
    'filter_by_num_atoms',
    'normalize_spectrum',
    'remove_noise_peaks',
    'peaks_to_spectrum_array',
    'spectrum_to_peaks',
    'compute_spectrum_statistics',
    'compute_molecular_descriptors',
    'split_dataset',
    'batch_to_device',

    # Augmentation
    'label_distribution_smoothing',
    'isotope_substitution',
    'generate_conformers',
]
