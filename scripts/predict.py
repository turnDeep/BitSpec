#!/usr/bin/env python3
# scripts/predict.py
"""
NEIMS v2.0 Prediction Script

Predict mass spectra using trained Student model (fast inference).
Can also use Teacher model for uncertainty-aware predictions.

Usage:
  # Single prediction
  python scripts/predict.py --config config.yaml \\
      --checkpoint checkpoints/student/best_student.pt --smiles 'CC(C)O'

  # Batch prediction
  python scripts/predict.py --config config.yaml \\
      --checkpoint checkpoints/student/best_student.pt --batch smiles_list.txt

  # Use Teacher model (with uncertainty)
  python scripts/predict.py --config config.yaml \\
      --checkpoint checkpoints/teacher/best_finetune_teacher.pt \\
      --model teacher --smiles 'CC(C)O'
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import yaml
import torch
import numpy as np
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Tuple, Optional

from src.models.student import StudentModel
from src.models.teacher import TeacherModel
from src.data.nist_dataset import mol_to_ecfp, mol_to_count_fp, mol_to_graph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpectrumPredictor:
    """NEIMS v2.0 Spectrum Predictor"""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        model_type: str = 'student',  # 'student' or 'teacher'
        device: str = 'cuda'
    ):
        """
        Args:
            config_path: Config file path
            checkpoint_path: Model checkpoint path
            model_type: 'student' (fast) or 'teacher' (uncertainty-aware)
            device: Device (cuda/cpu)
        """
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load model
        if model_type == 'student':
            self.model = StudentModel(self.config)
        else:  # teacher
            self.model = TeacherModel(self.config)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if model_type == 'student':
            self.model.load_state_dict(checkpoint['student_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded {model_type} model from {checkpoint_path}")
        logger.info(f"Model epoch: {checkpoint.get('epoch', 'unknown')}")

        self.max_mz = self.config['data']['max_mz']

    def predict_from_smiles(
        self,
        smiles: str,
        return_uncertainty: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict spectrum from SMILES

        Args:
            smiles: SMILES string
            return_uncertainty: Return uncertainty (Teacher only)

        Returns:
            spectrum: Predicted spectrum [501]
            uncertainty: Uncertainty (if Teacher + return_uncertainty) [501]
        """
        # Parse molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        if self.model_type == 'student':
            # Student: ECFP + Count FP
            ecfp = mol_to_ecfp(mol)
            count_fp = mol_to_count_fp(mol)

            # Concatenate
            input_fp = np.concatenate([ecfp, count_fp])
            input_tensor = torch.tensor(input_fp, dtype=torch.float32).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # Predict
            with torch.no_grad():
                spectrum = self.model(input_tensor)

            spectrum = spectrum.squeeze().cpu().numpy()
            return spectrum, None

        else:  # teacher
            # Teacher: Graph + ECFP
            graph = mol_to_graph(mol).to(self.device)
            ecfp = torch.tensor(mol_to_ecfp(mol), dtype=torch.float32).unsqueeze(0)
            ecfp = ecfp.to(self.device)

            # Add batch to graph
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)

            with torch.no_grad():
                if return_uncertainty:
                    # MC Dropout for uncertainty
                    spectrum, uncertainty = self.model.predict_with_uncertainty(
                        graph, ecfp, n_samples=30
                    )
                    return spectrum.cpu().numpy(), uncertainty.cpu().numpy()
                else:
                    spectrum = self.model(graph, ecfp)
                    return spectrum.squeeze().cpu().numpy(), None

    def predict_batch(
        self,
        smiles_list: list,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Batch prediction

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size

        Returns:
            spectra: Predicted spectra [N, 501]
        """
        all_spectra = []

        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(smiles_list) + batch_size - 1) // batch_size}")

            batch_spectra = []
            for smiles in batch_smiles:
                try:
                    spectrum, _ = self.predict_from_smiles(smiles)
                    batch_spectra.append(spectrum)
                except Exception as e:
                    logger.warning(f"Failed to predict {smiles}: {e}")
                    batch_spectra.append(np.zeros(self.max_mz + 1))

            all_spectra.extend(batch_spectra)

        return np.array(all_spectra)

    def find_top_peaks(
        self,
        spectrum: np.ndarray,
        top_n: int = 20,
        threshold: float = 0.01
    ) -> list:
        """
        Find top peaks in spectrum

        Args:
            spectrum: Spectrum array
            top_n: Number of top peaks
            threshold: Minimum intensity threshold

        Returns:
            peaks: List of (m/z, intensity) tuples
        """
        # Filter by threshold
        mask = spectrum > threshold
        indices = np.where(mask)[0]

        if len(indices) == 0:
            return []

        intensities = spectrum[mask]

        # Create (m/z, intensity) pairs
        peaks_with_intensity = list(zip(indices, intensities))

        # Sort by intensity (descending)
        peaks_with_intensity.sort(key=lambda x: x[1], reverse=True)

        # Top N
        n = min(len(peaks_with_intensity), top_n)
        peaks = [(int(mz), float(intensity)) for mz, intensity in peaks_with_intensity[:n]]

        return peaks

    def export_msp(
        self,
        smiles: str,
        output_path: str,
        compound_name: Optional[str] = None
    ):
        """
        Export prediction to MSP format

        Args:
            smiles: SMILES string
            output_path: Output file path
            compound_name: Compound name (optional)
        """
        spectrum, _ = self.predict_from_smiles(smiles)
        peaks = self.find_top_peaks(spectrum)

        mol = Chem.MolFromSmiles(smiles)
        if compound_name is None:
            compound_name = f"Predicted_{Chem.MolToInchiKey(mol)}"

        with open(output_path, 'w') as f:
            f.write(f"Name: {compound_name}\n")
            f.write(f"SMILES: {smiles}\n")
            f.write(f"InChIKey: {Chem.MolToInchiKey(mol)}\n")
            f.write(f"Num Peaks: {len(peaks)}\n")

            for mz, intensity in peaks:
                f.write(f"{mz} {intensity:.6f}; ")
            f.write("\n")

        logger.info(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='NEIMS v2.0 Mass Spectrum Prediction'
    )
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--model', type=str, default='student', choices=['student', 'teacher'],
                        help='Model type (default: student)')
    parser.add_argument('--smiles', type=str, help='SMILES string')
    parser.add_argument('--batch', type=str, help='File with SMILES list (one per line)')
    parser.add_argument('--output', type=str, default='prediction.msp', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--uncertainty', action='store_true',
                        help='Return uncertainty (Teacher only)')

    args = parser.parse_args()

    # Create predictor
    predictor = SpectrumPredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        model_type=args.model,
        device=args.device
    )

    if args.smiles:
        # Single prediction
        logger.info(f"Predicting spectrum for: {args.smiles}")

        spectrum, uncertainty = predictor.predict_from_smiles(
            args.smiles,
            return_uncertainty=args.uncertainty and args.model == 'teacher'
        )

        # Display top peaks
        peaks = predictor.find_top_peaks(spectrum)
        logger.info(f"\nTop 10 peaks:")
        for i, (mz, intensity) in enumerate(peaks[:10], 1):
            logger.info(f"  {i}. m/z {mz}: {intensity:.4f}")

        # Display uncertainty if available
        if uncertainty is not None:
            mean_uncertainty = uncertainty.mean()
            logger.info(f"\nMean uncertainty: {mean_uncertainty:.4f}")

        # Export
        predictor.export_msp(args.smiles, args.output)

    elif args.batch:
        # Batch prediction
        logger.info(f"Batch prediction from: {args.batch}")

        with open(args.batch, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]

        logger.info(f"Predicting {len(smiles_list)} spectra...")
        spectra = predictor.predict_batch(smiles_list)

        # Save results
        output_dir = Path(args.output).parent / 'batch_predictions'
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, (smiles, spectrum) in enumerate(zip(smiles_list, spectra)):
            output_file = output_dir / f'prediction_{i + 1}.msp'
            try:
                predictor.export_msp(smiles, str(output_file))
            except Exception as e:
                logger.warning(f"Failed to export {smiles}: {e}")

        logger.info(f"Saved {len(smiles_list)} predictions to {output_dir}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
