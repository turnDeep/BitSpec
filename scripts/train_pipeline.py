#!/usr/bin/env python3
# scripts/train_pipeline.py
"""
NEIMS v2.0 Complete Training Pipeline

Orchestrates 3-phase training workflow:
  Phase 1: Teacher pretraining on PCQM4Mv2 (Bond Masking)
  Phase 2: Teacher finetuning on NIST EI-MS (MC Dropout)
  Phase 3: Student distillation (Knowledge Distillation)

Usage:
  # Full pipeline
  python scripts/train_pipeline.py --config config.yaml

  # Resume from phase 2
  python scripts/train_pipeline.py --config config.yaml --start-phase 2

  # Skip pretraining (use existing checkpoint)
  python scripts/train_pipeline.py --config config.yaml --skip-pretrain
"""

import argparse
import yaml
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NEIMSPipeline:
    """NEIMS v2.0 3-Phase Training Pipeline"""

    def __init__(
        self,
        config_path: str,
        config_pretrain_path: str = None,
        start_phase: int = 1,
        skip_pretrain: bool = False,
        device: str = 'cuda'
    ):
        """
        Args:
            config_path: Main config file (config.yaml)
            config_pretrain_path: Pretraining config file (config_pretrain.yaml)
            start_phase: Starting phase (1, 2, or 3)
            skip_pretrain: Skip phase 1 (use existing checkpoint)
            device: Device to use (cuda/cpu)
        """
        self.check_environment(device)

        self.config_path = Path(config_path)
        self.config_pretrain_path = Path(config_pretrain_path or 'config_pretrain.yaml')
        self.start_phase = start_phase
        self.skip_pretrain = skip_pretrain
        self.device = device

        # Load configs
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        if self.config_pretrain_path.exists():
            with open(self.config_pretrain_path) as f:
                self.config_pretrain = yaml.safe_load(f)
        else:
            self.config_pretrain = self.config

        # Checkpoint paths
        self.teacher_pretrain_checkpoint = Path(
            self.config_pretrain.get('training', {})
            .get('teacher_pretrain', {})
            .get('checkpoint_dir', 'checkpoints/teacher')
        ) / 'best_pretrain_teacher.pt'

        self.teacher_finetune_checkpoint = Path(
            self.config['training']['teacher_finetune']['checkpoint_dir']
        ) / 'best_finetune_teacher.pt'

        self.student_checkpoint = Path(
            self.config['training']['student_distill']['checkpoint_dir']
        ) / 'best_student.pt'

    def check_environment(self, device: str):
        """Check if environment is compatible with requested device"""
        if device == 'cpu':
            return

        logger.info("Checking environment compatibility...")
        try:
            import torch
            import torch_scatter
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            # We can't really proceed if basic deps are missing, but we let the pipeline fail naturally or setup script handle it
            return

        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA requested but not available in PyTorch. Training will fail if CUDA is forced.")
            return

        # Check torch-scatter CUDA support
        try:
            # Simple CUDA test for torch-scatter
            x = torch.randn(5, 5).cuda()
            index = torch.tensor([0, 0, 1, 1, 2]).cuda()
            _ = torch_scatter.scatter_max(x, index, dim=0)
            logger.info("✅ Environment check passed: CUDA and torch-scatter are working.")
        except RuntimeError as e:
            if "Not compiled with CUDA support" in str(e):
                logger.error("❌ torch-scatter is not compiled with CUDA support!")
                logger.error("This usually happens when torch-scatter is installed for CPU but CUDA is requested.")
                logger.error("To fix this, please run the included fix script:")
                logger.error("  ./fix_torch_scatter.sh")
                logger.error("Then restart the training pipeline.")
                sys.exit(1)
            else:
                logger.warning(f"⚠️  Environment check warning: {e}")

    def phase1_teacher_pretrain(self):
        """Phase 1: Teacher Pretraining on PCQM4Mv2"""
        if self.skip_pretrain or self.start_phase > 1:
            logger.info("⏭️  Skipping Phase 1: Teacher Pretraining")
            if self.teacher_pretrain_checkpoint.exists():
                logger.info(f"Using existing checkpoint: {self.teacher_pretrain_checkpoint}")
            return

        logger.info("=" * 80)
        logger.info("Phase 1: Teacher Pretraining on PCQM4Mv2")
        logger.info("=" * 80)

        try:
            script = Path(__file__).parent / 'train_teacher.py'
            cmd = [
                sys.executable,
                str(script),
                '--config', str(self.config_pretrain_path),
                '--phase', 'pretrain',
                '--device', self.device
            ]

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)

            if result.returncode == 0:
                logger.info("✅ Phase 1 completed: Teacher pretrained")
            else:
                raise RuntimeError(f"Phase 1 failed with return code {result.returncode}")

        except Exception as e:
            logger.error(f"❌ Phase 1 failed: {e}")
            raise

    def phase2_teacher_finetune(self):
        """Phase 2: Teacher Finetuning on NIST EI-MS"""
        if self.start_phase > 2:
            logger.info("⏭️  Skipping Phase 2: Teacher Finetuning")
            if self.teacher_finetune_checkpoint.exists():
                logger.info(f"Using existing checkpoint: {self.teacher_finetune_checkpoint}")
            return

        logger.info("=" * 80)
        logger.info("Phase 2: Teacher Finetuning on NIST EI-MS")
        logger.info("=" * 80)

        try:
            script = Path(__file__).parent / 'train_teacher.py'
            cmd = [
                sys.executable,
                str(script),
                '--config', str(self.config_path),
                '--phase', 'finetune',
                '--device', self.device
            ]

            # Add pretrained checkpoint if available
            if self.teacher_pretrain_checkpoint.exists():
                cmd.extend(['--pretrained', str(self.teacher_pretrain_checkpoint)])
                logger.info(f"Using pretrained checkpoint: {self.teacher_pretrain_checkpoint}")
            else:
                logger.warning("⚠️  No pretrained checkpoint found. Training from scratch.")

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)

            if result.returncode == 0:
                logger.info("✅ Phase 2 completed: Teacher finetuned")
            else:
                raise RuntimeError(f"Phase 2 failed with return code {result.returncode}")

        except Exception as e:
            logger.error(f"❌ Phase 2 failed: {e}")
            raise

    def phase3_student_distill(self):
        """Phase 3: Student Knowledge Distillation"""
        if self.start_phase > 3:
            logger.info("⏭️  Skipping Phase 3: Student Distillation")
            return

        logger.info("=" * 80)
        logger.info("Phase 3: Student Knowledge Distillation")
        logger.info("=" * 80)

        try:
            # Check teacher checkpoint
            if not self.teacher_finetune_checkpoint.exists():
                raise FileNotFoundError(
                    f"Teacher checkpoint not found: {self.teacher_finetune_checkpoint}\n"
                    "Please complete Phase 2 first."
                )

            script = Path(__file__).parent / 'train_student.py'
            cmd = [
                sys.executable,
                str(script),
                '--config', str(self.config_path),
                '--teacher', str(self.teacher_finetune_checkpoint),
                '--device', self.device
            ]

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)

            if result.returncode == 0:
                logger.info("✅ Phase 3 completed: Student distilled")
            else:
                raise RuntimeError(f"Phase 3 failed with return code {result.returncode}")

        except Exception as e:
            logger.error(f"❌ Phase 3 failed: {e}")
            raise

    def summary(self):
        """Display pipeline summary"""
        logger.info("=" * 80)
        logger.info("🎉 NEIMS v2.0 Training Pipeline Completed!")
        logger.info("=" * 80)

        logger.info("\n📁 Generated Checkpoints:")

        if self.teacher_pretrain_checkpoint.exists():
            logger.info(f"  ✅ Teacher (Pretrained): {self.teacher_pretrain_checkpoint}")
        else:
            logger.info(f"  ❌ Teacher (Pretrained): Not found")

        if self.teacher_finetune_checkpoint.exists():
            logger.info(f"  ✅ Teacher (Finetuned):  {self.teacher_finetune_checkpoint}")
        else:
            logger.info(f"  ❌ Teacher (Finetuned):  Not found")

        if self.student_checkpoint.exists():
            logger.info(f"  ✅ Student (Distilled):  {self.student_checkpoint}")
        else:
            logger.info(f"  ❌ Student (Distilled):  Not found")

        logger.info("\n🚀 Next Steps:")
        logger.info("  1. Evaluate Student model:")
        logger.info(f"     python scripts/evaluate.py --config {self.config_path} \\")
        logger.info(f"         --checkpoint {self.student_checkpoint} --device {self.device}")
        logger.info("\n  2. Run predictions:")
        logger.info(f"     python scripts/predict.py --config {self.config_path} \\")
        logger.info(f"         --checkpoint {self.student_checkpoint} --smiles 'CC(C)O'")

    def run(self):
        """Run complete pipeline"""
        start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("NEIMS v2.0 Training Pipeline")
        logger.info("=" * 80)
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Pretrain Config: {self.config_pretrain_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Start Phase: {self.start_phase}")
        logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")

        try:
            # Phase 1: Teacher Pretraining
            if self.start_phase <= 1:
                self.phase1_teacher_pretrain()

            # Phase 2: Teacher Finetuning
            if self.start_phase <= 2:
                self.phase2_teacher_finetune()

            # Phase 3: Student Distillation
            if self.start_phase <= 3:
                self.phase3_student_distill()

            # Summary
            self.summary()

            # Elapsed time
            end_time = datetime.now()
            elapsed = end_time - start_time
            logger.info(f"\n⏱️  Total Time: {elapsed}")

        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='NEIMS v2.0 Complete Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 3-phase pipeline
  python scripts/train_pipeline.py --config config.yaml

  # Resume from phase 2 (Teacher finetuning)
  python scripts/train_pipeline.py --config config.yaml --start-phase 2

  # Skip pretraining (use existing Teacher checkpoint)
  python scripts/train_pipeline.py --config config.yaml --skip-pretrain

  # Use custom pretraining config
  python scripts/train_pipeline.py --config config.yaml \\
      --config-pretrain config_pretrain.yaml
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Main config file (config.yaml)'
    )
    parser.add_argument(
        '--config-pretrain',
        type=str,
        default=None,
        help='Pretraining config file (default: config_pretrain.yaml)'
    )
    parser.add_argument(
        '--start-phase',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='Starting phase (1: pretrain, 2: finetune, 3: distill)'
    )
    parser.add_argument(
        '--skip-pretrain',
        action='store_true',
        help='Skip phase 1 (use existing pretrained checkpoint)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda/cpu)'
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = NEIMSPipeline(
        config_path=args.config,
        config_pretrain_path=args.config_pretrain,
        start_phase=args.start_phase,
        skip_pretrain=args.skip_pretrain,
        device=args.device
    )

    pipeline.run()


if __name__ == '__main__':
    main()
