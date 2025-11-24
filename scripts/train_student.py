#!/usr/bin/env python3
# scripts/train_student.py
"""
NEIMS v2.0 Student Training Script (Knowledge Distillation)

Usage:
  python scripts/train_student.py --config config.yaml \
      --teacher checkpoints/teacher/best_finetune_teacher.pt
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
import logging
from torch.utils.data import DataLoader

from src.models.teacher import TeacherModel
from src.models.student import StudentModel
from src.training.student_trainer import StudentTrainer
from src.data.nist_dataset import NISTDataset, collate_fn_distill

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train NEIMS v2.0 Student Model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--teacher', type=str, required=True, help='Teacher checkpoint path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize Teacher
    logger.info("Loading Teacher model...")
    teacher = TeacherModel(config)
    teacher_checkpoint = torch.load(args.teacher, map_location=args.device)
    teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher.eval()
    logger.info(f"Loaded Teacher from {args.teacher}")

    # Initialize Student
    logger.info("Initializing Student model...")
    student = StudentModel(config)
    num_params = sum(p.numel() for p in student.parameters()) / 1e6
    logger.info(f"Student model initialized: {num_params:.1f}M parameters")

    # Initialize trainer
    trainer = StudentTrainer(
        student_model=student,
        teacher_model=teacher,
        config=config,
        device=args.device
    )

    # Training configuration
    train_config = config['training']['student_distill']
    data_config = config['data']

    # Create NIST datasets for Knowledge Distillation
    # Use 'distill' mode which provides all features: graph, ecfp, count_fp, spectrum
    logger.info("Loading NIST EI-MS datasets for distillation...")

    train_dataset = NISTDataset(
        data_config=data_config,
        mode='distill',
        split='train',
        augment=True  # Apply augmentation for student training
    )
    val_dataset = NISTDataset(
        data_config=data_config,
        mode='distill',
        split='val',
        augment=False
    )

    # Create unified data loaders with collate_fn_distill
    # This collate function combines teacher and student data in a single batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,  # Shuffle for training
        num_workers=train_config['num_workers'],
        prefetch_factor=train_config.get('prefetch_factor', 2),
        pin_memory=train_config.get('pin_memory', True),
        collate_fn=collate_fn_distill
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        prefetch_factor=train_config.get('prefetch_factor', 2),
        pin_memory=train_config.get('pin_memory', True),
        collate_fn=collate_fn_distill
    )

    num_epochs = train_config['num_epochs']
    checkpoint_dir = train_config['checkpoint_dir']
    save_interval = train_config['save_interval']

    logger.info(f"Starting Student distillation for {num_epochs} epochs")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Start training
    trainer.train(train_loader, val_loader, num_epochs, checkpoint_dir, save_interval)

    logger.info("Student distillation completed!")

if __name__ == '__main__':
    main()
