#!/usr/bin/env python3
# scripts/train_student.py
"""
NEIMS v2.0 Student Training Script (Knowledge Distillation)

Usage:
  python scripts/train_student.py --config config.yaml \
      --teacher checkpoints/teacher/best_finetune_teacher.pt
"""

import argparse
import yaml
import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.teacher import TeacherModel
from src.models.student import StudentModel
from src.training.student_trainer import StudentTrainer
from src.data.nist_dataset import NISTDataset, collate_fn_teacher, collate_fn_student

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

    # Create NIST datasets for both Teacher and Student modes
    # Teacher mode: for generating soft labels with MC Dropout
    # Student mode: for training Student model
    logger.info("Loading NIST EI-MS datasets for distillation...")

    train_dataset_teacher = NISTDataset(
        data_config=data_config,
        mode='teacher',
        split='train',
        augment=False  # No augmentation for teacher predictions
    )
    val_dataset_teacher = NISTDataset(
        data_config=data_config,
        mode='teacher',
        split='val',
        augment=False
    )

    train_dataset_student = NISTDataset(
        data_config=data_config,
        mode='student',
        split='train',
        augment=True  # Apply augmentation for student training
    )
    val_dataset_student = NISTDataset(
        data_config=data_config,
        mode='student',
        split='val',
        augment=False
    )

    # Create data loaders for Teacher (generating soft labels)
    train_loader_teacher = DataLoader(
        train_dataset_teacher,
        batch_size=train_config['batch_size'],
        shuffle=False,  # Keep order consistent with student loader
        num_workers=train_config['num_workers'],
        prefetch_factor=train_config.get('prefetch_factor', 2),
        pin_memory=train_config.get('pin_memory', True),
        collate_fn=collate_fn_teacher
    )
    val_loader_teacher = DataLoader(
        val_dataset_teacher,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        prefetch_factor=train_config.get('prefetch_factor', 2),
        pin_memory=train_config.get('pin_memory', True),
        collate_fn=collate_fn_teacher
    )

    # Create data loaders for Student
    train_loader_student = DataLoader(
        train_dataset_student,
        batch_size=train_config['batch_size'],
        shuffle=False,  # Keep order consistent with teacher loader
        num_workers=train_config['num_workers'],
        prefetch_factor=train_config.get('prefetch_factor', 2),
        pin_memory=train_config.get('pin_memory', True),
        collate_fn=collate_fn_student
    )
    val_loader_student = DataLoader(
        val_dataset_student,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        prefetch_factor=train_config.get('prefetch_factor', 2),
        pin_memory=train_config.get('pin_memory', True),
        collate_fn=collate_fn_student
    )

    # Combine loaders into tuples
    train_loaders = (train_loader_teacher, train_loader_student)
    val_loaders = (val_loader_teacher, val_loader_student)

    num_epochs = train_config['num_epochs']
    checkpoint_dir = train_config['checkpoint_dir']
    save_interval = train_config['save_interval']

    logger.info(f"Starting Student distillation for {num_epochs} epochs")
    logger.info(f"Train samples: {len(train_dataset_student)}, Val samples: {len(val_dataset_student)}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Start training
    trainer.train(train_loaders, val_loaders, num_epochs, checkpoint_dir, save_interval)

    logger.info("Student distillation completed!")

if __name__ == '__main__':
    main()
