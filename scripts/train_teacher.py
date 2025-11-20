#!/usr/bin/env python3
# scripts/train_teacher.py
"""
NEIMS v2.0 Teacher Training Script

Usage:
  # Phase 1: Pretrain on PCQM4Mv2
  python scripts/train_teacher.py --config config_pretrain.yaml --phase pretrain

  # Phase 2: Finetune on NIST EI-MS
  python scripts/train_teacher.py --config config.yaml --phase finetune \
      --pretrained checkpoints/teacher/best_pretrain_teacher.pt
"""

import argparse
import yaml
import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.teacher import TeacherModel
from src.training.teacher_trainer import TeacherTrainer
from src.data.pcqm4m_dataset import PCQM4Mv2Dataset, collate_fn_pretrain
from src.data.nist_dataset import NISTDataset, collate_fn_teacher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train NEIMS v2.0 Teacher Model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--phase', type=str, required=True, choices=['pretrain', 'finetune'])
    parser.add_argument('--pretrained', type=str, help='Pretrained checkpoint path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = TeacherModel(config)
    
    # Load pretrained weights if provided
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained weights from {args.pretrained}")
    
    # Initialize trainer
    trainer = TeacherTrainer(
        model=model,
        config=config,
        device=args.device,
        phase=args.phase
    )

    # Training configuration
    if args.phase == 'pretrain':
        train_config = config['training']['teacher_pretrain']
        data_config = config['data']

        # Create PCQM4Mv2 datasets
        logger.info("Loading PCQM4Mv2 dataset for pretraining...")
        train_dataset = PCQM4Mv2Dataset(
            data_config=data_config,
            split='train',
            mask_ratio=0.15,
            download=True
        )
        val_dataset = PCQM4Mv2Dataset(
            data_config=data_config,
            split='val',
            mask_ratio=0.15,
            download=False
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=train_config['num_workers'],
            prefetch_factor=train_config.get('prefetch_factor', 2),
            pin_memory=train_config.get('pin_memory', True),
            collate_fn=collate_fn_pretrain
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config['num_workers'],
            prefetch_factor=train_config.get('prefetch_factor', 2),
            pin_memory=train_config.get('pin_memory', True),
            collate_fn=collate_fn_pretrain
        )

    else:  # finetune
        train_config = config['training']['teacher_finetune']
        data_config = config['data']

        # Create NIST datasets
        logger.info("Loading NIST EI-MS dataset for finetuning...")
        train_dataset = NISTDataset(
            data_config=data_config,
            mode='teacher',
            split='train',
            augment=True
        )
        val_dataset = NISTDataset(
            data_config=data_config,
            mode='teacher',
            split='val',
            augment=False
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=train_config['num_workers'],
            prefetch_factor=train_config.get('prefetch_factor', 2),
            pin_memory=train_config.get('pin_memory', True),
            collate_fn=collate_fn_teacher
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config['num_workers'],
            prefetch_factor=train_config.get('prefetch_factor', 2),
            pin_memory=train_config.get('pin_memory', True),
            collate_fn=collate_fn_teacher
        )

    num_epochs = train_config['num_epochs']
    checkpoint_dir = train_config['checkpoint_dir']
    save_interval = train_config['save_interval']

    logger.info(f"Starting {args.phase} training for {num_epochs} epochs")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Start training
    trainer.train(train_loader, val_loader, num_epochs, checkpoint_dir, save_interval)

    logger.info(f"{args.phase.capitalize()} training completed!")

if __name__ == '__main__':
    main()
