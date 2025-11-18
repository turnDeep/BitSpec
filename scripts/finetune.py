# scripts/finetune.py
"""
事前学習済みGCNバックボーンをEI-MSタスクでファインチューニングするスクリプト
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from datetime import datetime
import sys
import math
sys.path.append(str(Path(__file__).parent.parent))

from src.models.gcn_model import GCNMassSpecPredictor
from src.training.loss import ModifiedCosineLoss
from src.data.dataset import MassSpecDataset, NISTDataLoader
from src.utils.rtx50_compat import setup_rtx50_compatibility
from src.utils.metrics import calculate_metrics

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinetuneTrainer:
    """ファインチューニングトレーナー"""

    def __init__(self, config_path: str, rebuild_cache: bool = False, max_samples: int = None, random_seed: int = 42):
        """
        Args:
            config_path: 設定ファイルのパス
            rebuild_cache: キャッシュを再構築するかどうか
            max_samples: ランダムサンプリングする最大サンプル数（Noneの場合は全て使用）
            random_seed: ランダムシード
        """
        # 設定の読み込み
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.rebuild_cache = rebuild_cache
        self.max_samples = max_samples
        self.random_seed = random_seed

        # デバイス設定とRTX 50互換性
        self.device = setup_rtx50_compatibility()
        logger.info(f"Using device: {self.device}")

        # Weights & Biasesの初期化
        if self.config.get('finetuning', {}).get('use_wandb', False):
            wandb.init(
                project=self.config['finetuning'].get('wandb_project', 'bitspec-finetune'),
                config=self.config,
                name=f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        # データローダーの作成
        logger.info("Creating dataset...")
        cache_file = str(Path(self.config['data']['output_dir']) / 'dataset_cache.pkl')

        # キャッシュ再構築フラグが有効な場合、既存のキャッシュを削除
        if self.rebuild_cache and Path(cache_file).exists():
            logger.info(f"Rebuilding cache: removing {cache_file}")
            Path(cache_file).unlink()

        dataset = MassSpecDataset(
            msp_file=self.config['data']['nist_msp_path'],
            mol_files_dir=self.config['data']['mol_files_dir'],
            max_mz=self.config['data']['max_mz'],
            mz_bin_size=self.config['data']['mz_bin_size'],
            cache_file=cache_file,
            max_samples=self.max_samples,
            random_seed=self.random_seed,
            use_functional_groups=self.config['model'].get('use_functional_groups', True)
        )

        logger.info("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = NISTDataLoader.create_dataloaders(
            dataset=dataset,
            train_ratio=self.config['data']['train_split'],
            val_ratio=self.config['data']['val_split'],
            batch_size=self.config['finetuning']['batch_size'],
            num_workers=self.config.get('finetuning', {}).get('num_workers', 4),
            prefetch_factor=self.config.get('finetuning', {}).get('prefetch_factor', 2),
            persistent_workers=self.config.get('finetuning', {}).get('persistent_workers', True)
        )

        # モデルの作成
        logger.info("Creating model...")
        self.model = GCNMassSpecPredictor(
            node_features=self.config['model']['node_features'],
            edge_features=self.config['model']['edge_features'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            spectrum_dim=self.config['data']['max_mz'],
            dropout=self.config['model']['dropout'],
            conv_type=self.config['model'].get('gcn', {}).get('conv_type', 'GCNConv'),
            pooling=self.config['model'].get('pooling', 'attention'),
            activation=self.config['model'].get('gcn', {}).get('activation', 'relu'),
            batch_norm=self.config['model'].get('gcn', {}).get('batch_norm', True),
            residual=self.config['model'].get('gcn', {}).get('residual', True),
            use_functional_groups=self.config['model'].get('use_functional_groups', True),
            num_functional_groups=self.config['model'].get('num_functional_groups', 48)
        ).to(self.device)

        # 事前学習済みチェックポイントのロード
        pretrained_checkpoint = self.config.get('finetuning', {}).get('pretrained_checkpoint', None)
        if pretrained_checkpoint and Path(pretrained_checkpoint).exists():
            logger.info(f"Loading pretrained checkpoint from {pretrained_checkpoint}")
            checkpoint = torch.load(pretrained_checkpoint, map_location=self.device, weights_only=False)

            # バックボーンの重みをロード
            if 'backbone_state_dict' in checkpoint:
                # pretrain.pyで保存された形式
                backbone_state_dict = checkpoint['backbone_state_dict']
            else:
                # 直接保存された形式
                backbone_state_dict = checkpoint

            # バックボーンの重みのみをロード（spectrum_predictorは除外）
            model_state_dict = self.model.state_dict()
            pretrained_dict = {
                k: v for k, v in backbone_state_dict.items()
                if k in model_state_dict and not k.startswith('spectrum_predictor')
            }
            model_state_dict.update(pretrained_dict)
            self.model.load_state_dict(model_state_dict)
            logger.info(f"Loaded {len(pretrained_dict)} pretrained weights")

            # バックボーンの凍結設定
            if self.config.get('finetuning', {}).get('freeze_backbone', False):
                logger.info("Freezing backbone layers...")
                for name, param in self.model.named_parameters():
                    if not name.startswith('spectrum_predictor'):
                        param.requires_grad = False
                logger.info("Backbone frozen. Only training spectrum_predictor.")
            else:
                freeze_layers = self.config.get('finetuning', {}).get('freeze_layers', 0)
                if freeze_layers > 0:
                    logger.info(f"Freezing first {freeze_layers} conv layers...")
                    for i in range(min(freeze_layers, len(self.model.conv_layers))):
                        for param in self.model.conv_layers[i].parameters():
                            param.requires_grad = False
                    logger.info(f"Froze {freeze_layers} layers")
        else:
            logger.warning("No pretrained checkpoint provided. Training from scratch.")

        # 損失関数
        self.criterion = ModifiedCosineLoss(
            tolerance=self.config['finetuning'].get('loss_tolerance', 0.1)
        )

        # オプティマイザ（異なる学習率）
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name.startswith('spectrum_predictor'):
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        param_groups = []
        if len(backbone_params) > 0:
            param_groups.append({
                'params': backbone_params,
                'lr': self.config['finetuning'].get('backbone_lr', self.config['finetuning']['learning_rate'])
            })
        if len(head_params) > 0:
            param_groups.append({
                'params': head_params,
                'lr': self.config['finetuning'].get('head_lr', self.config['finetuning']['learning_rate'])
            })

        if len(param_groups) == 0:
            raise ValueError("No trainable parameters found!")

        logger.info(f"Trainable backbone params: {len(backbone_params)}")
        logger.info(f"Trainable head params: {len(head_params)}")

        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.config['finetuning']['weight_decay']
        )

        # スケジューラの選択
        scheduler_type = self.config['finetuning'].get('scheduler_type', 'cosine_warmup')
        num_epochs = self.config['finetuning']['num_epochs']
        steps_per_epoch = len(self.train_loader)

        if scheduler_type == 'onecycle':
            # OneCycleLR: より積極的な学習率スケジューリング
            max_lr_backbone = self.config['finetuning'].get('backbone_lr', self.config['finetuning']['learning_rate'])
            max_lr_head = self.config['finetuning'].get('head_lr', self.config['finetuning']['learning_rate'])
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=[max_lr_backbone, max_lr_head] if len(param_groups) > 1 else max_lr_head,
                total_steps=num_epochs * steps_per_epoch,
                pct_start=self.config['finetuning'].get('warmup_pct', 0.3),
                div_factor=self.config['finetuning'].get('div_factor', 25.0),
                final_div_factor=self.config['finetuning'].get('final_div_factor', 10000.0),
                anneal_strategy='cos'
            )
            self.scheduler_step_on_batch = True
        elif scheduler_type == 'cosine_warmup':
            # Cosine Annealing with Warmup
            warmup_epochs = self.config['finetuning'].get('warmup_epochs', 5)
            self.warmup_epochs = warmup_epochs
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['finetuning'].get('scheduler_t0', 10),
                T_mult=self.config['finetuning'].get('scheduler_tmult', 2)
            )
            self.scheduler_step_on_batch = False
        else:
            # デフォルト: CosineAnnealingWarmRestarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['finetuning'].get('scheduler_t0', 10),
                T_mult=self.config['finetuning'].get('scheduler_tmult', 2)
            )
            self.scheduler_step_on_batch = False
            self.warmup_epochs = 0

        # ReduceLROnPlateau（追加のスケジューラー）
        self.reduce_on_plateau = None
        if self.config['finetuning'].get('use_reduce_on_plateau', False):
            self.reduce_on_plateau = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config['finetuning'].get('plateau_patience', 3)
            )

        # Stochastic Weight Averaging (SWA)
        self.use_swa = self.config['finetuning'].get('use_swa', False)
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_start = self.config['finetuning'].get('swa_start_epoch', num_epochs // 2)
            self.swa_scheduler = SWALR(
                self.optimizer,
                swa_lr=self.config['finetuning'].get('swa_lr', 0.0001)
            )
            logger.info(f"SWA enabled, starting at epoch {self.swa_start}")
        else:
            self.swa_model = None

        # Mixed Precision Training
        self.scaler = torch.amp.GradScaler('cuda') if self.config['finetuning'].get('use_amp', True) else None

        # 保存ディレクトリ
        self.save_dir = Path(self.config['finetuning'].get('checkpoint_dir', 'checkpoints/finetune'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ベストモデルの追跡
        self.best_val_loss = float('inf')
        self.best_model_path = None

    def train_epoch(self, epoch: int) -> dict:
        """1エポックのトレーニング"""
        self.model.train()
        total_loss = 0
        gradient_accumulation_steps = self.config['finetuning'].get('gradient_accumulation_steps', 1)

        # Warmup学習率の計算
        if hasattr(self, 'warmup_epochs') and epoch <= self.warmup_epochs:
            warmup_factor = epoch / max(1, self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
        elif not hasattr(self, 'warmup_epochs'):
            self.warmup_epochs = 0

        # 初回エポックでinitial_lrを保存
        if epoch == 1:
            for param_group in self.optimizer.param_groups:
                if 'initial_lr' not in param_group:
                    param_group['initial_lr'] = param_group['lr']

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (graphs, spectra, metadatas) in enumerate(pbar):
            # データをデバイスに転送
            graphs = graphs.to(self.device)
            spectra = spectra.to(self.device)

            # Mixed Precision Training
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    # 順伝播
                    pred_spectra = self.model(graphs)
                    loss = self.criterion(pred_spectra, spectra)
                    loss = loss / gradient_accumulation_steps

                # 逆伝播
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # OneCycleLRの場合はバッチごとにステップ
                    if hasattr(self, 'scheduler_step_on_batch') and self.scheduler_step_on_batch:
                        self.scheduler.step()

            else:
                # 順伝播
                pred_spectra = self.model(graphs)
                loss = self.criterion(pred_spectra, spectra)
                loss = loss / gradient_accumulation_steps

                # 逆伝播
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # OneCycleLRの場合はバッチごとにステップ
                    if hasattr(self, 'scheduler_step_on_batch') and self.scheduler_step_on_batch:
                        self.scheduler.step()

            total_loss += loss.item() * gradient_accumulation_steps

            # プログレスバーの更新
            pbar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

            # Weights & Biasesへのログ
            if self.config.get('finetuning', {}).get('use_wandb', False):
                wandb.log({
                    'train/loss': loss.item() * gradient_accumulation_steps,
                    'train/lr_backbone': self.optimizer.param_groups[0]['lr'] if len(self.optimizer.param_groups) > 1 else self.optimizer.param_groups[0]['lr'],
                    'train/lr_head': self.optimizer.param_groups[-1]['lr'],
                    'epoch': epoch
                })

        avg_loss = total_loss / len(self.train_loader)
        return {'loss': avg_loss}

    def validate(self, epoch: int) -> dict:
        """検証"""
        self.model.eval()
        total_loss = 0
        all_metrics = []

        with torch.no_grad():
            for graphs, spectra, metadatas in tqdm(self.val_loader, desc="Validation"):
                graphs = graphs.to(self.device)
                spectra = spectra.to(self.device)

                # 予測
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        pred_spectra = self.model(graphs)
                        loss = self.criterion(pred_spectra, spectra)
                else:
                    pred_spectra = self.model(graphs)
                    loss = self.criterion(pred_spectra, spectra)

                total_loss += loss.item()

                # メトリクスの計算
                metrics = calculate_metrics(
                    pred_spectra.cpu().numpy(),
                    spectra.cpu().numpy()
                )
                all_metrics.append(metrics)

        # 平均メトリクスの計算
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {
            key: sum(m[key] for m in all_metrics) / len(all_metrics)
            for key in all_metrics[0].keys()
        }
        avg_metrics['loss'] = avg_loss

        # Weights & Biasesへのログ
        if self.config.get('finetuning', {}).get('use_wandb', False):
            wandb.log({f'val/{k}': v for k, v in avg_metrics.items()})

        return avg_metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """チェックポイントの保存"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # 定期的な保存
        save_interval = self.config['finetuning'].get('save_interval', 10)
        if epoch % save_interval == 0:
            path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint: {path}")

        # ベストモデルの保存
        if is_best:
            path = self.save_dir / "best_finetuned_model.pt"
            torch.save(checkpoint, path)
            self.best_model_path = path
            logger.info(f"Saved best model: {path}")

    def train(self):
        """トレーニングの実行"""
        logger.info("Starting finetuning...")

        num_epochs = self.config['finetuning']['num_epochs']
        for epoch in range(1, num_epochs + 1):
            # トレーニング
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}")

            # 検証
            val_metrics = self.validate(epoch)
            logger.info(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_metrics['loss']:.4f}, "
                        f"Cosine Sim: {val_metrics['cosine_similarity']:.4f}")

            # スケジューラのステップ（OneCycleLR以外）
            if not (hasattr(self, 'scheduler_step_on_batch') and self.scheduler_step_on_batch):
                # Warmup期間後にスケジューラーを適用
                if epoch > self.warmup_epochs:
                    self.scheduler.step()

            # ReduceLROnPlateauの適用
            if self.reduce_on_plateau is not None:
                self.reduce_on_plateau.step(val_metrics['loss'])

            # SWAの更新
            if self.use_swa and epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()

            # チェックポイントの保存
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            self.save_checkpoint(epoch, val_metrics, is_best)

        # SWAモデルの最終処理
        if self.use_swa:
            logger.info("Updating SWA batch normalization statistics...")
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.device)
            # SWAモデルを保存
            swa_checkpoint = {
                'model_state_dict': self.swa_model.module.state_dict(),
                'config': self.config
            }
            swa_path = self.save_dir / "swa_model.pt"
            torch.save(swa_checkpoint, swa_path)
            logger.info(f"Saved SWA model: {swa_path}")

        logger.info("Finetuning completed!")
        logger.info(f"Best model saved at: {self.best_model_path}")

        # Weights & Biasesのクローズ
        if self.config.get('finetuning', {}).get('use_wandb', False):
            wandb.finish()


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='Finetune Pretrained Model on EI-MS Task')
    parser.add_argument('--config', type=str, default='config_pretrain.yaml',
                        help='Path to config file')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Rebuild the dataset cache from scratch')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to randomly select for finetuning (None = use all available)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    args = parser.parse_args()

    # トレーナーの作成と実行
    trainer = FinetuneTrainer(
        args.config,
        rebuild_cache=args.rebuild_cache,
        max_samples=args.max_samples,
        random_seed=args.random_seed
    )
    trainer.train()


if __name__ == '__main__':
    main()
