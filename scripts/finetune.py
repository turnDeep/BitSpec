# scripts/finetune.py
"""
事前学習済みGCNバックボーンをEI-MSタスクでファインチューニングするスクリプト
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from datetime import datetime
import sys
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

    def __init__(self, config_path: str, rebuild_cache: bool = False):
        """
        Args:
            config_path: 設定ファイルのパス
            rebuild_cache: キャッシュを再構築するかどうか
        """
        # 設定の読み込み
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.rebuild_cache = rebuild_cache

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
            cache_file=cache_file
        )

        logger.info("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = NISTDataLoader.create_dataloaders(
            dataset=dataset,
            train_ratio=self.config['data']['train_split'],
            val_ratio=self.config['data']['val_split'],
            batch_size=self.config['finetuning']['batch_size'],
            num_workers=self.config.get('finetuning', {}).get('num_workers', 4)
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
            residual=self.config['model'].get('gcn', {}).get('residual', True)
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

        # スケジューラ
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['finetuning'].get('scheduler_t0', 10),
            T_mult=self.config['finetuning'].get('scheduler_tmult', 2)
        )

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

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (graphs, spectra, metadatas) in enumerate(pbar):
            # データをデバイスに転送
            graphs = graphs.to(self.device)
            spectra = spectra.to(self.device)

            # 勾配をゼロ化
            self.optimizer.zero_grad()

            # Mixed Precision Training
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    # 順伝播
                    pred_spectra = self.model(graphs)
                    loss = self.criterion(pred_spectra, spectra)

                # 逆伝播
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 順伝播
                pred_spectra = self.model(graphs)
                loss = self.criterion(pred_spectra, spectra)

                # 逆伝播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()

            # プログレスバーの更新
            pbar.set_postfix({'loss': loss.item()})

            # Weights & Biasesへのログ
            if self.config.get('finetuning', {}).get('use_wandb', False):
                wandb.log({
                    'train/loss': loss.item(),
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

            # スケジューラのステップ
            self.scheduler.step()

            # チェックポイントの保存
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            self.save_checkpoint(epoch, val_metrics, is_best)

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
    args = parser.parse_args()

    # トレーナーの作成と実行
    trainer = FinetuneTrainer(args.config, rebuild_cache=args.rebuild_cache)
    trainer.train()


if __name__ == '__main__':
    main()
