# scripts/pretrain.py
"""
PCQM4Mv2データセットでGCNバックボーンを事前学習するスクリプト
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from datetime import datetime
import sys
import time
sys.path.append(str(Path(__file__).parent.parent))

from src.models.gcn_model import GCNMassSpecPredictor, PretrainHead, MultiTaskPretrainHead
from src.data.pcqm4mv2_loader import PCQM4Mv2DataLoader
from src.utils.rtx50_compat import setup_rtx50_compatibility

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PretrainTrainer:
    """事前学習トレーナー"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定の読み込み
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # デバイス設定とRTX 50互換性
        self.device = setup_rtx50_compatibility()
        logger.info(f"Using device: {self.device}")

        # Weights & Biasesの初期化
        if self.config.get('pretraining', {}).get('use_wandb', False):
            wandb.init(
                project=self.config['pretraining'].get('wandb_project', 'bitspec-pretrain'),
                config=self.config,
                name=f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        # データローダーの作成
        logger.info("Creating PCQM4Mv2 dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = PCQM4Mv2DataLoader.create_dataloaders(
            root=self.config['pretraining']['data_path'],
            batch_size=self.config['pretraining']['batch_size'],
            num_workers=self.config['pretraining'].get('num_workers', 4),
            node_feature_dim=self.config['model']['node_features'],
            edge_feature_dim=self.config['model']['edge_features'],
            use_subset=self.config['pretraining'].get('use_subset', None),
            prefetch_factor=self.config['pretraining'].get('prefetch_factor', 2)
        )

        # GCNバックボーンの作成
        logger.info("Creating GCN backbone...")
        self.backbone = GCNMassSpecPredictor(
            node_features=self.config['model']['node_features'],
            edge_features=self.config['model']['edge_features'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            spectrum_dim=self.config['model'].get('spectrum_dim', 1000),
            dropout=self.config['model']['dropout'],
            conv_type=self.config['model'].get('gcn', {}).get('conv_type', 'GCNConv'),
            pooling=self.config['model'].get('pooling', 'attention'),
            activation=self.config['model'].get('gcn', {}).get('activation', 'relu'),
            batch_norm=self.config['model'].get('gcn', {}).get('batch_norm', True),
            residual=self.config['model'].get('gcn', {}).get('residual', True)
        ).to(self.device)

        # 事前学習ヘッドの作成
        task = self.config['pretraining'].get('task', 'homo_lumo_gap')
        if task == 'homo_lumo_gap':
            logger.info("Using single-task pretraining (HOMO-LUMO gap)")
            self.pretrain_head = PretrainHead(
                hidden_dim=self.config['model']['hidden_dim'],
                dropout=self.config['model']['dropout']
            ).to(self.device)
        elif task == 'multi_task':
            logger.info("Using multi-task pretraining")
            self.pretrain_head = MultiTaskPretrainHead(
                hidden_dim=self.config['model']['hidden_dim'],
                dropout=self.config['model']['dropout']
            ).to(self.device)
        else:
            raise ValueError(f"Unknown task: {task}")

        self.task = task

        # torch.compileの適用（パフォーマンス最適化）
        if self.config.get('gpu', {}).get('compile', False):
            compile_mode = self.config.get('gpu', {}).get('compile_mode', 'reduce-overhead')
            logger.info(f"Applying torch.compile with mode: {compile_mode}")
            self.backbone = torch.compile(self.backbone, mode=compile_mode)
            self.pretrain_head = torch.compile(self.pretrain_head, mode=compile_mode)

        # オプティマイザ
        self.optimizer = AdamW(
            list(self.backbone.parameters()) + list(self.pretrain_head.parameters()),
            lr=self.config['pretraining']['learning_rate'],
            weight_decay=self.config['pretraining']['weight_decay']
        )

        # スケジューラ
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['pretraining'].get('scheduler_t0', 10),
            T_mult=self.config['pretraining'].get('scheduler_tmult', 2)
        )

        # Mixed Precision Training
        self.scaler = torch.amp.GradScaler('cuda') if self.config['pretraining'].get('use_amp', True) else None

        # 保存ディレクトリ
        self.save_dir = Path(self.config['pretraining'].get('checkpoint_dir', 'checkpoints/pretrain'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ベストモデルの追跡
        self.best_val_loss = float('inf')
        self.best_model_path = None

        # Gradient accumulation設定
        self.gradient_accumulation_steps = self.config['pretraining'].get('gradient_accumulation_steps', 1)
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config['pretraining']['batch_size'] * self.gradient_accumulation_steps}")

    def train_epoch(self, epoch: int) -> dict:
        """1エポックのトレーニング"""
        self.backbone.train()
        self.pretrain_head.train()
        total_loss = 0
        num_batches = 0

        # GPU使用率モニタリング用
        batch_times = []
        gpu_utils = []
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (graphs, targets) in enumerate(pbar):
            batch_start = time.time()
            # データをデバイスに転送
            graphs = graphs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # NaN/Infのターゲットをスキップ
            valid_mask = torch.isfinite(targets)
            if not valid_mask.all():
                graphs = graphs
                targets = targets[valid_mask]
                if targets.numel() == 0:
                    continue

            # 勾配をゼロ化（gradient accumulationの最初のステップでのみ）
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            # Mixed Precision Training
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    # GCNでグラフ表現を抽出
                    graph_features = self.backbone.extract_graph_features(graphs)

                    # ターゲット予測
                    if self.task == 'homo_lumo_gap':
                        pred = self.pretrain_head(graph_features).squeeze()
                        # valid_maskでフィルタリング
                        if valid_mask.all():
                            loss = F.mse_loss(pred, targets)
                        else:
                            pred = pred[valid_mask]
                            loss = F.mse_loss(pred, targets)
                    else:  # multi_task
                        homo_lumo, dipole, energy = self.pretrain_head(graph_features)
                        # マルチタスク損失（ここではHOMO-LUMOのみ使用）
                        loss = F.mse_loss(homo_lumo.squeeze(), targets)

                    # Gradient accumulationのためにlossをスケール
                    loss = loss / self.gradient_accumulation_steps

                # 逆伝播
                self.scaler.scale(loss).backward()

                # Gradient accumulationのステップに達したらオプティマイザを更新
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.backbone.parameters()) + list(self.pretrain_head.parameters()),
                        1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # GCNでグラフ表現を抽出
                graph_features = self.backbone.extract_graph_features(graphs)

                # ターゲット予測
                if self.task == 'homo_lumo_gap':
                    pred = self.pretrain_head(graph_features).squeeze()
                    if valid_mask.all():
                        loss = F.mse_loss(pred, targets)
                    else:
                        pred = pred[valid_mask]
                        loss = F.mse_loss(pred, targets)
                else:  # multi_task
                    homo_lumo, dipole, energy = self.pretrain_head(graph_features)
                    loss = F.mse_loss(homo_lumo.squeeze(), targets)

                # Gradient accumulationのためにlossをスケール
                loss = loss / self.gradient_accumulation_steps

                # 逆伝播
                loss.backward()

                # Gradient accumulationのステップに達したらオプティマイザを更新
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.backbone.parameters()) + list(self.pretrain_head.parameters()),
                        1.0
                    )
                    self.optimizer.step()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # バッチ処理時間の記録
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # GPU使用率の取得（10バッチごと）
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                try:
                    gpu_util = torch.cuda.utilization(self.device)
                    gpu_utils.append(gpu_util)
                except:
                    pass

            # プログレスバーの更新（GPU使用率を追加）
            postfix = {'loss': loss.item() * self.gradient_accumulation_steps}
            if gpu_utils:
                postfix['GPU%'] = f"{gpu_utils[-1]}"
            if batch_times:
                postfix['batch/s'] = f"{1.0/batch_time:.2f}"
            pbar.set_postfix(postfix)

            # Weights & Biasesへのログ
            if self.config.get('pretraining', {}).get('use_wandb', False):
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/batch_time': batch_time,
                    'train/gpu_util': gpu_utils[-1] if gpu_utils else 0,
                    'epoch': epoch
                })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # エポック統計の出力
        if batch_times:
            avg_batch_time = sum(batch_times) / len(batch_times)
            throughput = self.config['pretraining']['batch_size'] / avg_batch_time
            logger.info(f"  Avg batch time: {avg_batch_time:.3f}s, Throughput: {throughput:.1f} samples/s")

        if gpu_utils:
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
            logger.info(f"  Avg GPU utilization: {avg_gpu_util:.1f}%")

        return {'loss': avg_loss}

    def validate(self, epoch: int) -> dict:
        """検証"""
        self.backbone.eval()
        self.pretrain_head.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for graphs, targets in tqdm(self.val_loader, desc="Validation"):
                graphs = graphs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # NaN/Infのターゲットをスキップ
                valid_mask = torch.isfinite(targets)
                if not valid_mask.all():
                    targets = targets[valid_mask]
                    if targets.numel() == 0:
                        continue

                # 予測
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        graph_features = self.backbone.extract_graph_features(graphs)

                        if self.task == 'homo_lumo_gap':
                            pred = self.pretrain_head(graph_features).squeeze()
                            if valid_mask.all():
                                loss = F.mse_loss(pred, targets)
                            else:
                                pred = pred[valid_mask]
                                loss = F.mse_loss(pred, targets)
                        else:  # multi_task
                            homo_lumo, dipole, energy = self.pretrain_head(graph_features)
                            loss = F.mse_loss(homo_lumo.squeeze(), targets)
                else:
                    graph_features = self.backbone.extract_graph_features(graphs)

                    if self.task == 'homo_lumo_gap':
                        pred = self.pretrain_head(graph_features).squeeze()
                        if valid_mask.all():
                            loss = F.mse_loss(pred, targets)
                        else:
                            pred = pred[valid_mask]
                            loss = F.mse_loss(pred, targets)
                    else:  # multi_task
                        homo_lumo, dipole, energy = self.pretrain_head(graph_features)
                        loss = F.mse_loss(homo_lumo.squeeze(), targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = {'loss': avg_loss}

        # Weights & Biasesへのログ
        if self.config.get('pretraining', {}).get('use_wandb', False):
            wandb.log({f'val/{k}': v for k, v in metrics.items()})

        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """チェックポイントの保存"""
        checkpoint = {
            'epoch': epoch,
            'backbone_state_dict': self.backbone.state_dict(),
            'pretrain_head_state_dict': self.pretrain_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # 定期的な保存
        save_interval = self.config['pretraining'].get('save_interval', 10)
        if epoch % save_interval == 0:
            path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint: {path}")

        # ベストモデルの保存
        if is_best:
            path = self.save_dir / "best_pretrained_model.pt"
            torch.save(checkpoint, path)
            self.best_model_path = path
            logger.info(f"Saved best model: {path}")

            # バックボーンのみも保存（ファインチューニング用）
            backbone_path = self.save_dir / "pretrained_backbone.pt"
            torch.save(self.backbone.state_dict(), backbone_path)
            logger.info(f"Saved backbone weights: {backbone_path}")

    def train(self):
        """トレーニングの実行"""
        logger.info("Starting pretraining...")
        logger.info(f"Task: {self.task}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")

        num_epochs = self.config['pretraining']['num_epochs']
        for epoch in range(1, num_epochs + 1):
            # トレーニング
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}")

            # 検証
            val_metrics = self.validate(epoch)
            logger.info(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_metrics['loss']:.4f}")

            # スケジューラのステップ
            self.scheduler.step()

            # チェックポイントの保存
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            self.save_checkpoint(epoch, val_metrics, is_best)

        logger.info("Pretraining completed!")
        logger.info(f"Best model saved at: {self.best_model_path}")

        # Weights & Biasesのクローズ
        if self.config.get('pretraining', {}).get('use_wandb', False):
            wandb.finish()


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='Pretrain GCN Backbone on PCQM4Mv2')
    parser.add_argument('--config', type=str, default='config_pretrain.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # トレーナーの作成と実行
    trainer = PretrainTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
