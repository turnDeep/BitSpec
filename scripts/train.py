# scripts/train.py
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

from src.models.gcn_model import MassSpectrumGCN
from src.models.loss import CombinedSpectrumLoss
from src.data.dataloader import create_dataloaders
from src.utils.rtx50_compat import setup_rtx50_compatibility
from src.utils.metrics import calculate_metrics

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Trainer:
    """トレーニングクラス"""
    
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
        if self.config['training']['use_wandb']:
            wandb.init(
                project=self.config['training']['wandb_project'],
                config=self.config,
                name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # データローダーの作成
        logger.info("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_path=self.config['data']['train_path'],
            val_path=self.config['data']['val_path'],
            test_path=self.config['data']['test_path'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            max_mz=self.config['data']['max_mz']
        )
        
        # モデルの作成
        logger.info("Creating model...")
        self.model = MassSpectrumGCN(
            node_features=self.config['model']['node_features'],
            edge_features=self.config['model']['edge_features'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            output_dim=self.config['data']['max_mz'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # 損失関数
        self.criterion = CombinedSpectrumLoss(
            mse_weight=self.config['training']['mse_weight'],
            cosine_weight=self.config['training']['cosine_weight'],
            kl_weight=self.config['training']['kl_weight']
        )
        
        # オプティマイザ
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # スケジューラ
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['training']['scheduler_t0'],
            T_mult=self.config['training']['scheduler_tmult']
        )
        
        # Mixed Precision Training
        self.scaler = torch.amp.GradScaler('cuda') if self.config['training']['use_amp'] else None
        
        # 保存ディレクトリ
        self.save_dir = Path(self.config['training']['checkpoint_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # ベストモデルの追跡
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def train_epoch(self, epoch: int) -> dict:
        """1エポックのトレーニング"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (graphs, spectra) in enumerate(pbar):
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
            if self.config['training']['use_wandb']:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
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
            for graphs, spectra in tqdm(self.val_loader, desc="Validation"):
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
        if self.config['training']['use_wandb']:
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
        if epoch % self.config['training']['save_interval'] == 0:
            path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint: {path}")
        
        # ベストモデルの保存
        if is_best:
            path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, path)
            self.best_model_path = path
            logger.info(f"Saved best model: {path}")
    
    def train(self):
        """トレーニングの実行"""
        logger.info("Starting training...")
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            # トレーニング
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
            
            # 検証
            val_metrics = self.validate(epoch)
            logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                       f"Cosine Sim: {val_metrics['cosine_similarity']:.4f}")
            
            # スケジューラのステップ
            self.scheduler.step()
            
            # チェックポイントの保存
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(epoch, val_metrics, is_best)
        
        logger.info("Training completed!")
        logger.info(f"Best model saved at: {self.best_model_path}")
        
        # Weights & Biasesのクローズ
        if self.config['training']['use_wandb']:
            wandb.finish()

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Mass Spectrum Prediction Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # トレーナーの作成と実行
    trainer = Trainer(args.config)
    trainer.train()

if __name__ == '__main__':
    main()
