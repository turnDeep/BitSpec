#!/usr/bin/env python3
"""
10個のMOLファイルを使用したテストトレーニングスクリプト
Dev container環境での動作検証用
"""

import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
import numpy as np

# パスの追加
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import MassSpecDataset, NISTDataLoader
from src.models.gcn_model import GCNMassSpecPredictor
from src.training.loss import SpectrumLoss
from src.utils.metrics import calculate_metrics

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_training_with_10_samples():
    """10サンプルでの訓練テスト"""

    logger.info("=" * 60)
    logger.info("10個のMOLファイルを使用したテストトレーニング")
    logger.info("=" * 60)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用デバイス: {device}")

    if torch.cuda.is_available():
        logger.info(f"CUDA デバイス: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA バージョン: {torch.version.cuda}")

    # データセットの作成（10サンプルのみ）
    logger.info("\n1. データセットの作成")
    logger.info("-" * 60)

    try:
        dataset = MassSpecDataset(
            msp_file='data/NIST17.MSP',
            mol_files_dir='data/mol_files',
            max_mz=1000,
            mz_bin_size=1.0,
            cache_file='data/processed/test_cache.pkl'
        )

        logger.info(f"全データ数: {len(dataset)} サンプル")

        # 最初の10サンプルのみ使用
        num_samples = min(10, len(dataset))
        indices = list(range(num_samples))
        subset = torch.utils.data.Subset(dataset, indices)

        logger.info(f"テスト用サンプル数: {len(subset)}")

        # Train/Val/Test分割（8:1:1）
        train_size = int(len(subset) * 0.8)
        val_size = int(len(subset) * 0.1)
        test_size = len(subset) - train_size - val_size

        logger.info(f"  訓練データ: {train_size}")
        logger.info(f"  検証データ: {val_size}")
        logger.info(f"  テストデータ: {test_size}")

        # データローダーの作成
        train_loader, val_loader, test_loader = NISTDataLoader.create_dataloaders(
            dataset=subset,
            train_ratio=0.8,
            val_ratio=0.1,
            batch_size=2,  # 小さいバッチサイズ
            num_workers=0,  # CPUのみの場合は0
            seed=42
        )

        logger.info(f"  訓練バッチ数: {len(train_loader)}")
        logger.info(f"  検証バッチ数: {len(val_loader)}")

    except Exception as e:
        logger.error(f"データセット作成エラー: {e}")
        raise

    # モデルの作成
    logger.info("\n2. モデルの作成")
    logger.info("-" * 60)

    # サンプルデータからノード特徴量次元を取得
    sample_graph, _, _ = dataset[0]
    node_features = sample_graph.x.shape[1]
    edge_features = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 0

    logger.info(f"  ノード特徴量次元: {node_features}")
    logger.info(f"  エッジ特徴量次元: {edge_features}")

    model = GCNMassSpecPredictor(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=128,  # テスト用に小さく
        num_layers=3,    # テスト用に少なく
        spectrum_dim=1000,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"  総パラメータ数: {total_params:,}")
    logger.info(f"  訓練可能パラメータ数: {trainable_params:,}")

    # 損失関数とオプティマイザ
    logger.info("\n3. 訓練設定")
    logger.info("-" * 60)

    criterion = SpectrumLoss(
        loss_type="combined",
        alpha=1.0,  # MSE weight
        beta=1.0    # Cosine weight
    )

    optimizer = Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    logger.info(f"  損失関数: SpectrumLoss (combined)")
    logger.info(f"  オプティマイザ: Adam")
    logger.info(f"  学習率: 0.001")

    # トレーニング
    logger.info("\n4. トレーニング開始")
    logger.info("-" * 60)

    num_epochs = 5  # テスト用に短く

    for epoch in range(1, num_epochs + 1):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_metrics_list = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for batch_idx, (graphs, spectra, metadata) in enumerate(pbar):
            graphs = graphs.to(device)
            spectra = spectra.to(device)

            optimizer.zero_grad()

            # 順伝播
            pred_spectra = model(graphs)
            loss = criterion(pred_spectra, spectra)

            # 逆伝播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # メトリクス計算
            with torch.no_grad():
                metrics = calculate_metrics(
                    pred_spectra.cpu().numpy(),
                    spectra.cpu().numpy()
                )
                train_metrics_list.append(metrics)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cos_sim': f'{metrics["cosine_similarity"]:.4f}'
            })

        avg_train_loss = train_loss / len(train_loader)
        avg_train_metrics = {
            key: np.mean([m[key] for m in train_metrics_list])
            for key in train_metrics_list[0].keys()
        }

        # 検証フェーズ
        if len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            val_metrics_list = []

            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]")
                for graphs, spectra, metadata in pbar:
                    graphs = graphs.to(device)
                    spectra = spectra.to(device)

                    pred_spectra = model(graphs)
                    loss = criterion(pred_spectra, spectra)

                    val_loss += loss.item()

                    metrics = calculate_metrics(
                        pred_spectra.cpu().numpy(),
                        spectra.cpu().numpy()
                    )
                    val_metrics_list.append(metrics)

                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'cos_sim': f'{metrics["cosine_similarity"]:.4f}'
                    })

            avg_val_loss = val_loss / len(val_loader)
            avg_val_metrics = {
                key: np.mean([m[key] for m in val_metrics_list])
                for key in val_metrics_list[0].keys()
            }
        else:
            avg_val_loss = 0.0
            avg_val_metrics = {}

        # エポック結果の表示
        logger.info(f"\nEpoch {epoch} 結果:")
        logger.info(f"  訓練損失: {avg_train_loss:.4f}")
        logger.info(f"  訓練コサイン類似度: {avg_train_metrics['cosine_similarity']:.4f}")

        if avg_val_metrics:
            logger.info(f"  検証損失: {avg_val_loss:.4f}")
            logger.info(f"  検証コサイン類似度: {avg_val_metrics['cosine_similarity']:.4f}")

    # モデルの保存
    logger.info("\n5. モデルの保存")
    logger.info("-" * 60)

    checkpoint_dir = Path('checkpoints/test')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / 'test_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'train_loss': avg_train_loss,
        'config': {
            'node_features': node_features,
            'edge_features': edge_features,
            'hidden_dim': 128,
            'num_layers': 3,
            'output_dim': 1000
        }
    }, checkpoint_path)

    logger.info(f"モデル保存: {checkpoint_path}")

    # 完了
    logger.info("\n" + "=" * 60)
    logger.info("テストトレーニング完了！")
    logger.info("=" * 60)
    logger.info(f"最終訓練損失: {avg_train_loss:.4f}")
    logger.info(f"最終訓練コサイン類似度: {avg_train_metrics['cosine_similarity']:.4f}")

    if avg_val_metrics:
        logger.info(f"最終検証損失: {avg_val_loss:.4f}")
        logger.info(f"最終検証コサイン類似度: {avg_val_metrics['cosine_similarity']:.4f}")

    return model, avg_train_loss, avg_train_metrics


if __name__ == "__main__":
    try:
        model, loss, metrics = test_training_with_10_samples()
        logger.info("\n✓ テストトレーニングが正常に完了しました")
    except Exception as e:
        logger.error(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
