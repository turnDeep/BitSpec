#!/usr/bin/env python3
# scripts/train_pipeline.py
"""
BitSpec統合トレーニングパイプライン

PCQM4Mv2データセットのダウンロード → 事前学習 → ファインチューニング → 予測
を一つのスクリプトで実行します。
"""

import torch
import yaml
from pathlib import Path
import logging
import argparse
import sys
import subprocess
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.data.pcqm4mv2_loader import PCQM4Mv2DataLoader
from src.utils.rtx50_compat import setup_rtx50_compatibility

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BitSpecPipeline:
    """BitSpec統合トレーニングパイプライン"""

    def __init__(self, config_path: str, skip_download: bool = False,
                 skip_pretrain: bool = False, skip_finetune: bool = False,
                 pretrain_subset: int = None):
        """
        Args:
            config_path: 設定ファイルのパス
            skip_download: PCQM4Mv2ダウンロードをスキップ
            skip_pretrain: 事前学習をスキップ
            skip_finetune: ファインチューニングをスキップ
            pretrain_subset: 事前学習で使用するサンプル数（デバッグ用）
        """
        self.config_path = Path(config_path)
        self.skip_download = skip_download
        self.skip_pretrain = skip_pretrain
        self.skip_finetune = skip_finetune
        self.pretrain_subset = pretrain_subset

        # 設定の読み込み
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # デバイス設定
        self.device = setup_rtx50_compatibility()
        logger.info(f"Using device: {self.device}")

        # パスの設定
        self.data_dir = Path(self.config['pretraining']['data_path'])
        self.pretrain_checkpoint_dir = Path(self.config['pretraining']['checkpoint_dir'])
        self.finetune_checkpoint_dir = Path(self.config['finetuning']['checkpoint_dir'])

        # ディレクトリの作成
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pretrain_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.finetune_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def step1_download_pcqm4mv2(self):
        """ステップ1: PCQM4Mv2データセットのダウンロード"""
        if self.skip_download:
            logger.info("⏭️  Skipping PCQM4Mv2 download (--skip-download)")
            return

        logger.info("=" * 80)
        logger.info("ステップ1: PCQM4Mv2データセットのダウンロード")
        logger.info("=" * 80)

        try:
            # OGBを使用してPCQM4Mv2をダウンロード
            logger.info("Downloading PCQM4Mv2 dataset via OGB...")
            logger.info("This may take a while (dataset size: ~3.8 million molecules)")

            # データローダーを使用してダウンロードを実行
            # キャッシュを無効化して高速化（存在確認のみ）
            _, _, _ = PCQM4Mv2DataLoader.create_dataloaders(
                root=str(self.data_dir),
                batch_size=1,  # ダウンロードのみなので小さいバッチサイズ
                num_workers=0,
                node_feature_dim=self.config['model']['node_features'],
                edge_feature_dim=self.config['model']['edge_features'],
                use_subset=100,  # 最初の100サンプルだけロードして存在確認
                use_cache=False  # ダウンロード確認時はキャッシュ不要
            )

            logger.info("✓ PCQM4Mv2 dataset downloaded successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to download PCQM4Mv2: {e}")
            raise

    def step2_pretrain(self):
        """ステップ2: PCQM4Mv2での事前学習"""
        if self.skip_pretrain:
            logger.info("⏭️  Skipping pretraining (--skip-pretrain)")
            return

        logger.info("=" * 80)
        logger.info("ステップ2: PCQM4Mv2事前学習")
        logger.info("=" * 80)

        try:
            # 事前学習用の一時設定ファイルを作成（サブセット指定がある場合）
            config_to_use = self.config_path
            if self.pretrain_subset is not None:
                logger.info(f"Using subset of {self.pretrain_subset} samples for pretraining")
                temp_config = self.config.copy()
                temp_config['pretraining']['use_subset'] = self.pretrain_subset
                temp_config_path = self.config_path.parent / f"temp_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(temp_config, f)
                config_to_use = temp_config_path

            # pretrain.pyを実行
            pretrain_script = Path(__file__).parent / "pretrain.py"
            cmd = [sys.executable, str(pretrain_script), "--config", str(config_to_use)]

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)

            if result.returncode == 0:
                logger.info("✓ Pretraining completed successfully!")
            else:
                raise RuntimeError(f"Pretraining failed with return code {result.returncode}")

            # 一時ファイルの削除
            if self.pretrain_subset is not None and temp_config_path.exists():
                temp_config_path.unlink()

        except Exception as e:
            logger.error(f"❌ Pretraining failed: {e}")
            raise

    def step3_finetune(self):
        """ステップ3: EI-MSタスクでのファインチューニング"""
        if self.skip_finetune:
            logger.info("⏭️  Skipping finetuning (--skip-finetune)")
            return

        logger.info("=" * 80)
        logger.info("ステップ3: EI-MSタスクでのファインチューニング")
        logger.info("=" * 80)

        try:
            # 事前学習済みモデルの存在確認
            pretrained_backbone = self.pretrain_checkpoint_dir / "pretrained_backbone.pt"
            if not pretrained_backbone.exists() and not self.skip_pretrain:
                logger.warning(f"⚠️  Pretrained backbone not found at {pretrained_backbone}")
                logger.warning("Training from scratch instead...")

            # finetune.pyを実行
            finetune_script = Path(__file__).parent / "finetune.py"
            cmd = [sys.executable, str(finetune_script), "--config", str(self.config_path)]

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)

            if result.returncode == 0:
                logger.info("✓ Finetuning completed successfully!")
            else:
                raise RuntimeError(f"Finetuning failed with return code {result.returncode}")

        except Exception as e:
            logger.error(f"❌ Finetuning failed: {e}")
            raise

    def step4_summary(self):
        """ステップ4: サマリー表示"""
        logger.info("=" * 80)
        logger.info("🎉 パイプライン完了!")
        logger.info("=" * 80)

        # 保存されたモデルの確認
        pretrained_backbone = self.pretrain_checkpoint_dir / "pretrained_backbone.pt"
        pretrained_best = self.pretrain_checkpoint_dir / "best_pretrained_model.pt"
        finetuned_best = self.finetune_checkpoint_dir / "best_finetuned_model.pt"

        logger.info("\n📁 生成されたファイル:")
        if pretrained_backbone.exists():
            logger.info(f"  ✓ 事前学習済みバックボーン: {pretrained_backbone}")
        if pretrained_best.exists():
            logger.info(f"  ✓ 事前学習ベストモデル: {pretrained_best}")
        if finetuned_best.exists():
            logger.info(f"  ✓ ファインチューニング済みモデル: {finetuned_best}")

        logger.info("\n🚀 次のステップ:")
        logger.info("  予測を実行する:")
        logger.info(f"    python scripts/predict.py --checkpoint {finetuned_best} --config {self.config_path} --smiles 'CC(=O)OC1=CC=CC=C1C(=O)O'")

    def run(self):
        """パイプライン全体を実行"""
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("BitSpec統合トレーニングパイプライン開始")
        logger.info("=" * 80)
        logger.info(f"設定ファイル: {self.config_path}")
        logger.info(f"デバイス: {self.device}")
        logger.info(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")

        try:
            # ステップ1: データセットのダウンロード
            self.step1_download_pcqm4mv2()

            # ステップ2: 事前学習
            self.step2_pretrain()

            # ステップ3: ファインチューニング
            self.step3_finetune()

            # ステップ4: サマリー
            self.step4_summary()

            # 実行時間の計算
            end_time = datetime.now()
            elapsed_time = end_time - start_time
            logger.info(f"\n⏱️  総実行時間: {elapsed_time}")

        except Exception as e:
            logger.error(f"\n❌ パイプラインが失敗しました: {e}")
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='BitSpec統合トレーニングパイプライン',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 完全なパイプラインを実行（ダウンロード→事前学習→ファインチューニング）
  python scripts/train_pipeline.py --config config_pretrain.yaml

  # ダウンロードをスキップ（既にダウンロード済みの場合）
  python scripts/train_pipeline.py --config config_pretrain.yaml --skip-download

  # 事前学習をスキップ（スクラッチから学習）
  python scripts/train_pipeline.py --config config_pretrain.yaml --skip-pretrain

  # デバッグ用（小さなサブセットで事前学習）
  python scripts/train_pipeline.py --config config_pretrain.yaml --pretrain-subset 10000

  # ファインチューニングのみ実行
  python scripts/train_pipeline.py --config config_pretrain.yaml --skip-download --skip-pretrain
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config_pretrain.yaml',
        help='設定ファイルのパス（デフォルト: config_pretrain.yaml）'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='PCQM4Mv2のダウンロードをスキップ'
    )

    parser.add_argument(
        '--skip-pretrain',
        action='store_true',
        help='事前学習をスキップ（スクラッチから学習）'
    )

    parser.add_argument(
        '--skip-finetune',
        action='store_true',
        help='ファインチューニングをスキップ'
    )

    parser.add_argument(
        '--pretrain-subset',
        type=int,
        default=None,
        help='事前学習で使用するサンプル数（デバッグ用、例: 10000）'
    )

    args = parser.parse_args()

    # パイプラインの作成と実行
    pipeline = BitSpecPipeline(
        config_path=args.config,
        skip_download=args.skip_download,
        skip_pretrain=args.skip_pretrain,
        skip_finetune=args.skip_finetune,
        pretrain_subset=args.pretrain_subset
    )

    pipeline.run()


if __name__ == '__main__':
    main()
