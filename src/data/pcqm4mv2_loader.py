# src/data/pcqm4mv2_loader.py
"""
PCQM4Mv2データセット用のデータローダー
事前学習用の量子化学的性質を持つ大規模分子データセット
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import PCQM4Mv2
from typing import Tuple, Optional
from tqdm import tqdm


class PCQM4Mv2Wrapper(Dataset):
    """
    PCQM4Mv2データセットのラッパー
    BitSpecのモデルに適合するようにデータを変換

    パフォーマンス最適化：
    - 初回読み込み時に特徴量を事前処理してキャッシュ
    - CPUボトルネックを削減
    """

    def __init__(
        self,
        root: str = 'data/',
        split: str = 'train',
        transform=None,
        node_feature_dim: int = 48,
        edge_feature_dim: int = 6,
        use_cache: bool = False,  # デフォルトをFalseに変更（オンデマンド処理でGPU利用を優先）
        preload_cache: bool = False,  # 事前キャッシュするかどうか
    ):
        """
        Args:
            root: データセットのルートディレクトリ
            split: 'train', 'val', 'test', 'holdout'のいずれか
            transform: データ変換関数
            node_feature_dim: ノード特徴量の次元（BitSpecに合わせる）
            edge_feature_dim: エッジ特徴量の次元（BitSpecに合わせる）
            use_cache: オンデマンドでキャッシュするか（メモリ効率的）
            preload_cache: 起動時に全データを事前キャッシュするか（時間がかかる）
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.use_cache = use_cache
        self.preload_cache = preload_cache

        print(f"Loading PCQM4Mv2 dataset (split: {split})...")
        self.dataset = PCQM4Mv2(root=str(self.root), split=split)
        print(f"Loaded {len(self.dataset)} molecules")

        # キャッシュの初期化（オンデマンドキャッシング）
        self._cache = {} if use_cache else None
        if use_cache and preload_cache:
            # 事前キャッシュは時間がかかるので、明示的に有効化した場合のみ
            print(f"⚠️  Preloading cache for {split} data (this may take a while)...")
            self._preprocess_all()
            print(f"✓ Cache created for {len(self._cache)} samples")
        elif use_cache:
            print(f"✓ On-demand caching enabled (faster startup, cache builds during training)")

    def __len__(self) -> int:
        return len(self.dataset)

    def _preprocess_all(self):
        """
        全データを事前処理してキャッシュ
        CPUボトルネックを削減するための最適化
        """
        # 少量のサンプルで試す（メモリ使用量を考慮）
        # 本番環境ではバッチごとにキャッシュを構築
        cache_limit = 10000 if self.split == 'train' else len(self.dataset)

        for idx in tqdm(range(min(cache_limit, len(self.dataset))), desc="Caching"):
            data = self.dataset[idx]
            self._cache[idx] = self._adapt_features(data)

    def _adapt_features(self, data: Data) -> Data:
        """
        PCQM4Mv2のデータをBitSpecのフォーマットに適応させる

        Args:
            data: PCQM4Mv2のData object

        Returns:
            適応されたData object
        """
        # ノード特徴量の適応
        # PCQM4Mv2は9次元のノード特徴（原子番号など）
        # BitSpecは48次元を期待しているので、パディングまたは埋め込みが必要
        if data.x.shape[1] < self.node_feature_dim:
            # ゼロパディングで次元を合わせる
            padding = torch.zeros(
                data.x.shape[0],
                self.node_feature_dim - data.x.shape[1],
                dtype=data.x.dtype
            )
            data.x = torch.cat([data.x, padding], dim=1)
        elif data.x.shape[1] > self.node_feature_dim:
            # 次元が多い場合は切り詰める
            data.x = data.x[:, :self.node_feature_dim]

        # エッジ特徴量の適応
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.shape[1] < self.edge_feature_dim:
                padding = torch.zeros(
                    data.edge_attr.shape[0],
                    self.edge_feature_dim - data.edge_attr.shape[1],
                    dtype=data.edge_attr.dtype
                )
                data.edge_attr = torch.cat([data.edge_attr, padding], dim=1)
            elif data.edge_attr.shape[1] > self.edge_feature_dim:
                data.edge_attr = data.edge_attr[:, :self.edge_feature_dim]
        else:
            # エッジ特徴がない場合はダミーを作成
            num_edges = data.edge_index.shape[1]
            data.edge_attr = torch.zeros(num_edges, self.edge_feature_dim)

        return data

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        """
        Args:
            idx: インデックス

        Returns:
            graph_data: グラフデータ
            target: HOMO-LUMO gap
        """
        # キャッシュを使用する場合
        if self.use_cache and idx in self._cache:
            data = self._cache[idx]
        else:
            data = self.dataset[idx]
            # データの適応
            data = self._adapt_features(data)
            # キャッシュに追加（オンデマンドキャッシング）
            if self.use_cache:
                self._cache[idx] = data

        if self.transform:
            data = self.transform(data)

        # ターゲット（HOMO-LUMO gap）
        target = data.y if hasattr(data, 'y') and data.y is not None else torch.tensor([0.0])

        return data, target.float()


class PCQM4Mv2DataLoader:
    """PCQM4Mv2用のカスタムデータローダー"""

    @staticmethod
    def collate_fn(batch):
        """バッチ処理用のcollate関数"""
        graphs, targets = zip(*batch)

        # グラフをバッチ化（最適化：高速化のためリスト変換を最小化）
        batched_graphs = Batch.from_data_list(list(graphs))

        # ターゲットをスタック（最適化：事前にdim確認）
        batched_targets = torch.stack([t if t.dim() > 0 else t.unsqueeze(0) for t in targets]).squeeze()

        return batched_graphs, batched_targets

    @staticmethod
    def create_dataloaders(
        root: str = 'data/',
        batch_size: int = 256,
        num_workers: int = 4,
        node_feature_dim: int = 48,
        edge_feature_dim: int = 6,
        use_subset: Optional[int] = None,
        prefetch_factor: int = 2,
        use_cache: bool = False,  # デフォルトをFalseに変更
        preload_cache: bool = False,  # 事前キャッシュのオプション追加
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        訓練、検証、テストのデータローダーを作成

        Args:
            root: データセットのルートディレクトリ
            batch_size: バッチサイズ
            num_workers: ワーカー数
            node_feature_dim: ノード特徴量の次元
            edge_feature_dim: エッジ特徴量の次元
            use_subset: サブセットのサイズ（テスト用、Noneで全データ）
            prefetch_factor: 各ワーカーの先読みバッチ数
            use_cache: オンデマンドでキャッシュするか
            preload_cache: 起動時に全データを事前キャッシュするか

        Returns:
            train_loader, val_loader, test_loader
        """
        # データセットの作成
        train_dataset = PCQM4Mv2Wrapper(
            root=root,
            split='train',
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            use_cache=use_cache,
            preload_cache=preload_cache
        )

        val_dataset = PCQM4Mv2Wrapper(
            root=root,
            split='val',
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            use_cache=use_cache,
            preload_cache=preload_cache
        )

        # テストデータセット（PCQM4Mv2にはtest-devとtest-challengeがある）
        # ここではvalidをテストとしても使用
        test_dataset = val_dataset

        # サブセットを使用する場合（デバッグ・テスト用）
        if use_subset is not None:
            train_dataset = Subset(train_dataset, range(min(use_subset, len(train_dataset))))
            val_dataset = Subset(val_dataset, range(min(use_subset // 10, len(val_dataset))))
            test_dataset = Subset(test_dataset, range(min(use_subset // 10, len(test_dataset))))

        # データローダーの作成（最適化設定）
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=PCQM4Mv2DataLoader.collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            # マルチプロセスの起動方式を最適化
            multiprocessing_context='fork' if num_workers > 0 else None
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=PCQM4Mv2DataLoader.collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=PCQM4Mv2DataLoader.collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # テスト
    print("Testing PCQM4Mv2Wrapper and DataLoader...")

    # 小さなサブセットでテスト
    train_loader, val_loader, test_loader = PCQM4Mv2DataLoader.create_dataloaders(
        root='data/',
        batch_size=32,
        num_workers=0,
        use_subset=100  # 100サンプルのみ
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 最初のバッチを取得
    for graphs, targets in train_loader:
        print(f"\nBatch info:")
        print(f"  Graphs: {graphs}")
        print(f"  Node features shape: {graphs.x.shape}")
        print(f"  Edge features shape: {graphs.edge_attr.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Target range: [{targets.min():.4f}, {targets.max():.4f}]")
        break

    print("\nDataLoader test passed!")
