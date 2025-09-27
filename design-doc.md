# BitSpec-GCN 設計書
## 化学構造からのマススペクトル予測システム

### 1. システムアーキテクチャ

#### 1.1 全体構成

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  MOLファイル │ --> │ 前処理モジュール│ --> │ GCNエンコーダ │
└─────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ MSPファイル │ <-- │ 後処理モジュール│ <-- │BitNetデコーダ│
└─────────────┘     └──────────────┘     └──────────────┘
```

#### 1.2 モジュール構成

```python
BitSpec-GCN/
├── data/
│   ├── mol_loader.py      # MOLファイル読み込み
│   ├── msp_parser.py      # MSPファイル解析
│   └── preprocessor.py    # データ前処理
├── models/
│   ├── gcn_encoder.py     # GCNエンコーダ
│   ├── bitnet_decoder.py  # BitNetデコーダ
│   └── bitlinear.py       # BitLinearレイヤー
├── training/
│   ├── trainer.py         # 訓練ループ
│   ├── loss.py           # 損失関数
│   └── optimizer.py      # 最適化
├── evaluation/
│   ├── metrics.py        # 評価指標
│   └── visualizer.py    # 可視化
└── utils/
    ├── quantization.py   # 量子化ユーティリティ
    └── config.py        # 設定管理
```

### 2. データ設計

#### 2.1 入力データ構造

##### 分子グラフ表現
```python
class MolecularGraph:
    node_features: torch.Tensor  # [n_atoms, feature_dim]
    edge_index: torch.Tensor     # [2, n_edges]
    edge_features: torch.Tensor  # [n_edges, edge_feature_dim]
    batch: torch.Tensor          # バッチ処理用インデックス
```

##### ノード特徴量（原子）
- 原子番号（one-hot: 118次元）
- 形式電荷（-3 to +3）
- 混成軌道（sp, sp2, sp3, sp3d, sp3d2）
- 芳香族性（boolean）
- 水素結合数（0-4）
- 不飽和度

##### エッジ特徴量（結合）
- 結合タイプ（single, double, triple, aromatic）
- 共役性（boolean）
- 立体化学（cis/trans, E/Z）
- 環構造への参加

#### 2.2 出力データ構造

```python
class MassSpectrum:
    mz_values: torch.Tensor      # [n_peaks]
    intensities: torch.Tensor    # [n_peaks]
    molecular_weight: float
    base_peak: int
```

### 3. モデル設計

#### 3.1 GCNエンコーダアーキテクチャ

```python
class GCNEncoder(nn.Module):
    def __init__(self):
        self.conv_layers = nn.ModuleList([
            GCNConv(input_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, output_dim)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(3)
        ])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch):
        # メッセージパッシング
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # グラフレベル表現の集約
        x = self.conv_layers[-1](x, edge_index)
        graph_repr = global_mean_pool(x, batch)
        return graph_repr
```

#### 3.2 BitNetデコーダ設計

##### BitLinearレイヤー実装
```python
class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 学習可能パラメータ（フル精度で保存）
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(out_features))
        
    def forward(self, x):
        # 重みの三値量子化 (-1, 0, 1)
        weight_abs_mean = self.weight.abs().mean()
        weight_quant = torch.sign(self.weight)
        weight_quant[self.weight.abs() < 0.7 * weight_abs_mean] = 0
        
        # 活性化の量子化（8ビット）
        x_scale = 127.0 / x.abs().max()
        x_quant = torch.round(x * x_scale).clamp(-128, 127)
        
        # 量子化演算
        output = F.linear(x_quant / x_scale, weight_quant * self.scale.view(-1, 1))
        
        return output
```

##### デコーダアーキテクチャ
```python
class BitNetDecoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=1000):
        super().__init__()
        self.layers = nn.ModuleList([
            BitLinear(input_dim, hidden_dim),
            BitLinear(hidden_dim, hidden_dim),
            BitLinear(hidden_dim, hidden_dim),
            BitLinear(hidden_dim, output_dim * 2)  # m/z + intensity
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        self.activation = nn.SiLU()  # Swish activation
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.layer_norms[i](x)
            x = self.activation(x)
        
        # 最終層
        output = self.layers[-1](x)
        
        # m/z値と強度に分割
        mz_logits, intensity_logits = output.chunk(2, dim=-1)
        
        # m/z値の予測（1-1000の範囲）
        mz_values = torch.sigmoid(mz_logits) * 999 + 1
        
        # 強度の予測（0-100%）
        intensities = torch.softmax(intensity_logits, dim=-1) * 100
        
        return mz_values, intensities
```

### 4. 学習設計

#### 4.1 損失関数

##### 重み付きコサイン類似度損失
```python
class WeightedCosineLoss(nn.Module):
    def __init__(self, intensity_power=0.5, mz_power=1.3):
        super().__init__()
        self.intensity_power = intensity_power
        self.mz_power = mz_power
        
    def forward(self, pred_mz, pred_intensity, true_mz, true_intensity):
        # スペクトルの重み付き表現を計算
        pred_weighted = self.apply_weights(pred_mz, pred_intensity)
        true_weighted = self.apply_weights(true_mz, true_intensity)
        
        # コサイン類似度を計算
        similarity = F.cosine_similarity(pred_weighted, true_weighted, dim=-1)
        
        # 損失は1 - 類似度
        loss = 1 - similarity.mean()
        
        # ピーク位置の誤差も追加
        mz_loss = F.mse_loss(pred_mz, true_mz)
        
        return loss + 0.1 * mz_loss
    
    def apply_weights(self, mz, intensity):
        weighted_intensity = intensity ** self.intensity_power
        mz_weight = (mz / 1000) ** self.mz_power
        return weighted_intensity * mz_weight
```

#### 4.2 最適化戦略

##### 量子化認識訓練（QAT）
```python
class QuantizationAwareTrainer:
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2
        )
        
    def train_step(self, batch):
        # フォワードパス（量子化あり）
        with torch.cuda.amp.autocast():
            pred_mz, pred_intensity = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.loss_fn(pred_mz, pred_intensity, batch.y_mz, batch.y_intensity)
        
        # バックワードパス（Straight-Through Estimator使用）
        self.scaler.scale(loss).backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # パラメータ更新
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

### 5. 推論設計

#### 5.1 推論パイプライン

```python
class InferencePipeline:
    def __init__(self, model_path):
        self.model = self.load_quantized_model(model_path)
        self.preprocessor = MolecularPreprocessor()
        
    def predict(self, mol_file):
        # 1. MOLファイルから分子グラフを構築
        mol_graph = self.preprocessor.process(mol_file)
        
        # 2. 量子化モデルで推論
        with torch.no_grad():
            mz_values, intensities = self.model(
                mol_graph.x, 
                mol_graph.edge_index,
                mol_graph.batch
            )
        
        # 3. 後処理（ピーク抽出、正規化）
        spectrum = self.postprocess(mz_values, intensities)
        
        return spectrum
    
    def postprocess(self, mz_values, intensities):
        # 閾値以上の強度を持つピークのみ抽出
        mask = intensities > 0.5
        filtered_mz = mz_values[mask]
        filtered_intensity = intensities[mask]
        
        # 強度の正規化（最大値を100に）
        if len(filtered_intensity) > 0:
            filtered_intensity = filtered_intensity / filtered_intensity.max() * 100
        
        return MassSpectrum(filtered_mz, filtered_intensity)
```

### 6. 評価設計

#### 6.1 評価指標

```python
class EvaluationMetrics:
    @staticmethod
    def weighted_cosine_similarity(pred_spectrum, true_spectrum):
        # 重み付きコサイン類似度の計算
        pass
    
    @staticmethod
    def top_n_accuracy(pred_spectrum, true_spectrum, n=10):
        # トップNピークの一致率
        pass
    
    @staticmethod
    def spectral_entropy_similarity(pred_spectrum, true_spectrum):
        # スペクトルエントロピー類似度
        pass
    
    @staticmethod
    def model_size(model):
        # 量子化後のモデルサイズ（MB）
        return sum(p.numel() * 1.58 / 8 for p in model.parameters()) / 1e6
```

### 7. デプロイメント設計

#### 7.1 モデル変換

```python
class ModelExporter:
    def export_to_onnx(self, model, output_path):
        # ONNX形式へのエクスポート（エッジデバイス用）
        dummy_input = self.create_dummy_input()
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['molecular_graph'],
            output_names=['mz_values', 'intensities'],
            dynamic_axes={'molecular_graph': {0: 'batch_size'}}
        )
    
    def export_to_tflite(self, model, output_path):
        # TensorFlow Lite形式へのエクスポート（モバイル用）
        pass
```

### 8. パフォーマンス最適化

#### 8.1 メモリ最適化
- グラフバッチング: 類似サイズの分子をまとめて処理
- 勾配累積: メモリ制約下での大規模バッチ学習
- Mixed Precision Training: FP16/BF16の活用

#### 8.2 計算最適化
- カーネル融合: BitLinear演算の最適化
- グラフスパース性の活用
- キャッシュ効率的なメモリアクセスパターン

### 9. セキュリティ設計

- 入力検証: MOLファイルの妥当性チェック
- モデル改竄防止: チェックサム検証
- プライバシー保護: 推論データのログ管理

### 10. エラーハンドリング

```python
class ErrorHandler:
    @staticmethod
    def handle_invalid_mol(mol_file):
        # 不正なMOLファイルの処理
        raise ValueError(f"Invalid MOL file: {mol_file}")
    
    @staticmethod
    def handle_prediction_failure(error):
        # 予測失敗時の処理
        logger.error(f"Prediction failed: {error}")
        return default_spectrum()
```