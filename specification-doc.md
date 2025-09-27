# BitSpec-GCN 仕様書
## 化学構造からのマススペクトル予測システム

### 1. API仕様

#### 1.1 予測API

##### エンドポイント
```
POST /api/v1/predict
```

##### リクエスト
```json
{
  "mol_file": "string (base64 encoded MOL file)",
  "parameters": {
    "confidence_threshold": 0.5,
    "max_peaks": 100,
    "mz_range": [1, 1000]
  }
}
```

##### レスポンス
```json
{
  "status": "success",
  "spectrum": {
    "peaks": [
      {"mz": 43.0, "intensity": 100.0},
      {"mz": 71.0, "intensity": 85.2},
      {"mz": 99.0, "intensity": 45.6}
    ],
    "metadata": {
      "molecular_weight": 322.0,
      "base_peak": 43,
      "num_peaks": 25,
      "prediction_confidence": 0.87
    }
  },
  "processing_time_ms": 45
}
```

#### 1.2 バッチ予測API

##### エンドポイント
```
POST /api/v1/predict/batch
```

##### リクエスト
```json
{
  "molecules": [
    {"id": "mol_1", "mol_file": "..."},
    {"id": "mol_2", "mol_file": "..."}
  ],
  "parameters": {
    "parallel_jobs": 4,
    "timeout_seconds": 60
  }
}
```

### 2. データ形式仕様

#### 2.1 入力MOLファイル形式

```
1,4-Benzenediol, 2,3,5,6-tetrafluoro-, bis(3-methylbutyl) ether
-LIB2NIST-05049912252D 1   1.0         0.0         0.0
CAS rn = 1000395365, Spec ID = 200001
 22 22  0  0  0  0  0  0  0  0  1 V2000
    0.6921    1.0179    0.0000 C   0  0  0     0  0  0  0  0  0
    0.6921    0.2036    0.0000 C   0  0  0     0  0  0  0  0  0
    ...
  1  2  2  0  0  0  0
  1  3  1  0  0  0  0
    ...
M  END
$$$$
```

#### 2.2 出力MSPファイル形式

```
Name: Predicted spectrum
InChIKey: COOGXJIFRJFYQY-UHFFFAOYSA-N
Formula: C16H22F4O2
MW: 322
ExactMass: 322.155592
Num peaks: 25
15 10
26 20
27 650
28 180
43 9999
...
```

### 3. モデル仕様

#### 3.1 GCNエンコーダ仕様

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| 入力次元 | 128 | ノード特徴量の次元 |
| 隠れ層次元 | 256 | 隠れ層のノード特徴次元 |
| 出力次元 | 256 | グラフ埋め込みの次元 |
| GCN層数 | 4 | グラフ畳み込み層の数 |
| 活性化関数 | ReLU | 各層の活性化関数 |
| ドロップアウト率 | 0.2 | 訓練時のドロップアウト |
| 正規化 | BatchNorm | バッチ正規化 |
| プーリング | Global Mean Pool | グラフレベル集約 |

#### 3.2 BitNetデコーダ仕様

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| 入力次元 | 256 | エンコーダ出力次元 |
| 隠れ層次元 | 512 | デコーダ隠れ層次元 |
| 出力次元 | 2000 | m/z + 強度の予測次元 |
| BitLinear層数 | 4 | 量子化線形層の数 |
| 量子化ビット | 1.58 | 重み量子化（-1, 0, 1） |
| 活性化量子化 | 8ビット | 活性化の量子化ビット数 |
| 活性化関数 | SiLU | Swish活性化関数 |
| 正規化 | LayerNorm | 層正規化 |

### 4. 訓練仕様

#### 4.1 ハイパーパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| バッチサイズ | 32 | 訓練時のバッチサイズ |
| 学習率 | 1e-3 | 初期学習率 |
| エポック数 | 100 | 訓練エポック数 |
| Weight Decay | 1e-5 | 重み減衰係数 |
| 勾配クリッピング | 1.0 | 最大勾配ノルム |
| ウォームアップ | 5エポック | 学習率ウォームアップ期間 |
| 最適化手法 | AdamW | オプティマイザ |
| スケジューラ | CosineAnnealing | 学習率スケジューラ |

#### 4.2 データ分割

```python
{
    "train": 0.8,    # 80% 訓練データ
    "validation": 0.1,  # 10% 検証データ
    "test": 0.1      # 10% テストデータ
}
```

### 5. 量子化仕様

#### 5.1 重み量子化

```python
def quantize_weights(weights):
    """
    重みを三値（-1, 0, 1）に量子化
    
    Args:
        weights: フル精度の重みテンソル
    
    Returns:
        量子化された重みとスケーリング係数
    """
    # 絶対値の平均を計算
    abs_mean = weights.abs().mean()
    
    # 閾値の設定
    threshold = 0.7 * abs_mean
    
    # 三値量子化
    quantized = torch.sign(weights)
    quantized[weights.abs() < threshold] = 0
    
    # スケーリング係数
    scale = weights.abs().mean() / (quantized != 0).float().mean()
    
    return quantized, scale
```

#### 5.2 活性化量子化

```python
def quantize_activations(x, bits=8):
    """
    活性化を指定ビット数に量子化
    
    Args:
        x: 活性化テンソル
        bits: 量子化ビット数
    
    Returns:
        量子化された活性化
    """
    qmax = 2**(bits-1) - 1
    qmin = -2**(bits-1)
    
    scale = qmax / x.abs().max()
    x_quant = torch.round(x * scale).clamp(qmin, qmax)
    
    return x_quant, scale
```

### 6. 評価メトリクス仕様

#### 6.1 重み付きコサイン類似度

```python
def weighted_cosine_similarity(pred_spectrum, true_spectrum, 
                              intensity_power=0.5, mz_power=1.3):
    """
    重み付きコサイン類似度の計算
    
    Args:
        pred_spectrum: 予測スペクトル
        true_spectrum: 真のスペクトル
        intensity_power: 強度の重み係数
        mz_power: m/z値の重み係数
    
    Returns:
        類似度スコア (0-1)
    """
    # ピークのマッチング
    matched_peaks = match_peaks(pred_spectrum, true_spectrum, tolerance=0.5)
    
    # 重み付き強度の計算
    pred_weighted = []
    true_weighted = []
    
    for (pred_mz, pred_int), (true_mz, true_int) in matched_peaks:
        # 強度の重み付き
        pred_w = (pred_int ** intensity_power) * ((pred_mz/1000) ** mz_power)
        true_w = (true_int ** intensity_power) * ((true_mz/1000) ** mz_power)
        
        pred_weighted.append(pred_w)
        true_weighted.append(true_w)
    
    # コサイン類似度
    pred_vec = torch.tensor(pred_weighted)
    true_vec = torch.tensor(true_weighted)
    
    similarity = F.cosine_similarity(pred_vec.unsqueeze(0), 
                                    true_vec.unsqueeze(0))
    
    return similarity.item()
```

### 7. 推論パフォーマンス仕様

#### 7.1 ベンチマーク環境

| 項目 | 仕様 |
|------|------|
| CPU | Intel Core i7-9750H (6コア) |
| メモリ | 16GB DDR4 |
| GPU | NVIDIA RTX 2070 (8GB) |
| OS | Ubuntu 20.04 LTS |
| Python | 3.8.10 |
| PyTorch | 2.0.0 |

#### 7.2 パフォーマンス目標

| メトリクス | 目標値 | 測定条件 |
|-----------|--------|----------|
| 単一分子推論時間 | < 50ms | CPU環境 |
| バッチ推論速度 | > 100分子/秒 | GPU環境、バッチサイズ32 |
| モデルサイズ | < 50MB | 量子化後 |
| メモリ使用量 | < 1GB | 推論時最大 |
| 起動時間 | < 5秒 | モデルロード完了まで |

### 8. エラーコード仕様

| エラーコード | 説明 | HTTPステータス |
|-------------|------|---------------|
| E001 | 無効なMOLファイル形式 | 400 |
| E002 | 分子サイズが大きすぎる（原子数>500） | 400 |
| E003 | モデルロード失敗 | 500 |
| E004 | 推論タイムアウト | 504 |
| E005 | メモリ不足 | 507 |
| E006 | 無効なパラメータ | 400 |
| E007 | 認証エラー | 401 |
| E008 | レート制限超過 | 429 |

### 9. ロギング仕様

#### 9.1 ログレベル

| レベル | 用途 |
|--------|------|
| DEBUG | デバッグ情報、詳細な処理フロー |
| INFO | 正常な処理の記録 |
| WARNING | 警告、性能劣化の可能性 |
| ERROR | エラー、処理の失敗 |
| CRITICAL | 致命的エラー、システム停止 |

#### 9.2 ログフォーマット

```json
{
  "timestamp": "2024-01-20T10:30:45.123Z",
  "level": "INFO",
  "module": "inference",
  "message": "Prediction completed",
  "data": {
    "mol_id": "mol_001",
    "processing_time_ms": 45,
    "num_peaks": 25,
    "confidence": 0.87
  }
}
```

### 10. 設定ファイル仕様

#### 10.1 config.yaml

```yaml
model:
  encoder:
    input_dim: 128
    hidden_dim: 256
    output_dim: 256
    num_layers: 4
    dropout: 0.2
  
  decoder:
    input_dim: 256
    hidden_dim: 512
    output_dim: 2000
    num_layers: 4
    quantization_bits: 1.58
    activation_bits: 8

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  weight_decay: 0.00001
  gradient_clip: 1.0
  
inference:
  max_batch_size: 100
  timeout_seconds: 60
  num_workers: 4
  device: "cuda"  # or "cpu"

data:
  msp_path: "./data/NIST17.MSP"
  mol_dir: "./data/mol_files"
  cache_dir: "./cache"
  
logging:
  level: "INFO"
  file: "./logs/bitspec.log"
  max_size_mb: 100
  backup_count: 5
```

### 11. テスト仕様

#### 11.1 単体テスト

```python
class TestBitLinear:
    def test_quantization_range(self):
        """量子化値が-1, 0, 1のみであることを確認"""
        
    def test_forward_pass(self):
        """順伝播の動作確認"""
        
    def test_gradient_flow(self):
        """勾配が正しく伝播することを確認"""

class TestGCNEncoder:
    def test_graph_processing(self):
        """グラフデータの処理確認"""
        
    def test_batch_processing(self):
        """バッチ処理の動作確認"""
```

#### 11.2 統合テスト

```python
class TestEndToEnd:
    def test_mol_to_spectrum(self):
        """MOLファイルからスペクトル予測までの全体フロー"""
        
    def test_api_response(self):
        """APIレスポンスの形式確認"""
        
    def test_performance(self):
        """パフォーマンス要件の確認"""
```

### 12. デプロイメント仕様

#### 12.1 Dockerコンテナ

```dockerfile
FROM python:3.8-slim

WORKDIR /app

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# モデルファイルのダウンロード
RUN python scripts/download_model.py

# ポート公開
EXPOSE 8000

# アプリケーション起動
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 12.2 必要なシステム要件

| コンポーネント | 最小要件 | 推奨要件 |
|---------------|---------|----------|
| CPU | 2コア | 4コア以上 |
| メモリ | 4GB | 8GB以上 |
| ストレージ | 1GB | 5GB以上 |
| Python | 3.8+ | 3.9+ |
| CUDA | 11.0+ | 11.7+ |

### 13. セキュリティ仕様

- 入力サニタイゼーション: MOLファイルの検証
- API認証: JWTトークンによる認証
- レート制限: 1分あたり100リクエストまで
- データ暗号化: HTTPS通信の強制
- ログマスキング: 機密情報の除外