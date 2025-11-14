# src/utils/metrics.py
import numpy as np
from scipy import stats
from typing import Dict, Tuple

def weighted_cosine_similarity(pred: np.ndarray,
                               true: np.ndarray,
                               intensity_weight: float = 0.5,
                               mz_weight: float = 3.0) -> float:
    """
    Weighted Cosine Similarity (WCS) for mass spectra

    マススペクトル用の重み付きコサイン類似度。
    NIST標準やNEIMS論文で使用される評価指標。

    Args:
        pred: 予測スペクトル (max_mz,)
        true: 真のスペクトル (max_mz,)
        intensity_weight: 強度の重み（累乗）デフォルト: 0.5 (Stein & Scott)
        mz_weight: m/z値の重み（累乗）デフォルト: 3.0 (NIST標準)

    Returns:
        Weighted Cosine Similarity (0-1)

    参考文献:
        - NIST標準: (0.6, 3.0)
        - Stein & Scott: (0.5, 3.0) または (0.5, 2.0)
        - Kim et al.: (0.53, 1.3)
    """
    # m/z値の配列を生成（インデックスがm/z値に対応）
    mz_values = np.arange(len(pred))

    # ゼロ除算を避けるため、m/z=0を1に置き換え
    mz_values = np.where(mz_values == 0, 1, mz_values)

    # 重み付き変換: I^a * m^b
    weighted_pred = (pred ** intensity_weight) * (mz_values ** mz_weight)
    weighted_true = (true ** intensity_weight) * (mz_values ** mz_weight)

    # 重み付きコサイン類似度の計算
    norm_pred = np.linalg.norm(weighted_pred)
    norm_true = np.linalg.norm(weighted_true)

    if norm_pred > 0 and norm_true > 0:
        wcs = np.dot(weighted_pred, weighted_true) / (norm_pred * norm_true)
        return wcs
    else:
        return 0.0


def calculate_metrics(pred_spectra: np.ndarray,
                     true_spectra: np.ndarray,
                     include_weighted: bool = True) -> Dict[str, float]:
    """
    スペクトル予測のメトリクスを計算

    Args:
        pred_spectra: 予測スペクトル (batch_size, max_mz)
        true_spectra: 真のスペクトル (batch_size, max_mz)
        include_weighted: Weighted Cosine Similarityを計算するか

    Returns:
        メトリクスの辞書
    """
    batch_size = pred_spectra.shape[0]

    # コサイン類似度（非重み付き）
    cosine_sims = []
    for i in range(batch_size):
        pred = pred_spectra[i]
        true = true_spectra[i]

        norm_pred = np.linalg.norm(pred)
        norm_true = np.linalg.norm(true)

        if norm_pred > 0 and norm_true > 0:
            cosine_sim = np.dot(pred, true) / (norm_pred * norm_true)
            cosine_sims.append(cosine_sim)

    # Weighted Cosine Similarity（複数の重み付け）
    wcs_nist = []      # NIST標準: (0.6, 3.0)
    wcs_stein = []     # Stein & Scott: (0.5, 3.0)
    wcs_kim = []       # Kim et al.: (0.53, 1.3)

    if include_weighted:
        for i in range(batch_size):
            pred = pred_spectra[i]
            true = true_spectra[i]

            # NIST標準
            wcs_nist.append(weighted_cosine_similarity(pred, true, 0.6, 3.0))

            # Stein & Scott
            wcs_stein.append(weighted_cosine_similarity(pred, true, 0.5, 3.0))

            # Kim et al.
            wcs_kim.append(weighted_cosine_similarity(pred, true, 0.53, 1.3))

    # ピアソン相関係数
    pearson_corrs = []
    for i in range(batch_size):
        if np.std(pred_spectra[i]) > 0 and np.std(true_spectra[i]) > 0:
            corr, _ = stats.pearsonr(pred_spectra[i], true_spectra[i])
            if not np.isnan(corr):
                pearson_corrs.append(corr)

    # MSE
    mse = np.mean((pred_spectra - true_spectra) ** 2)

    # MAE
    mae = np.mean(np.abs(pred_spectra - true_spectra))

    # Top-K精度（強度上位Kピークの一致率）
    k = 20
    top_k_accuracies = []
    for i in range(batch_size):
        pred_top_k = set(np.argsort(pred_spectra[i])[-k:])
        true_top_k = set(np.argsort(true_spectra[i])[-k:])
        accuracy = len(pred_top_k & true_top_k) / k
        top_k_accuracies.append(accuracy)

    # メトリクス辞書の作成
    metrics = {
        'cosine_similarity': np.mean(cosine_sims) if cosine_sims else 0.0,
        'pearson_correlation': np.mean(pearson_corrs) if pearson_corrs else 0.0,
        'mse': mse,
        'mae': mae,
        'top_k_accuracy': np.mean(top_k_accuracies) if top_k_accuracies else 0.0
    }

    # Weighted Cosine Similarityを追加
    if include_weighted:
        metrics['wcs_nist'] = np.mean(wcs_nist) if wcs_nist else 0.0
        metrics['wcs_stein'] = np.mean(wcs_stein) if wcs_stein else 0.0
        metrics['wcs_kim'] = np.mean(wcs_kim) if wcs_kim else 0.0

    return metrics
