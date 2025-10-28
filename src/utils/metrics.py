# src/utils/metrics.py
import numpy as np
from scipy import stats
from typing import Dict

def calculate_metrics(pred_spectra: np.ndarray, 
                     true_spectra: np.ndarray) -> Dict[str, float]:
    """
    スペクトル予測のメトリクスを計算
    
    Args:
        pred_spectra: 予測スペクトル (batch_size, max_mz)
        true_spectra: 真のスペクトル (batch_size, max_mz)
        
    Returns:
        メトリクスの辞書
    """
    batch_size = pred_spectra.shape[0]
    
    # コサイン類似度
    cosine_sims = []
    for i in range(batch_size):
        pred = pred_spectra[i]
        true = true_spectra[i]
        
        norm_pred = np.linalg.norm(pred)
        norm_true = np.linalg.norm(true)
        
        if norm_pred > 0 and norm_true > 0:
            cosine_sim = np.dot(pred, true) / (norm_pred * norm_true)
            cosine_sims.append(cosine_sim)
    
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
    
    return {
        'cosine_similarity': np.mean(cosine_sims),
        'pearson_correlation': np.mean(pearson_corrs),
        'mse': mse,
        'mae': mae,
        'top_k_accuracy': np.mean(top_k_accuracies)
    }
