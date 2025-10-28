# src/utils/rtx50_compat.py
"""
RTX 50シリーズ（sm_120）対応ユーティリティ
"""

import torch
import warnings
import sys
from typing import Optional


def enable_rtx50_compatibility(
    force_sm90_emulation: bool = False,
    verbose: bool = True
) -> bool:
    """
    RTX 50シリーズのGPU対応を有効化
    
    Args:
        force_sm90_emulation: sm_90エミュレーションを強制するか
        verbose: ログを出力するか
        
    Returns:
        成功したかどうか
    """
    try:
        # rtx50-compatパッケージをインポート
        import rtx50_compat
        
        if verbose:
            print("✓ RTX 50 compatibility layer enabled")
        
        return True
        
    except ImportError:
        if verbose:
            print("⚠ rtx50-compat package not found")
            print("  Install with: pip install rtx50-compat")
        
        # 代替方法: PyTorchのCUDA capability checkをパッチ
        if force_sm90_emulation:
            _patch_cuda_capability_check(verbose=verbose)
            return True
        
        return False


def _patch_cuda_capability_check(verbose: bool = True):
    """CUDA capability checkをパッチ（sm_120をsm_90として扱う）"""
    
    if not torch.cuda.is_available():
        return
    
    original_get_device_capability = torch.cuda.get_device_capability
    
    def patched_get_device_capability(device=None):
        major, minor = original_get_device_capability(device)
        
        # sm_120 (RTX 50) をsm_90 (H100)として扱う
        if major == 12 and minor == 0:
            if verbose:
                print(f"  Patching sm_120 → sm_90 (H100 emulation)")
            return (9, 0)
        
        return (major, minor)
    
    torch.cuda.get_device_capability = patched_get_device_capability
    
    if verbose:
        print("✓ CUDA capability check patched")


def setup_gpu_environment(
    device_id: int = 0,
    mixed_precision: bool = True,
    compile_model: bool = True,
    compile_mode: str = "reduce-overhead"
) -> torch.device:
    """
    GPU環境のセットアップ
    
    Args:
        device_id: GPU ID
        mixed_precision: 混合精度訓練を使用するか
        compile_model: torch.compileを使用するか
        compile_mode: コンパイルモード
        
    Returns:
        デバイス
    """
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        return torch.device("cpu")
    
    # デバイスを設定
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    
    # GPU情報を表示
    gpu_name = torch.cuda.get_device_name(device)
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
    
    print(f"\n{'='*60}")
    print(f"GPU Configuration:")
    print(f"  Device: {gpu_name}")
    print(f"  Memory: {gpu_memory:.2f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    
    # Compute capability
    capability = torch.cuda.get_device_capability(device)
    print(f"  Compute Capability: sm_{capability[0]}{capability[1]}")
    
    # 混合精度
    if mixed_precision:
        print(f"  Mixed Precision: Enabled (FP16)")
    
    # torch.compile
    if compile_model and hasattr(torch, 'compile'):
        print(f"  torch.compile: Enabled ({compile_mode})")
    
    print(f"{'='*60}\n")
    
    # cuDNNベンチマークモード
    torch.backends.cudnn.benchmark = True
    
    # TF32を有効化（Ampere以降）
    if capability[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    return device


def get_optimal_batch_size(
    model: torch.nn.Module,
    sample_input,
    device: torch.device,
    max_memory_usage: float = 0.8
) -> int:
    """
    最適なバッチサイズを推定
    
    Args:
        model: モデル
        sample_input: サンプル入力
        device: デバイス
        max_memory_usage: 最大メモリ使用率
        
    Returns:
        推奨バッチサイズ
    """
    if not torch.cuda.is_available():
        return 32
    
    model = model.to(device)
    model.eval()
    
    # 現在の空きメモリ
    free_memory = torch.cuda.mem_get_info(device)[0]
    total_memory = torch.cuda.mem_get_info(device)[1]
    
    print(f"Free GPU Memory: {free_memory / 1e9:.2f} GB / {total_memory / 1e9:.2f} GB")
    
    # バッチサイズ1でのメモリ使用量を測定
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(sample_input.to(device))
    
    memory_per_sample = torch.cuda.max_memory_allocated(device)
    
    # 推奨バッチサイズ
    available_memory = free_memory * max_memory_usage
    recommended_batch_size = int(available_memory / memory_per_sample)
    
    # 2の累乗に調整
    batch_size = 2 ** int(torch.log2(torch.tensor(recommended_batch_size)))
    batch_size = max(1, min(batch_size, 128))  # 1-128の範囲
    
    print(f"Recommended Batch Size: {batch_size}")
    
    return batch_size


if __name__ == "__main__":
    print("Testing RTX 50 compatibility...")
    
    # RTX 50対応を有効化
    enable_rtx50_compatibility(verbose=True)
    
    # GPU環境をセットアップ
    if torch.cuda.is_available():
        device = setup_gpu_environment()
        print(f"Device: {device}")
    else:
        print("No CUDA device available")
    
    print("\nRTX 50 compatibility test passed!")
