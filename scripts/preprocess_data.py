# scripts/preprocess_data.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_nist_msp(input_path: str, output_dir: str, 
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   min_peaks: int = 5,
                   max_mz: int = 1000):
    """
    NIST MSPファイルを訓練/検証/テストセットに分割
    
    Args:
        input_path: 入力MSPファイルのパス
        output_dir: 出力ディレクトリ
        train_ratio: 訓練データの割合
        val_ratio: 検証データの割合
        test_ratio: テストデータの割合
        min_peaks: 最小ピーク数
        max_mz: 最大m/z値
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Parsing NIST MSP file: {input_path}")
    
    # データの読み込み
    samples = []
    current_sample = {}
    peaks = []
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Reading MSP"):
            line = line.strip()
            
            if line.startswith('NAME:'):
                if current_sample and peaks and len(peaks) >= min_peaks:
                    current_sample['peaks'] = peaks
                    samples.append(current_sample)
                current_sample = {'name': line.split(':', 1)[1].strip()}
                peaks = []
            
            elif line.startswith('SMILES:'):
                current_sample['smiles'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('INCHIKEY:'):
                current_sample['inchikey'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('MW:'):
                try:
                    current_sample['mw'] = float(line.split(':')[1].strip())
                except:
                    pass
            
            elif line.startswith('FORMULA:'):
                current_sample['formula'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('NUM PEAKS:'):
                try:
                    current_sample['num_peaks'] = int(line.split(':')[1].strip())
                except:
                    pass
            
            elif line and not line.startswith('#') and ':' not in line:
                # ピークデータ
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        if 0 <= mz < max_mz:
                            peaks.append((mz, intensity))
                except:
                    continue
        
        # 最後のサンプルを追加
        if current_sample and peaks and len(peaks) >= min_peaks:
            current_sample['peaks'] = peaks
            samples.append(current_sample)
    
    logger.info(f"Total samples: {len(samples)}")
    
    # データのフィルタリング
    valid_samples = []
    for sample in samples:
        if 'smiles' in sample and sample['smiles']:
            valid_samples.append(sample)
    
    logger.info(f"Valid samples with SMILES: {len(valid_samples)}")
    
    # データの分割
    indices = np.arange(len(valid_samples))
    
    # 訓練とテンポラリに分割
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=(val_ratio + test_ratio),
        random_state=42
    )
    
    # テンポラリを検証とテストに分割
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=42
    )
    
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # データセットの保存
    def save_dataset(samples_subset, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples_subset:
                f.write(f"NAME: {sample['name']}\n")
                f.write(f"SMILES: {sample['smiles']}\n")
                if 'inchikey' in sample:
                    f.write(f"INCHIKEY: {sample['inchikey']}\n")
                if 'mw' in sample:
                    f.write(f"MW: {sample['mw']}\n")
                if 'formula' in sample:
                    f.write(f"FORMULA: {sample['formula']}\n")
                f.write(f"NUM PEAKS: {len(sample['peaks'])}\n")
                for mz, intensity in sample['peaks']:
                    f.write(f"{mz:.4f} {intensity:.4f}\n")
                f.write("\n")
    
    # 各セットを保存
    train_samples = [valid_samples[i] for i in train_idx]
    val_samples = [valid_samples[i] for i in val_idx]
    test_samples = [valid_samples[i] for i in test_idx]
    
    save_dataset(train_samples, output_dir / 'train.msp')
    save_dataset(val_samples, output_dir / 'val.msp')
    save_dataset(test_samples, output_dir / 'test.msp')
    
    logger.info(f"Saved datasets to {output_dir}")
    
    # 統計情報の保存
    stats = {
        'total_samples': len(valid_samples),
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_idx),
        'avg_peaks': np.mean([len(s['peaks']) for s in valid_samples]),
        'max_peaks': max([len(s['peaks']) for s in valid_samples]),
        'min_peaks': min([len(s['peaks']) for s in valid_samples])
    }
    
    import json
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Dataset statistics: {stats}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess NIST MSP data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input NIST MSP file')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test data ratio')
    parser.add_argument('--min_peaks', type=int, default=5,
                       help='Minimum number of peaks')
    parser.add_argument('--max_mz', type=int, default=1000,
                       help='Maximum m/z value')
    
    args = parser.parse_args()
    
    parse_nist_msp(
        args.input,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.min_peaks,
        args.max_mz
    )

if __name__ == '__main__':
    main()
