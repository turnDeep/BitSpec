# scripts/predict.py
import os
import sys

# ヘッドレス環境用の環境変数設定（RDKitのX11依存を回避）
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')  # ヘッドレス環境用のバックエンドを設定（X11不要）

import torch
import yaml
from pathlib import Path
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from src.models.gcn_model import GCNMassSpecPredictor
from src.data.features import MolecularFeaturizer
from src.utils.rtx50_compat import setup_rtx50_compatibility

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MassSpectrumPredictor:
    """マススペクトル予測器"""
    
    def __init__(self, checkpoint_path: str, config_path: str):
        """
        Args:
            checkpoint_path: モデルチェックポイントのパス
            config_path: 設定ファイルのパス
        """
        # 設定の読み込み
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # デバイス設定
        self.device = setup_rtx50_compatibility()
        logger.info(f"Using device: {self.device}")

        # チェックポイントパスの検証と自動検出
        checkpoint_path = self._find_checkpoint(checkpoint_path)

        # モデルの読み込み（configからすべてのパラメータを取得）
        self.model = GCNMassSpecPredictor(
            node_features=self.config['model']['node_features'],
            edge_features=self.config['model']['edge_features'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            spectrum_dim=self.config['data']['max_mz'],
            dropout=self.config['model']['dropout'],
            pooling=self.config['model'].get('pooling', 'mean'),  # チェックポイント互換性のためデフォルトはmean
            conv_type=self.config['model'].get('gcn', {}).get('conv_type', 'GCNConv'),
            activation=self.config['model'].get('gcn', {}).get('activation', 'relu'),
            batch_norm=self.config['model'].get('gcn', {}).get('batch_norm', True),
            residual=self.config['model'].get('gcn', {}).get('residual', True)
        ).to(self.device)

        # チェックポイントの読み込み
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Loaded model from {checkpoint_path}")
        logger.info(f"Model trained for {checkpoint['epoch']} epochs")

        # 特徴量化器
        self.featurizer = MolecularFeaturizer()

        self.max_mz = self.config['data']['max_mz']

    def _find_checkpoint(self, checkpoint_path: str) -> str:
        """
        チェックポイントファイルを探す

        Args:
            checkpoint_path: ユーザー指定のチェックポイントパス

        Returns:
            実際に存在するチェックポイントパス

        Raises:
            FileNotFoundError: チェックポイントが見つからない場合
        """
        # パスのリストを作成（優先順位順）
        candidate_paths = [
            checkpoint_path,  # ユーザー指定のパス
            "checkpoints/finetune/best_finetuned_model.pt",  # ファインチューニング後のベストモデル
            "checkpoints/pretrain/pretrained_backbone.pt",  # 事前学習済みバックボーン
            "checkpoints/best_model.pt",  # 旧形式
            "/workspace/checkpoints/finetune/best_finetuned_model.pt",  # Dockerコンテナ内
            "/workspace/checkpoints/best_model.pt",  # Dockerコンテナ内（旧形式）
        ]

        # 絶対パスに変換して検索
        for path_str in candidate_paths:
            path = Path(path_str)
            if path.exists():
                logger.info(f"Found checkpoint at: {path}")
                return str(path)

        # どこにも見つからない場合はエラー
        error_msg = (
            f"チェックポイントファイルが見つかりません。\n"
            f"以下の場所を確認しました:\n"
        )
        for path in candidate_paths:
            error_msg += f"  - {path}\n"
        error_msg += (
            f"\n解決方法:\n"
            f"1. モデルのトレーニングを実行してください:\n"
            f"   python scripts/pretrain.py --config config_pretrain.yaml\n"
            f"   python scripts/finetune.py --config config_pretrain.yaml\n"
            f"2. または、正しいチェックポイントパスを --checkpoint オプションで指定してください\n"
        )
        raise FileNotFoundError(error_msg)
    
    def predict_from_smiles(self, smiles: str) -> np.ndarray:
        """
        SMILESから質量スペクトルを予測

        Args:
            smiles: SMILES文字列

        Returns:
            予測スペクトル (max_mz,)
        """
        # SMILESから分子オブジェクトを作成
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # 3D座標の生成
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            logger.warning("Could not generate 3D coordinates")

        # 分子グラフの特徴量化
        graph_data = self.featurizer.mol_to_graph(mol)
        graph_data = graph_data.to(self.device)

        # 予測
        with torch.no_grad():
            if self.config['training']['use_amp']:
                with torch.amp.autocast('cuda'):
                    pred_spectrum = self.model(graph_data)
            else:
                pred_spectrum = self.model(graph_data)

        # NumPy配列に変換し、バッチ次元を削除
        result = pred_spectrum.cpu().numpy()
        # 2次元以上の場合は1次元にする
        if result.ndim > 1:
            result = result.squeeze()
        return result
    
    def predict_from_mol_file(self, mol_path: str) -> np.ndarray:
        """
        MOLファイルから質量スペクトルを予測
        
        Args:
            mol_path: MOLファイルのパス
            
        Returns:
            予測スペクトル
        """
        mol = Chem.MolFromMolFile(mol_path)
        if mol is None:
            raise ValueError(f"Could not read molecule from {mol_path}")
        
        smiles = Chem.MolToSmiles(mol)
        return self.predict_from_smiles(smiles)
    
    def predict_batch(self, smiles_list: list) -> np.ndarray:
        """
        複数のSMILESから一括予測
        
        Args:
            smiles_list: SMILES文字列のリスト
            
        Returns:
            予測スペクトル (batch_size, max_mz)
        """
        from torch_geometric.data import Batch
        
        # 各SMILESをグラフデータに変換
        graphs = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                continue
            
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass

            graph_data = self.featurizer.mol_to_graph(mol)
            graphs.append(graph_data)
        
        if not graphs:
            raise ValueError("No valid molecules in batch")
        
        # バッチ化
        batch = Batch.from_data_list(graphs).to(self.device)
        
        # 予測
        with torch.no_grad():
            if self.config['training']['use_amp']:
                with torch.amp.autocast('cuda'):
                    pred_spectra = self.model(batch)
            else:
                pred_spectra = self.model(batch)
        
        return pred_spectra.cpu().numpy()
    
    def visualize_prediction(self, smiles: str, true_spectrum: np.ndarray = None,
                           save_path: str = None):
        """
        予測結果の可視化

        Args:
            smiles: SMILES文字列
            true_spectrum: 真のスペクトル（オプション）
            save_path: 保存先パス（オプション）
        """
        # 予測
        pred_spectrum = self.predict_from_smiles(smiles)

        # 分子構造の描画
        mol = Chem.MolFromSmiles(smiles)

        fig = plt.figure(figsize=(15, 10))

        # 分子構造（ヘッドレス環境対応）
        ax1 = plt.subplot(2, 1, 1)
        img = None

        # 複数の描画方法を試す（ヘッドレス環境対応）
        try:
            # 方法1: rdMolDraw2D (PNG) - X11不要
            from rdkit.Chem.Draw import rdMolDraw2D
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            img_data = drawer.GetDrawingText()
            img = Image.open(BytesIO(img_data))
            logger.debug("Drew molecule using MolDraw2DCairo")
        except Exception as e1:
            logger.debug(f"MolDraw2DCairo failed: {e1}")
            try:
                # 方法2: SVGベースの描画
                from rdkit.Chem.Draw import rdMolDraw2D
                drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                svg_data = drawer.GetDrawingText()
                # SVGをPNGに変換（cairosvgが必要だが、なければスキップ）
                try:
                    import cairosvg
                    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
                    img = Image.open(BytesIO(png_data))
                    logger.debug("Drew molecule using SVG")
                except ImportError:
                    logger.debug("cairosvg not available, skipping SVG rendering")
            except Exception as e2:
                logger.debug(f"SVG drawing failed: {e2}")

        # フォールバック: 分子構造なしでSMILES文字列のみ表示
        if img is None:
            logger.warning("Could not draw molecule structure, using text only")
            img = Image.new('RGB', (400, 400), color='white')

        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f'Molecule: {smiles}', fontsize=12, pad=10)
        
        # スペクトル
        ax2 = plt.subplot(2, 1, 2)
        
        mz_values = np.arange(self.max_mz)
        
        # 予測スペクトル
        ax2.stem(mz_values, pred_spectrum, linefmt='b-', markerfmt='bo', 
                basefmt=' ', label='Predicted')
        
        # 真のスペクトル（あれば）
        if true_spectrum is not None:
            ax2.stem(mz_values, true_spectrum, linefmt='r-', markerfmt='ro', 
                    basefmt=' ', label='True', alpha=0.5)
            
            # 類似度の計算
            cosine_sim = np.dot(pred_spectrum, true_spectrum) / \
                        (np.linalg.norm(pred_spectrum) * np.linalg.norm(true_spectrum))
            ax2.text(0.02, 0.98, f'Cosine Similarity: {cosine_sim:.4f}',
                    transform=ax2.transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel('m/z', fontsize=12)
        ax2.set_ylabel('Relative Intensity', fontsize=12)
        ax2.set_title('Mass Spectrum', fontsize=14, pad=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, self.max_mz)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def find_significant_peaks(self, spectrum: np.ndarray,
                             threshold: float = 0.05,
                             top_n: int = 20) -> list:
        """
        有意なピークを検出

        Args:
            spectrum: スペクトルデータ
            threshold: 強度の閾値
            top_n: 上位N個のピーク

        Returns:
            [(mz, intensity), ...] のリスト
        """
        # 配列を1次元にフラット化
        spectrum = np.atleast_1d(spectrum).flatten()

        # 閾値以上のピークを検出
        peak_indices = np.where(spectrum > threshold)[0]
        peak_intensities = spectrum[peak_indices]

        # ピークが見つからない場合
        if len(peak_indices) == 0:
            logger.warning("No peaks found above threshold")
            return []

        # 強度でソート
        sorted_idx = np.argsort(peak_intensities)[::-1]

        # 上位N個を取得（実際のピーク数とtop_nの小さい方）
        n_peaks = min(len(peak_indices), top_n)
        top_peaks = [(int(peak_indices[sorted_idx[i]]), float(peak_intensities[sorted_idx[i]]))
                     for i in range(n_peaks)]

        return top_peaks
    
    def export_to_msp(self, smiles: str, output_path: str, 
                     compound_name: str = None):
        """
        予測結果をMSP形式でエクスポート
        
        Args:
            smiles: SMILES文字列
            output_path: 出力ファイルパス
            compound_name: 化合物名
        """
        pred_spectrum = self.predict_from_smiles(smiles)
        peaks = self.find_significant_peaks(pred_spectrum)
        
        mol = Chem.MolFromSmiles(smiles)
        if compound_name is None:
            compound_name = f"Unknown_{Chem.MolToInchiKey(mol)}"
        
        with open(output_path, 'w') as f:
            f.write(f"NAME: {compound_name}\n")
            f.write(f"SMILES: {smiles}\n")
            f.write(f"INCHIKEY: {Chem.MolToInchiKey(mol)}\n")
            f.write(f"NUM PEAKS: {len(peaks)}\n")
            
            for mz, intensity in peaks:
                f.write(f"{mz} {intensity:.6f}\n")
        
        logger.info(f"Exported spectrum to {output_path}")

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict Mass Spectrum from Molecular Structure')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--smiles', type=str,
                       help='SMILES string')
    parser.add_argument('--mol_file', type=str,
                       help='Path to MOL file')
    parser.add_argument('--batch_file', type=str,
                       help='Path to file containing SMILES list (one per line)')
    parser.add_argument('--output', type=str, default='prediction.png',
                       help='Output path for visualization')
    parser.add_argument('--export_msp', action='store_true',
                       help='Export prediction to MSP format')
    
    args = parser.parse_args()
    
    # 予測器の作成
    predictor = MassSpectrumPredictor(args.checkpoint, args.config)
    
    if args.smiles:
        # 単一SMILES予測
        logger.info(f"Predicting spectrum for: {args.smiles}")
        spectrum = predictor.predict_from_smiles(args.smiles)
        
        # ピークの表示
        peaks = predictor.find_significant_peaks(spectrum)
        logger.info(f"Top peaks: {peaks[:10]}")
        
        # 可視化
        predictor.visualize_prediction(args.smiles, save_path=args.output)
        
        # MSPエクスポート
        if args.export_msp:
            msp_path = Path(args.output).with_suffix('.msp')
            predictor.export_to_msp(args.smiles, str(msp_path))
    
    elif args.mol_file:
        # MOLファイル予測
        logger.info(f"Predicting spectrum from MOL file: {args.mol_file}")
        spectrum = predictor.predict_from_mol_file(args.mol_file)
        peaks = predictor.find_significant_peaks(spectrum)
        logger.info(f"Top peaks: {peaks[:10]}")
    
    elif args.batch_file:
        # バッチ予測
        logger.info(f"Batch prediction from: {args.batch_file}")
        with open(args.batch_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        
        spectra = predictor.predict_batch(smiles_list)
        logger.info(f"Predicted {len(spectra)} spectra")
        
        # 結果の保存
        output_dir = Path(args.output).parent / "batch_results"
        output_dir.mkdir(exist_ok=True)
        
        for i, (smiles, spectrum) in enumerate(zip(smiles_list, spectra)):
            save_path = output_dir / f"prediction_{i+1}.png"
            predictor.visualize_prediction(smiles, save_path=str(save_path))
            
            if args.export_msp:
                msp_path = output_dir / f"prediction_{i+1}.msp"
                predictor.export_to_msp(smiles, str(msp_path))
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
