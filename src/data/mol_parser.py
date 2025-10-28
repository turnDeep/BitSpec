# src/data/mol_parser.py
"""
MOLファイルとNIST MSPファイルのパーサー
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem


class MOLParser:
    """MOLファイルパーサー"""
    
    def __init__(self):
        self.mol = None
        
    def parse_file(self, mol_file: str) -> Chem.Mol:
        """
        MOLファイルを読み込んでRDKit分子オブジェクトを返す
        
        Args:
            mol_file: MOLファイルのパス
            
        Returns:
            RDKit分子オブジェクト
        """
        self.mol = Chem.MolFromMolFile(mol_file, removeHs=False)
        
        if self.mol is None:
            raise ValueError(f"Failed to parse MOL file: {mol_file}")
        
        # 座標が無い場合は3D座標を生成
        if self.mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(self.mol)
            AllChem.MMFFOptimizeMolecule(self.mol)
        
        return self.mol
    
    def get_molecular_formula(self) -> str:
        """分子式を取得"""
        if self.mol is None:
            raise ValueError("No molecule loaded")
        return Chem.rdMolDescriptors.CalcMolFormula(self.mol)
    
    def get_molecular_weight(self) -> float:
        """分子量を取得"""
        if self.mol is None:
            raise ValueError("No molecule loaded")
        return Chem.rdMolDescriptors.CalcExactMolWt(self.mol)


class NISTMSPParser:
    """NIST MSPファイルパーサー"""
    
    def __init__(self):
        self.current_compound = {}
        self.compounds = []
        
    def parse_file(self, msp_file: str) -> List[Dict]:
        """
        MSPファイルを解析して化合物データのリストを返す
        
        Args:
            msp_file: MSPファイルのパス
            
        Returns:
            化合物データのリスト
        """
        self.compounds = []
        self.current_compound = {}
        
        with open(msp_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('Name:'):
                # 新しい化合物の開始
                if self.current_compound:
                    self.compounds.append(self.current_compound)
                self.current_compound = {}
                self.current_compound['Name'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('InChIKey:'):
                self.current_compound['InChIKey'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('Formula:'):
                self.current_compound['Formula'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('MW:'):
                self.current_compound['MW'] = float(line.split(':', 1)[1].strip())
            
            elif line.startswith('ExactMass:'):
                self.current_compound['ExactMass'] = float(line.split(':', 1)[1].strip())
            
            elif line.startswith('CASNO:'):
                self.current_compound['CASNO'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('ID:'):
                self.current_compound['ID'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('RI:'):
                self.current_compound['RI'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('Num peaks:'):
                num_peaks = int(line.split(':', 1)[1].strip())
                self.current_compound['NumPeaks'] = num_peaks
                
                # マススペクトルデータを読み込み
                spectrum = []
                i += 1
                while i < len(lines) and len(spectrum) < num_peaks:
                    peak_line = lines[i].strip()
                    if peak_line and not peak_line.startswith('$$$$'):
                        # "mz intensity" 形式のデータを解析
                        parts = peak_line.split()
                        for j in range(0, len(parts), 2):
                            if j + 1 < len(parts):
                                try:
                                    mz = float(parts[j])
                                    intensity = float(parts[j + 1])
                                    spectrum.append((mz, intensity))
                                except ValueError:
                                    pass
                    i += 1
                
                self.current_compound['Spectrum'] = spectrum
                i -= 1  # 次のループで正しい位置に
            
            elif line.startswith('$$$$'):
                # 化合物データの終了
                if self.current_compound:
                    self.compounds.append(self.current_compound)
                self.current_compound = {}
            
            i += 1
        
        # 最後の化合物を追加
        if self.current_compound:
            self.compounds.append(self.current_compound)
        
        return self.compounds
    
    def normalize_spectrum(
        self, 
        spectrum: List[Tuple[float, float]],
        max_mz: int = 1000,
        mz_bin_size: float = 1.0
    ) -> np.ndarray:
        """
        マススペクトルを正規化してビン化
        
        Args:
            spectrum: [(m/z, intensity), ...] のリスト
            max_mz: 最大m/z値
            mz_bin_size: ビンのサイズ
            
        Returns:
            正規化されたスペクトル配列 [max_mz / mz_bin_size]
        """
        num_bins = int(max_mz / mz_bin_size)
        spectrum_array = np.zeros(num_bins, dtype=np.float32)
        
        for mz, intensity in spectrum:
            if mz < max_mz:
                bin_idx = int(mz / mz_bin_size)
                if bin_idx < num_bins:
                    spectrum_array[bin_idx] = max(spectrum_array[bin_idx], intensity)
        
        # 正規化（最大強度を1に）
        max_intensity = spectrum_array.max()
        if max_intensity > 0:
            spectrum_array = spectrum_array / max_intensity
        
        return spectrum_array
    
    def get_compound_by_id(self, compound_id: str) -> Optional[Dict]:
        """IDで化合物を検索"""
        for compound in self.compounds:
            if compound.get('ID') == compound_id:
                return compound
        return None


if __name__ == "__main__":
    # テスト
    print("Testing parsers...")
    
    # MSPパーサーのテスト
    msp_parser = NISTMSPParser()
    
    # テストデータ
    test_msp = """Name: Test Compound
InChIKey: TESTKEY
Formula: C10H10O
MW: 150
ExactMass: 150.068
CASNO: 12345-67-8
ID: 200001
RI:1500
Comment: Test compound
Num peaks: 5
41 100
55 50
69 25
82 10
150 999

$$$$
"""
    
    # 一時ファイルに書き込んでテスト
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.msp') as f:
        f.write(test_msp)
        temp_file = f.name
    
    compounds = msp_parser.parse_file(temp_file)
    print(f"Parsed {len(compounds)} compounds")
    
    if compounds:
        compound = compounds[0]
        print(f"Name: {compound['Name']}")
        print(f"Formula: {compound['Formula']}")
        print(f"Num peaks: {compound['NumPeaks']}")
        
        # スペクトル正規化
        spectrum_array = msp_parser.normalize_spectrum(compound['Spectrum'])
        print(f"Normalized spectrum shape: {spectrum_array.shape}")
        print(f"Non-zero peaks: {np.count_nonzero(spectrum_array)}")
    
    # クリーンアップ
    import os
    os.remove(temp_file)
    
    print("Parser test passed!")
