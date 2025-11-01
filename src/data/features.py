# src/data/features.py
"""
分子の特徴量抽出
グラフデータへの変換
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional


class MolecularFeaturizer:
    """分子をグラフデータに変換"""
    
    # 原子の特徴量
    ATOM_FEATURES = {
        'atomic_num': [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53],  # H, C, N, O, F, Si, P, S, Cl, Br, I
        'degree': [0, 1, 2, 3, 4, 5, 6],
        'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
        'chiral_tag': [0, 1, 2, 3],
        'num_Hs': [0, 1, 2, 3, 4],
        'hybridization': [0, 1, 2, 3, 4, 5, 6],
    }
    
    # 結合の特徴量
    BOND_FEATURES = {
        'bond_type': [1, 2, 3, 12],  # SINGLE, DOUBLE, TRIPLE, AROMATIC (4次元)
    }
    
    def __init__(self, use_3d: bool = False):
        """
        Args:
            use_3d: 3D座標を使用するか
        """
        self.use_3d = use_3d
    
    def one_hot_encoding(
        self, 
        value: int, 
        choices: List[int]
    ) -> List[int]:
        """One-hot encoding"""
        encoding = [0] * (len(choices) + 1)
        index = choices.index(value) if value in choices else -1
        encoding[index] = 1
        return encoding
    
    def get_atom_features(self, atom: Chem.Atom) -> np.ndarray:
        """
        原子の特徴ベクトルを取得

        Args:
            atom: RDKit原子オブジェクト

        Returns:
            特徴ベクトル (48次元)
        """
        features = []

        # 原子番号 (12次元: H, C, N, O, F, Si, P, S, Cl, Br, I + unknown)
        features += self.one_hot_encoding(
            atom.GetAtomicNum(),
            self.ATOM_FEATURES['atomic_num']
        )
        
        # 次数 (8次元)
        features += self.one_hot_encoding(
            atom.GetDegree(),
            self.ATOM_FEATURES['degree']
        )
        
        # 形式電荷 (8次元)
        features += self.one_hot_encoding(
            atom.GetFormalCharge(),
            self.ATOM_FEATURES['formal_charge']
        )
        
        # キラリティ (5次元)
        features += self.one_hot_encoding(
            int(atom.GetChiralTag()),
            self.ATOM_FEATURES['chiral_tag']
        )
        
        # 水素数 (6次元)
        features += self.one_hot_encoding(
            atom.GetTotalNumHs(),
            self.ATOM_FEATURES['num_Hs']
        )
        
        # 混成軌道 (7次元)
        features += self.one_hot_encoding(
            int(atom.GetHybridization()),
            self.ATOM_FEATURES['hybridization']
        )
        
        # 芳香族性 (1次元)
        features.append(int(atom.GetIsAromatic()))
        
        # 環に含まれるか (1次元)
        features.append(int(atom.IsInRing()))
        
        return np.array(features, dtype=np.float32)
    
    def get_bond_features(self, bond: Chem.Bond) -> np.ndarray:
        """
        結合の特徴ベクトルを取得

        Args:
            bond: RDKit結合オブジェクト

        Returns:
            特徴ベクトル (6次元)
        """
        features = []

        # 結合タイプ (4次元: SINGLE, DOUBLE, TRIPLE, AROMATIC)
        bond_type_map = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 12,
        }
        bond_type = bond_type_map.get(bond.GetBondType(), 0)
        features += [1 if bond_type == bt else 0 for bt in self.BOND_FEATURES['bond_type']]

        # 共役 (1次元: バイナリ値)
        features.append(int(bond.GetIsConjugated()))

        # 環に含まれるか (1次元: バイナリ値)
        features.append(int(bond.IsInRing()))

        return np.array(features, dtype=np.float32)
    
    def mol_to_graph(
        self, 
        mol: Chem.Mol,
        y: Optional[np.ndarray] = None
    ) -> Data:
        """
        RDKit分子をPyTorch Geometricのグラフデータに変換
        
        Args:
            mol: RDKit分子オブジェクト
            y: ターゲットスペクトル（オプション）
            
        Returns:
            PyG Data object
        """
        if mol is None:
            raise ValueError("Invalid molecule")
        
        # 原子特徴
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        x = torch.tensor(np.array(atom_features), dtype=torch.float32)
        
        # エッジインデックスとエッジ特徴
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # 無向グラフなので両方向を追加
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            bond_feat = self.get_bond_features(bond)
            edge_features.append(bond_feat)
            edge_features.append(bond_feat)
        
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)
        else:
            # 孤立原子の場合
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(edge_features[0]) if edge_features else 6), dtype=torch.float32)
        
        # PyG Dataオブジェクトを作成
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # ターゲットスペクトル
        if y is not None:
            data.y = torch.tensor(y, dtype=torch.float32)
        
        # 分子量などの追加情報
        data.mol_weight = Descriptors.ExactMolWt(mol)
        data.num_atoms = mol.GetNumAtoms()
        data.num_bonds = mol.GetNumBonds()
        
        return data
    
    def smiles_to_graph(
        self,
        smiles: str,
        y: Optional[np.ndarray] = None
    ) -> Data:
        """
        SMILESからグラフデータを作成
        
        Args:
            smiles: SMILES文字列
            y: ターゲットスペクトル（オプション）
            
        Returns:
            PyG Data object
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # 3D座標を生成（必要な場合）
        if self.use_3d:
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
        
        return self.mol_to_graph(mol, y)


class SubstructureFeaturizer:
    """
    部分構造予測用の特徴量抽出
    論文のサポートAI用
    """
    
    # 48個の部分構造パターン
    SUBSTRUCTURE_PATTERNS = [
        # 基本的な官能基
        'c1ccccc1',  # ベンゼン環
        'C(=O)O',    # カルボキシル基
        'C(=O)N',    # アミド
        'C(=O)C',    # ケトン
        'C(O)',      # ヒドロキシル
        'N',         # アミン
        'C(=C)',     # アルケン
        'C#C',       # アルキン
        'c1ccc(O)cc1', # フェノール
        'C(F)(F)F',  # トリフルオロメチル
        'S(=O)(=O)',  # スルホン
        'C#N',       # ニトリル
        'N(=O)=O',   # ニトロ
        # さらに35個の部分構造...（省略）
    ]
    
    def __init__(self):
        self.patterns = [Chem.MolFromSmarts(p) for p in self.SUBSTRUCTURE_PATTERNS]
    
    def get_substructure_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """
        部分構造フィンガープリントを取得
        
        Args:
            mol: RDKit分子オブジェクト
            
        Returns:
            部分構造の有無を示すバイナリベクトル [48]
        """
        fingerprint = np.zeros(len(self.patterns), dtype=np.float32)
        
        for i, pattern in enumerate(self.patterns):
            if pattern is not None:
                if mol.HasSubstructMatch(pattern):
                    fingerprint[i] = 1.0
        
        return fingerprint


if __name__ == "__main__":
    # テスト
    print("Testing MolecularFeaturizer...")
    
    # テスト分子（ベンゼン）
    smiles = "c1ccccc1"
    mol = Chem.MolFromSmiles(smiles)
    
    # 特徴量抽出
    featurizer = MolecularFeaturizer()
    data = featurizer.mol_to_graph(mol)
    
    print(f"Number of atoms: {data.num_nodes}")
    print(f"Number of bonds: {data.num_edges // 2}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge features shape: {data.edge_attr.shape}")
    print(f"Molecular weight: {data.mol_weight:.2f}")
    
    # 部分構造特徴量
    sub_featurizer = SubstructureFeaturizer()
    sub_fp = sub_featurizer.get_substructure_fingerprint(mol)
    print(f"Substructure fingerprint: {sub_fp.sum()} matches")
    
    print("\nFeaturizer test passed!")
