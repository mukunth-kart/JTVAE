import networkx as nx
from rdkit import Chem
import torch

class Vocab:
    def __init__(self, smiles_list, embedding_size=64):
        self.smiles = smiles_list
        self.mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        self.embedding_size = embedding_size
        self.size = len(smiles_list)
        self.idx_map = {smi: idx for idx, smi in enumerate(smiles_list)}
        self.embeddings = torch.randn(self.size, embedding_size)  # random init or pretrained

    def get_index(self, smiles):
        """Return index of a given fragment SMILES."""
        return self.idx_map.get(smiles, -1)

    def get_smiles(self, idx):
        """Return fragment SMILES given an index."""
        if 0 <= idx < self.size:
            return self.smiles[idx]
        return None

    def get_embedding(self, smiles_or_idx):
        """Return embedding for a fragment."""
        if isinstance(smiles_or_idx, str):
            idx = self.get_index(smiles_or_idx)
        else:
            idx = smiles_or_idx
        return self.embeddings[idx] if 0 <= idx < self.size else None
    
    def get_size(self):
        return self.size
    
    def __repr__(self):
        return f"Vocab(size={self.size}, embedding_size={self.embedding_size})"


class MolTreeNode:
    def __init__(self, smiles, idx, atoms=None, label=None, frag_type=None):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.idx = idx
        self.atoms = atoms or []
        self.label = label
        self.frag_type = frag_type
        self.is_ring = self.mol.GetRingInfo().NumRings() > 0 if self.mol else False
        self.neighbors = []
        self.feature = None  # placeholder for graph embedding or one-hot features
    
    def add_neighbor(self, node_idx):
        if node_idx not in self.neighbors:
            self.neighbors.append(node_idx)
    
    def to_dict(self):
        return {
            "idx": self.idx,
            "smiles": self.smiles,
            "atoms": self.atoms,
            "neighbors": self.neighbors,
            "label": self.label,
            "frag_type": self.frag_type,
            "is_ring": self.is_ring,
        }

    def __repr__(self):
        return f"MolTreeNode(idx={self.idx}, smiles='{self.smiles}', type={self.frag_type}, neighbors={self.neighbors})"

class MolTree:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.nodes = []
        self.edges = []
        self.adj_list = {}
        self.root = None
        self.vocab_id = None
        self.tree_graph = nx.Graph()

    def add_node(self, node: MolTreeNode):
        self.nodes.append(node)
        self.adj_list[node.idx] = []
        self.tree_graph.add_node(node.idx)

    def add_edge(self, u: int, v: int):
        self.edges.append((u, v))
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)
        self.nodes[u].add_neighbor(v)
        self.nodes[v].add_neighbor(u)
        self.tree_graph.add_edge(u, v)
    
    def get_node_idx(self, node: MolTreeNode):
        return node.idx if node in self.nodes else -1

    def get_node(self, idx: int):
        return self.nodes[idx] if 0 <= idx < len(self.nodes) else None
    
    def get_neighbors(self, idx: int):
        return self.adj_list.get(idx, [])

    def __repr__(self):
        return f"MolTree(smiles='{self.smiles}', nodes={len(self.nodes)}, edges={len(self.edges)})"
