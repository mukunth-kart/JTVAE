# ...existing code...
import networkx as nx
import rdkit
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
        self.neighbors = []      # list of MolTreeNode objects
        self.vid = None  # vocab index
        self.feature = None  # placeholder for graph embedding or one-hot features
    
    def __hash__(self):
        # hash by stable identity (idx) so the node can be used in dicts/sets/NetworkX
        return hash(self.idx)

    def __eq__(self, other):
        return isinstance(other, MolTreeNode) and self.idx == other.idx
    
    def add_neighbor(self, node):
        """Add a neighbor node object (ensure the neighbor list stores node objects)."""
        if node not in self.neighbors:
            self.neighbors.append(node)
    
    def to_dict(self):
        return {
            "idx": self.idx,
            "smiles": self.smiles,
            "atoms": self.atoms,
            "neighbors": [n.idx for n in self.neighbors],
            "label": self.label,
            "frag_type": self.frag_type,
            "is_ring": self.is_ring,
        }
    
    def __repr__(self):
        return f"MolTreeNode(idx={self.idx}, smiles='{self.smiles}', type={self.frag_type}, neighbors={[n.idx for n in self.neighbors]})"


class MolTree:
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.nodes: list[MolTreeNode] = []
        self.edges: list[tuple[MolTreeNode, MolTreeNode]] = []        # (u, v) as nodes
        self.adj: dict[MolTreeNode, list[MolTreeNode]] = {}            # node -> neighbor nodes
        self.root: MolTreeNode | None = None                           # root is a node
        self.vocab_id = None
        self.tree_graph = nx.Graph()                                   # graph of node objects

    # ---- Node-centric API ----
    def add_node(self, node: MolTreeNode):
        """Register a node object (neighbors list will also hold node objects)."""
        self.nodes.append(node)
        self.adj[node] = []
        self.tree_graph.add_node(node)

    def add_edge(self, u, v):
        """Undirected edge between two node objects.
        Accepts either MolTreeNode objects or integer indices for convenience.
        """
        if isinstance(u, int):
            u = self.get_node(u)
        if isinstance(v, int):
            v = self.get_node(v)
        if u is None or v is None:
            raise ValueError("add_edge received invalid node or index")
        self.edges.append((u, v))
        # ensure adj maps node objects -> list[node objects]
        if u not in self.adj:
            self.adj[u] = []
        if v not in self.adj:
            self.adj[v] = []
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)
        u.add_neighbor(v)     # node.neighbors stores node objects
        v.add_neighbor(u)
        self.tree_graph.add_edge(u, v)

    def get_node(self, idx: int) -> MolTreeNode | None:
        """Convenience accessor by idx if you still need it."""
        for n in self.nodes:
            if n.idx == idx:
                return n
        return None

    def get_neighbors(self, node_or_idx) -> list:
        """Return neighbor node objects. Accepts a node object or an index.
        Always returns a list of MolTreeNode objects.
        """
        if isinstance(node_or_idx, int):
            node_obj = self.get_node(node_or_idx)
            if node_obj is None:
                return []
            return self.adj.get(node_obj, [])
        return self.adj.get(node_or_idx, [])

    def __repr__(self):
        return f"MolTree(smiles='{self.smiles}', nodes={len(self.nodes)}, edges={len(self.edges)})"



if __name__ == "__main__":
    import sys
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    cset = set()
    for i,line in enumerate(sys.stdin):
        smiles = line.split()[0]
        try:
            mol = MolTree(smiles)
        except Exception as e:
            print(f"[Line {i+1}] ❌ Failed on SMILES: {smiles}")
            raise e
        for c in mol.nodes:
            cset.add(c.smiles)
    for x in cset:
        print(x)