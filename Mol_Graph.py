from rdkit import Chem
import networkx as nx

class MolGraphNode:
    def __init__(self, idx: int, atom: Chem.Atom):
        self.idx = idx
        self.atom = atom
        self.symbol = atom.GetSymbol()
        self.atomic_num = atom.GetAtomicNum()
        self.degree = atom.GetDegree()
        self.valence = atom.GetTotalValence()
        self.hybridization = str(atom.GetHybridization())
        self.is_aromatic = atom.GetIsAromatic()
        self.formal_charge = atom.GetFormalCharge()
        self.neighbors = []
        self.feature_vector = self._build_feature_vector()

    def add_neighbor(self, neighbor_idx: int):
        if neighbor_idx not in self.neighbors:
            self.neighbors.append(neighbor_idx)

    def _build_feature_vector(self):
        # You can customize this embedding vector as needed
        # Example: a compact atom feature vector
        return [
            self.atomic_num,
            self.degree,
            self.valence,
            int(self.is_aromatic),
            self.formal_charge
        ]

    def __repr__(self):
        return f"MolGraphNode(idx={self.idx}, symbol='{self.symbol}', degree={self.degree})"


class MolGraph:
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.nodes = []
        self.edges = []
        self.adj_list = {}
        self.graph = nx.Graph()

        if self.mol is not None:
            self._build_graph()

    def _build_graph(self):
        # Build nodes
        for atom in self.mol.GetAtoms():
            node = MolGraphNode(atom.GetIdx(), atom)
            self.add_node(node)

        # Build edges
        for bond in self.mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            #bond_type = str(bond.GetBondType())
            self.add_edge(u, v)

    def add_node(self, node: MolGraphNode):
        self.nodes.append(node)
        self.adj_list[node.idx] = []
        self.graph.add_node(node.idx)

    def add_edge(self, u: int, v: int):
        self.edges.append((u, v))
        self.adj_list[u].append((v))
        self.adj_list[v].append((u))
        self.nodes[u].add_neighbor(v)
        self.nodes[v].add_neighbor(u)
        self.graph.add_edge(u, v)

    def get_node(self, idx: int):
        return self.nodes[idx] if 0 <= idx < len(self.nodes) else None

    def get_neighbors(self, idx: int):
        return self.adj_list.get(idx, [])

    def __repr__(self):
        return f"MolGraph(smiles='{self.smiles}', nodes={len(self.nodes)}, edges={len(self.edges)})"

