from rdkit import Chem
from rdkit.Chem import rdmolops
from Mol_Tree import MolTreeNode, MolTree
import torch
from torch.nn import Module
from torch.nn.functional import sigmoid, tanh
def smiles2graph(smiles):
    """
    Converts SMILES into a molecular graph representation (V, E).
    Returns:
        V: list of atom features (dicts)
        E: list of (u, v, bond_type)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    V = []
    for atom in mol.GetAtoms():
        V.append({
            "atom_idx": atom.GetIdx(),
            "symbol": atom.GetSymbol(),
            "atomic_num": atom.GetAtomicNum(),
            "degree": atom.GetDegree(),
            "is_aromatic": atom.GetIsAromatic(),
            "num_hs": atom.GetTotalNumHs(),
        })

    E = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        E.append((u, v, bond.GetBondTypeAsDouble()))
        E.append((v, u, bond.GetBondTypeAsDouble()))  # undirected

    return V, E

def tree_decompose(smiles, vocab):
    """
    Decompose a molecule into a tree of fragments based on vocab.
    Returns a MolTree object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol_tree = MolTree(smiles)

    # Step 1: find rings
    ri = mol.GetRingInfo()
    ring_atom_sets = [list(r) for r in ri.AtomRings()]

    # Step 1.5: merge overlapping rings
    merged = True
    while merged:
        merged = False
        new_ring_sets = []
        skip_indices = set()
        for i in range(len(ring_atom_sets)):
            if i in skip_indices:
                continue
            current_set = set(ring_atom_sets[i])
            for j in range(i + 1, len(ring_atom_sets)):
                if j in skip_indices:
                    continue
                if current_set & set(ring_atom_sets[j]):
                    current_set |= set(ring_atom_sets[j])
                    skip_indices.add(j)
                    merged = True
            new_ring_sets.append(list(current_set))
        ring_atom_sets = new_ring_sets

    # Step 2: create ring nodes
    for i, atoms in enumerate(ring_atom_sets):
        submol = Chem.PathToSubmol(mol, rdmolops.FindAtomEnvironmentOfRadiusN(mol, 1, atoms[0]))
        ring_smiles = Chem.MolToSmiles(submol)
        node = MolTreeNode(ring_smiles, i, atoms=atoms, frag_type="ring")
        mol_tree.add_node(node)

    # Step 3: add additional single-bond fragments from vocab if present
    node_idx = len(ring_atom_sets)
    for v_smi in vocab.smiles:
        v_mol = vocab.mol_list[vocab.get_index(v_smi)]
        if mol.HasSubstructMatch(v_mol):
            for match in mol.GetSubstructMatches(v_mol):
                node = MolTreeNode(v_smi, node_idx, atoms=list(match), frag_type="func_group")
                mol_tree.add_node(node)
                node_idx += 1

    # Step 4: connect nodes if they share atoms (tree edges)
    for i in range(len(mol_tree.nodes)):
        for j in range(i + 1, len(mol_tree.nodes)):
            if set(mol_tree.nodes[i].atoms) & set(mol_tree.nodes[j].atoms):
                mol_tree.add_edge(i, j)

    # Step 5: populate neighbor information into each node's feature
    # (MolTree.add_edge already updates node.neighbors)
    """for node in mol_tree.nodes:
        # copy neighbor indices
        nbr_indices = list(node.neighbors) if node.neighbors else []
        # get neighbor smiles strings (safely)
        nbr_smiles = []
        for nidx in nbr_indices:
            other = mol_tree.get_node(nidx)
            if other is not None:
                nbr_smiles.append(other.smiles)

        # store neighbor info in node.feature for downstream use
        node.feature = {
            "neighbor_indices": nbr_indices,
            "neighbor_smiles": nbr_smiles,
        }"""

    # Optional: set root
    mol_tree.root = 0 if mol_tree.nodes else None

    return mol_tree

def GRU(Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_z = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = torch.nn.Linear(hidden_size, hidden_size)
        self.W = torch.nn.Linear(input_size + hidden_size, hidden_size)
    def forward(self, x_i,msg_list):
        msg_list_sum = torch.sum(torch.stack(msg_list),dim=0)
        z = sigmoid(self.W_z(torch.cat([x_i, msg_list_sum], dim=1)))
        r = sigmoid(self.W_r(x_i) + self.U_r(msg_list_sum))
        msg_tilde = tanh(self.W(torch.cat([x_i, r * msg_list_sum], dim=1)))
        msg_new = (1 - z) * msg_list_sum + z * msg_tilde
        return msg_new

