# ...existing code...
from rdkit import Chem
from rdkit.Chem import rdmolops
from Mol_Tree import MolTreeNode, MolTree
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, tanh
from collections import defaultdict
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
MST_MAX_WEIGHT = 100

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
                # pass node objects (MolTree.add_edge accepts nodes or indices)
                mol_tree.add_edge(mol_tree.nodes[i], mol_tree.nodes[j])

    # Optional: set root to a node object (not an int)
    mol_tree.root = mol_tree.nodes[0] if mol_tree.nodes else None

    return mol_tree

def set_batch_nodeID(mol_batch, vocab):
    id = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = id
            node.label = vocab.get_index(node.smiles)
            id += 1

import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid(x):
    return torch.sigmoid(x)

def tanh(x):
    return torch.tanh(x)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_i, msg_list):
        device = x_i.device  # ensure all tensors are on same device as x_i

        # msg_list may be a tensor or a list
        if isinstance(msg_list, torch.Tensor):
            msg_list = msg_list.to(device)
            msg_list_sum = msg_list.sum(dim=1)
        else:
            # list of tensors -> move each to correct device then stack
            msg_list = [m.to(device) for m in msg_list]
            msg_list_sum = torch.sum(torch.stack(msg_list), dim=0)

        # Compute GRU update
        z = sigmoid(self.W_z(torch.cat([x_i, msg_list_sum], dim=1)))
        r = sigmoid(self.W_r(x_i) + self.U_r(msg_list_sum))
        msg_tilde = tanh(self.W(torch.cat([x_i, r * msg_list_sum], dim=1)))
        msg_new = (1 - z) * msg_list_sum + z * msg_tilde

        return msg_new

def create_var(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile=volatile)

def tree_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1,a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Merge Rings with intersection > 2 atoms
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []
    
    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
    
    #Build edges and add singleton cliques
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): #In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = 1
        elif len(rings) > 2: #Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1,c2 = cnei[i],cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):
                        edges[(c1,c2)] = len(inter) #cnei[i] < cnei[j] by construction

    edges = [u + (MST_MAX_WEIGHT-v,) for u,v in edges.items()]
    if len(edges) == 0:
        return cliques, edges

    #Compute Maximum Spanning Tree
    row,col,data = list(zip(*edges))
    n_clique = len(cliques)
    clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
    junc_tree = minimum_spanning_tree(clique_graph)
    row,col = junc_tree.nonzero()
    edges = [(row[i],col[i]) for i in range(len(row))]
    return (cliques, edges)
