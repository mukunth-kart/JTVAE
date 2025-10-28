import torch
import torch.nn as nn
from rdkit import Chem

ATOM_FEAT_DIM = 5  
EDGE_FEAT_DIM = 8  

def init_node_features(mol):
    features = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization().real,
            atom.GetIsAromatic(),
        ]
        features.append(feat)
        #print(f"Atom {atom.GetIdx()} features: {feat}")
    return torch.tensor(features, dtype=torch.float)


def init_edge_features(mol):
    features = []
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        feat = [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.GetBondTypeAsDouble(),
            bond.GetIsConjugated(),
            bond.IsInRing(),
            bond_type == Chem.rdchem.BondType.UNSPECIFIED,
        ]
        features.append(feat)
    return torch.tensor(features, dtype=torch.float)

class GMPN(nn.Module):
    def __init__(self, hidden_size, depth):
        super(GMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W1 = nn.Linear(ATOM_FEAT_DIM, hidden_size)
        self.W2 = nn.Linear(EDGE_FEAT_DIM, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size)

        self.U1 = nn.Linear(hidden_size, hidden_size)
        self.U2 = nn.Linear(hidden_size, hidden_size)
    def forward(self, Graph):

        atom_f = init_node_features(Graph.mol)   # Initial node features
        bond_f = init_edge_features(Graph.mol)   # Initial edge features
        print(atom_f[0].shape, atom_f[1].shape, atom_f[2].shape)
        atom_f = torch.stack((atom_f[0],atom_f[1],atom_f[2]), 0)
        bond_f = torch.stack([bond_f[i] for i in range(len(Graph.edges))], dim=0)
        print(f"atom_f shape: {atom_f.shape}, bond_f shape: {bond_f.shape}")

        atom_embeddings = self.W1(atom_f)  # Initial atom embeddings
        bond_embeddings = self.W2(bond_f)  # Initial bond embeddings
        msg_list = [[] for _ in range(len(Graph.edges))]
        msg = torch.zeros(self.hidden_size)
        for _ in range(self.depth):
            for bond_idx, (u,v) in enumerate(Graph.edges):
                for w in Graph.get_neighbors(u):
                    if w != v:
                        msg_list[bond_idx].append(msg)
                if msg_list[bond_idx]:
                    msg_sum = torch.sum(torch.stack(msg_list[bond_idx]), dim=0)
                else:
                    msg_sum = torch.zeros(self.hidden_size)
                #size check
                print(f"atom_embeddings shape: {atom_embeddings.shape}, bond_embeddings shape: {bond_embeddings.shape}, msg_sum shape: {msg_sum.shape}")
                msg = torch.relu(atom_embeddings+bond_embeddings+self.W3(msg_sum))
                msg_list[bond_idx] = [msg]
        # Aggregate messages to get final graph representation
        h_list = []
        for node_idx in len(Graph.nodes):
            node_msgs = []
            for bond_idx, (u,v) in enumerate(Graph.edges):
                if v == node_idx:
                    node_msgs.append(msg_list[bond_idx][0])
            if node_msgs:
                h_2 = torch.sum(torch.stack(node_msgs), dim=0)
            else:
                h_2 = torch.sum(torch.zeros(self.hidden_size))
            h_1 = self.U1(atom_embeddings[node_idx])
            h_2 = self.U2(h_2)
            h_u = torch.relu(h_1 + h_2)
            h_list.append(h_u)
        h_G = torch.sum(torch.stack(h_list), dim=0)/len(Graph.nodes)
        return h_G 
