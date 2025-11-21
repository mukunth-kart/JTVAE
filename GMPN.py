import torch
import torch.nn as nn
from rdkit import Chem

# ---------- small helpers ----------
def _edge_feats_from_uv(mol, u, v):
    b = mol.GetBondBetweenAtoms(int(u), int(v))
    if b is None:
        return [0,0,0,0,0.0,0,0,0]
    bt = b.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(b.GetBondTypeAsDouble()),
        float(b.GetIsConjugated()),
        float(b.IsInRing()),
        float(bt == Chem.rdchem.BondType.UNSPECIFIED),
    ]

def index_select_ND(source, dim, index):
    """Select along dim=0 with an arbitrary-shaped index tensor; reshape back."""
    assert dim == 0, "index_select_ND assumes select along dim=0"
    flat = index.reshape(-1)
    gathered = source.index_select(0, flat)
    return gathered.reshape(*index.shape, source.size(1))

# ---------- batched collate (MolGraph -> fatoms/fbonds/agraph/bgraph/scope) ----------
def _collate_graphs(graphs):
    """Batch a list[MolGraph] -> (fatoms, fbonds, agraph, bgraph, scope)."""
    if not isinstance(graphs, (list, tuple)):
        graphs = [graphs]

    ATOM_FDIM = len(graphs[0].nodes[0].feature_vector)
    BOND_FDIM = 8

    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)  # fbonds[0], 1-index bonds
    fatoms, fbonds = [], [padding]
    in_bonds = []                  # incoming directed bond-ids per atom (global indexing)
    all_bonds = [(-1, -1)]         # 1-indexed: b_id -> (x,y) in global atom idx
    scope = []
    total_atoms = 0

    for G in graphs:
        mol = G.mol
        n_atoms = len(G.nodes)

        # atoms
        for node in G.nodes:
            fatoms.append(torch.tensor(node.feature_vector, dtype=torch.float))
            in_bonds.append([])

        # undirected edges -> two directed bonds
        for (u, v) in G.edges:
            x = u + total_atoms
            y = v + total_atoms

            # u->v
            b = len(all_bonds)
            all_bonds.append((x, y))
            bf = _edge_feats_from_uv(mol, u, v)
            fbonds.append(torch.cat([fatoms[x], torch.tensor(bf, dtype=torch.float)], dim=0))
            in_bonds[y].append(b)

            # v->u
            b = len(all_bonds)
            all_bonds.append((y, x))
            bf = _edge_feats_from_uv(mol, v, u)
            fbonds.append(torch.cat([fatoms[y], torch.tensor(bf, dtype=torch.float)], dim=0))
            in_bonds[x].append(b)

        scope.append((total_atoms, n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds) - 1  # minus padding

    fatoms = torch.stack(fatoms, 0) if fatoms else torch.zeros(0, ATOM_FDIM)
    fbonds = torch.stack(fbonds, 0) if len(fbonds) else torch.zeros(1, ATOM_FDIM + BOND_FDIM)

    # agraph: incoming bonds per atom
    max_nb_atom = max((len(lst) for lst in in_bonds), default=1)
    agraph = torch.zeros(total_atoms, max_nb_atom, dtype=torch.long)
    for a in range(total_atoms):
        for i, b in enumerate(in_bonds[a][:max_nb_atom]):
            agraph[a, i] = b

    # bgraph: for each directed bond b1=(x->y), neighbors are incoming bonds of x except reverse (y->x)
    max_nb_bond = 1
    for b1 in range(1, total_bonds + 1):
        x, y = all_bonds[b1]
        cand = [b2 for b2 in in_bonds[x] if all_bonds[b2][0] != y]
        if cand:
            max_nb_bond = max(max_nb_bond, len(cand))

    bgraph = torch.zeros(total_bonds + 1, max_nb_bond, dtype=torch.long)  # include padding row
    for b1 in range(1, total_bonds + 1):
        x, y = all_bonds[b1]
        cand = [b2 for b2 in in_bonds[x] if all_bonds[b2][0] != y]
        for i, b2 in enumerate(cand[:max_nb_bond]):
            bgraph[b1, i] = b2

    return (fatoms, fbonds, agraph, bgraph, scope, ATOM_FDIM, BOND_FDIM)

# ---------- Batched GMPN (MPN-style) ----------
class GMPN(nn.Module):
    """
    Batched GMPN that mirrors the classic MPN (fatoms/fbonds/agraph/bgraph/scope).
    Input to forward: either a MolGraph or a List[MolGraph].
    Output: [B, H] molecule embeddings (B=len(graphs)).
    """
    def __init__(self, hidden_size: int, depth: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        # Will lazy-init after seeing atom/bond dims
        self.W_i = None  # nn.Linear(ATOM_FDIM + BOND_FDIM, H, bias=False)
        self.W_h = None  # nn.Linear(H, H, bias=False)
        self.W_o = None  # nn.Linear(ATOM_FDIM + H, H)

    def _lazy_init(self, atom_fdim, bond_fdim, device):
        if self.W_i is None:
            self.W_i = nn.Linear(atom_fdim + bond_fdim, self.hidden_size, bias=False).to(device)
            self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(device)
            self.W_o = nn.Linear(atom_fdim + self.hidden_size, self.hidden_size).to(device)

    def forward(self, graphs):
        # Collate graphs -> batched tensors
        fatoms, fbonds, agraph, bgraph, scope, ATOM_FDIM, BOND_FDIM = _collate_graphs(graphs)

        # choose device explicitly (no reliance on parameters())
        device = next((p.device for p in self.parameters()), torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        fatoms = fatoms.to(device)
        fbonds = fbonds.to(device)
        agraph = agraph.to(device)
        bgraph = bgraph.to(device)

        self._lazy_init(ATOM_FDIM, 8, device)

        # Bond messages
        binput = self.W_i(fbonds)           # [B+1, H]
        message = torch.relu(binput)        # [B+1, H]

        for _ in range(max(1, self.depth - 1)):
            nei_message = index_select_ND(message, 0, bgraph)  # [B+1, maxNB, H]
            nei_message = nei_message.sum(dim=1)               # [B+1, H]
            nei_message = self.W_h(nei_message)                # [B+1, H]
            message = torch.relu(binput + nei_message)         # [B+1, H]

        # Aggregate incoming bond messages per atom
        nei_message = index_select_ND(message, 0, agraph)      # [A, maxNB, H]
        nei_message = nei_message.sum(dim=1)                   # [A, H]

        # Atom update -> hidden
        ainput = torch.cat([fatoms, nei_message], dim=1)       # [A, Fa+H]
        atom_hiddens = torch.relu(self.W_o(ainput))            # [A, H]

        # Pool per molecule using scope
        mol_vecs = []
        for st, le in scope:
            if le == 0:
                mol_vecs.append(torch.zeros(self.hidden_size, device=device))
            else:
                mol_vecs.append(atom_hiddens.narrow(0, st, le).mean(dim=0))
        return torch.stack(mol_vecs, dim=0) if mol_vecs else torch.zeros(0, self.hidden_size, device=device)
