# ...existing code...
import torch
import torch.nn as nn
from utils import GRU, create_var
from collections import deque

ATOM_FDIM = 8
BOND_FDIM = 8
MAX_NB    = 8


class JTNNEncoder(nn.Module):
    def __init__(self, vocab, hidden_size, embedding=None):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.get_size()
        self.vocab = vocab

        self.W = nn.Linear(2 * hidden_size, hidden_size)
        
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding
    
    def forward(self, mol_tree_batch):
        orders = []
        # root_batch as node objects
        root_batch = [mol_tree.root for mol_tree in mol_tree_batch]
        for mol_tree in mol_tree_batch:
            order = get_prop_order(mol_tree)
            orders.append(order)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        msg_dict = {}
        padding = create_var(torch.zeros(self.hidden_size, device=device), False)

        MAX_DEPTH = max([len(x) for x in orders]) if orders else 0

        for t in range(MAX_DEPTH):
            prop_list = []

            for order in orders:
                if t < len(order):
                    prop_list.append(order[t])
            i_emb = []
            msg_nei_list = []

            # prop_list contains tuples of (node_i, node_j)
            for node_i, node_j in prop_list:
                i, j = node_i.idx, node_j.idx
                i_emb.append(self.vocab.get_index(node_i.smiles))

                msg_nei = []
                for node_k in node_i.neighbors:
                    k = node_k.idx
                    if k == j:
                        continue
                    if (k, i) in msg_dict:
                        msg_nei.append(msg_dict[(k, i)])
                    else:
                        msg_nei.append(padding)
                
                pad_len = MAX_NB - len(msg_nei)
                msg_nei.extend([padding] * pad_len)
                msg_nei_list.extend(msg_nei)
            
            if len(i_emb) == 0:
                continue
            device = next(self.parameters()).device
            i_emb = torch.LongTensor(i_emb).to(device)

            i_emb = self.embedding(i_emb)
            # msg_nei_list -> list of tensors length len(prop_list)*MAX_NB
            msg_nei_list = [m.to(device) for m in msg_nei_list]
            msg_nei_list = torch.stack(msg_nei_list, dim=0)
            msg_nei_list = msg_nei_list.view(-1, MAX_NB, self.hidden_size)

            gru = GRU(self.hidden_size, self.hidden_size).to(device)

            new_msg = gru(i_emb, msg_nei_list)  # GRU accepts tensor/list

            for idx, m in enumerate(prop_list):
                i, j = m[0].idx, m[1].idx
                msg_dict[(i, j)] = new_msg[idx]
        ###Node aggregation 
        root_vecs = node_aggregate(root_batch, msg_dict, self.embedding, self.W, self.vocab)

        return msg_dict, root_vecs

"""Helper Functions"""     

def get_prop_order(mol_tree, root_idx=None):
    """
    Return a single flat list of directed edges as (node_u, node_v) objects:
      1) all leaf->root edges in post-order (children before parent)
      2) all root->leaf edges in pre-order (parent before children)
    """
    # --- edge cases + root selection ---
    if not mol_tree.nodes:
        return []
    # mol_tree.root might be a node object or an int; normalize to index
    if root_idx is None:
        root_idx = mol_tree.root.idx if hasattr(mol_tree.root, "idx") else mol_tree.root
    if root_idx is None or root_idx < 0 or root_idx >= len(mol_tree.nodes):
        raise ValueError("Invalid or missing root index for MolTree.")

    # --- orient the tree (build parent/children lists) ---
    N = len(mol_tree.nodes)
    parent = [-1] * N
    children = [[] for _ in range(N)]

    q = deque([root_idx])
    parent[root_idx] = root_idx
    while q:
        u = q.popleft()
        # get_neighbors returns node objects; convert to indices
        nbrs = mol_tree.get_neighbors(u)
        for v_node in nbrs:
            v = v_node.idx if hasattr(v_node, "idx") else v_node
            if parent[v] == -1:              # not visited
                parent[v] = u
                children[u].append(v)
                q.append(v)

    # --- phase 1: leaf -> root (post-order) ---
    up_edges_idx = []
    def dfs_post(u):
        for v in children[u]:
            dfs_post(v)
            up_edges_idx.append((v, u))  # child -> parent
    dfs_post(root_idx)

    # --- phase 2: root -> leaf (pre-order) ---
    down_edges_idx = []
    def dfs_pre(u):
        for v in children[u]:
            down_edges_idx.append((u, v))  # parent -> child
            dfs_pre(v)
    dfs_pre(root_idx)

    # --- convert (idx, idx) -> (node, node) and concatenate ---
    to_node_pair = lambda a, b: (mol_tree.get_node(a), mol_tree.get_node(b))
    up_edges_nodes   = [to_node_pair(a, b) for (a, b) in up_edges_idx]
    down_edges_nodes = [to_node_pair(a, b) for (a, b) in down_edges_idx]

    return up_edges_nodes + down_edges_nodes



def node_aggregate(nodes, msg_dict, embedding, W, vocab):

    i_vid = []     # collect vocab ids for nodes
    msg_nei = []   # collect incoming neighbor messages (with padding)
    hidden_size = embedding.embedding_dim
    padding = create_var(torch.zeros(hidden_size), False)
    for node_i in nodes:
        i_vid.append(vocab.get_index(node_i.smiles))

        # neighbor list stores node objects; use tuple key for msg_dict
        nei = [msg_dict[(node_j.idx, node_i.idx)] for node_j in node_i.neighbors]
        pad_len = MAX_NB - len(nei)
        nei.extend([padding] * pad_len)
        msg_nei.extend(nei)
    msg_nei = torch.cat(msg_nei, dim=0).view(-1, MAX_NB, hidden_size)

    sum_msg_nei = msg_nei.sum(dim=1)  # [len(nodes), hidden_size]
    i_emb = create_var(torch.LongTensor(i_vid))
    i_emb = embedding(i_emb)          # [len(nodes), hidden_size]
    root_vec = torch.cat([i_emb, sum_msg_nei], dim=1)  # [len(nodes), 2*hidden_size]
    root_vec = torch.relu(W(root_vec))  # [len(nodes), hidden_size]
    return root_vec

