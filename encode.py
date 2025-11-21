import torch
from Mol_Graph import MolGraph
from GMPN import GMPN
from JTNN import JTNNEncoder
from utils import create_var, GRU, tree_decompose
from Mol_Tree import Vocab, MolTree, MolTreeNode

def encode(smiles, vocab, hidden_size):
    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### Tree Decomposition
    moltrees = [tree_decompose(smi, vocab) for smi in smiles]
    for moltree in moltrees:
        moltree.root = moltree.nodes[0]
        # Move node features to device if they exist
        for node in moltree.nodes:
            if hasattr(node, 'features'):
                node.features = node.features.to(device)

    ### Graph Preparation
    graphs = [MolGraph(smi) for smi in smiles]
    for g in graphs:
        # Move internal tensors to device
        if hasattr(g, 'node_features'):
            g.node_features = g.node_features.to(device)
        if hasattr(g, 'adj'):
            g.adj = g.adj.to(device)

    ### Model Initialization
    jtnn = JTNNEncoder(vocab, hidden_size=hidden_size).to(device)  # <-- Move to device
    model = GMPN(hidden_size=hidden_size, depth=3).to(device)
    model.eval()
    
    ### Tree Encoder
    msg_dict, root_vec = jtnn(moltrees)  # jtnn and moltrees now on same device
    
    ### Graph Encoder
    with torch.no_grad():
        H = model(graphs)   # All graph tensors on same device
    
    return msg_dict, root_vec, H

if __name__ == "__main__":
    smiles = ["CCO", "CCN", "CCC", "CCCl"]
    clusters = ["C", "O", "N", "Cl", "CC", "CO", "CN", "CCl"]
    vocab = Vocab(clusters)

    msg_dict, root_vec, H = encode(smiles, vocab, hidden_size=128)
    print(msg_dict)
    print("Root Vectors Size:", root_vec.size())
    print("Graph Vectors Size:", H.size())

    ### Mean and Variance of Graph Vectors
    mean = torch.mean(H, dim=0)
    var = torch.var(H, dim=0)

    ### Mean and Variance of Root Vectors
    mean_root = torch.mean(root_vec, dim=0)
    var_root = torch.var(root_vec, dim=0)

    print("Graph Vectors Mean:", mean)
    print("Graph Vectors Variance:", var)   
    print("Root Vectors Mean:", mean_root)
    print("Root Vectors Variance:", var_root)

    ### Sampling Reparametrization Trick
    z_graph = mean + torch.sqrt(var) * create_var(torch.randn_like(mean))
    z_root = mean_root + torch.sqrt(var_root) * create_var(torch.randn_like(mean_root))

    z = torch.cat([z_graph, z_root], dim=-1)
    print("Sampled Latent Vector z:", z)
    print("Sampled Latent Vector z Size:", z.size())
