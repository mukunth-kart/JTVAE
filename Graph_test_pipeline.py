from Mol_Graph import MolGraph, MolGraphNode
from GMPN import GMPN
from utils import smiles2graph, tree_decompose
from Mol_Tree import Vocab

if __name__ == "__main__":
    # Example SMILES
    smiles = "CCO"

    # Convert SMILES to molecular graph
    V, E = smiles2graph(smiles)
    mol_graph = MolGraph(smiles)
    #print(mol_graph)

    # Decompose molecule into tree structure
    vocab = ["C", "O", "CC", "CO", "CCO"]  # Example vocabulary
    vocab = Vocab(vocab)
    mol_tree = tree_decompose(smiles, vocab)
    #print(mol_tree.get_neighbors(1))
    #print(mol_tree.get_node_idx(mol_tree.nodes[0]))
    #print(mol_tree)

    # Initialize GMPN model
    hidden_size = 128
    depth = 3
    gmpn_model = GMPN(hidden_size, depth)

    #attributes of mol_graph
    #print(f"Edges: {mol_graph.edges}")
    #print(f"Adjacency List: {mol_graph.adj_list}")
    # Forward pass through GMPN
    gmpn_model.forward(mol_graph)