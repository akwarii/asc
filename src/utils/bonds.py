import torch
from dgl import DGLGraph

def compute_bond_cosines(edges: DGLGraph) -> torch.Tensor:
    """Compute the cosine of the angle between the bonds of a graph.
    
    Args:
        edges (dgl.DGLGraph): Edges of a graph.
        
    Returns:
        torch.Tensor: The cosine of the angle between the bonds of a graph.
    """
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    
    bond_vectors = r2 - r1
    
    dot_product = torch.sum(bond_vectors * bond_vectors, dim=1)
    norm = torch.norm(bond_vectors, dim=1)
    
    bond_cosine = torch.clamp(dot_product / (norm * norm, -1, 1))
    
    return {"bond_cosine": bond_cosine}