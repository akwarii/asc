from collections import defaultdict
from typing import Optional

import dgl
import numpy as np
import torch


def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
):
    """Construct k-NN edge list."""
    all_neighbors = atoms.get_all_neighbors(r=cutoff)

    # if a site has too few neighbors, increase the cutoff radius
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1

        return nearest_neighbor_edges(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )
    # build up edge list
    # NOTE: currently there's no guarantee that this creates undirected graphs
    # An undirected solution would build the full edge list where nodes are
    # keyed by (index, image), and ensure each edge has a complementary edge

    # indeed, JVASP-59628 is an example of a calculation where this produces
    # a graph where one site has no incident edges!

    # build an edge dictionary u -> v
    # so later we can run through the dictionary
    # and remove all pairs of edges
    # so what's left is the odd ones out
    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):
        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]

        # keep all edges out to the neighbor shell of the k-th neighbor
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]

        # keep track of cell-resolved edges
        # to enforce undirected graph construction
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges


def build_undirected_edgedata(
    atoms=None,
    edges={},
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    u, v, r = [], [], []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )

            # add edges for both directions
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
                
    u, v, r = (np.array(x) for x in (u, v, r))
    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r).type(torch.get_default_dtype())

    return u, v, r


def radius_graph(
    atoms=None,
    cutoff=5,
    bond_tol=0.5,
    id=None,
    atol=1e-5,
    cutoff_extra=3.5,
):
    """Construct edge list for radius graph."""

    def temp_graph(cutoff=5):
        """Construct edge list for radius graph."""
        cart_coords = torch.tensor(atoms.cart_coords).type(
            torch.get_default_dtype()
        )
        frac_coords = torch.tensor(atoms.frac_coords).type(
            torch.get_default_dtype()
        )
        lattice_mat = torch.tensor(atoms.lattice_mat).type(
            torch.get_default_dtype()
        )
        # elements = atoms.elements
        X_src = cart_coords
        num_atoms = X_src.shape[0]
        # determine how many supercells are needed for the cutoff radius
        recp = 2 * np.pi * torch.linalg.inv(lattice_mat).T
        recp_len = torch.tensor(
            [i for i in (torch.sqrt(torch.sum(recp**2, dim=1)))]
        )
        maxr = torch.ceil((cutoff + bond_tol) * recp_len / (2 * np.pi))
        nmin = torch.floor(torch.min(frac_coords, dim=0)[0]) - maxr
        nmax = torch.ceil(torch.max(frac_coords, dim=0)[0]) + maxr
        # construct the supercell index list

        all_ranges = [
            torch.arange(x, y, dtype=torch.get_default_dtype())
            for x, y in zip(nmin, nmax)
        ]
        cell_images = torch.cartesian_prod(*all_ranges)

        # tile periodic images into X_dst
        # index id_dst into X_dst maps to atom id as id_dest % num_atoms
        X_dst = (cell_images @ lattice_mat)[:, None, :] + X_src
        X_dst = X_dst.reshape(-1, 3)
        # pairwise distances between atoms in (0,0,0) cell
        # and atoms in all periodic image
        dist = torch.cdist(
            X_src, X_dst, compute_mode="donot_use_mm_for_euclid_dist"
        )
        neighbor_mask = torch.bitwise_and(
            dist <= cutoff,
            ~torch.isclose(
                dist,
                torch.tensor([0]).type(torch.get_default_dtype()),
                atol=atol,
            ),
        )
        # get node indices for edgelist from neighbor mask
        u, v = torch.where(neighbor_mask)

        r = (X_dst[v] - X_src[u]).float()
        v = v % num_atoms
        g = dgl.graph((u, v))
        return g, u, v, r

    g, u, v, r = temp_graph(cutoff)
    while (g.num_nodes()) != len(atoms.elements):
        try:
            cutoff += cutoff_extra
            g, u, v, r = temp_graph(cutoff)
            print("cutoff", id, cutoff)
            print(atoms)

        except Exception as exp:
            print("Graph exp", exp)
            pass
        return u, v, r

    return u, v, r


def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return {"h": bond_cosine}


class Graph:
    """Generate a graph object."""

    def __init__(
        self,
        nodes=[],
        node_attributes=[],
        edges=[],
        edge_attributes=[],
        color_map=None,
        labels=None,
    ):
        """
        Initialize the graph object.

        Args:
            nodes: IDs of the graph nodes as integer array.

            node_attributes: node features as multi-dimensional array.

            edges: connectivity as a (u,v) pair where u is
                   the source index and v the destination ID.

            edge_attributes: attributes for each connectivity.
                             as simple as euclidean distances.
        """
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    @staticmethod
    def atom_dgl_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = False,
        use_lattice_prop: bool = False,
        cutoff_extra=3.5,
    ):
        """Obtain a DGLGraph for Atoms object."""
        if neighbor_strategy == "k-nearest":
            edges = nearest_neighbor_edges(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
            )
            u, v, r = build_undirected_edgedata(atoms, edges)
        elif neighbor_strategy == "radius_graph":
            u, v, r = radius_graph(
                atoms, cutoff=cutoff, cutoff_extra=cutoff_extra
            )
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)

        # build up atom attribute tensor
        g = dgl.graph((u, v))
        g.edata["r"] = r
        g.ndata["coords"] = torch.tensor(atoms.cart_coords)

        if compute_line_graph:
            lg = g.line_graph(shared=True)
            lg.apply_edges(compute_bond_cosines)
            return g, lg
        else:
            return g