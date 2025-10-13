import pytorch3d.ops
import torch
import torch.nn as nn
import pytorch3d
from torch_geometric.utils import coalesce, scatter, cumsum
from torch_geometric.nn import MessagePassing
from modular_policy.model.common.module_attr_mixin import ModuleAttrMixin
from typing import Tuple, Optional

"""
NOTE Currently for performance reasons, we assume homogeneous batch (samples have the same number of points).
But it's not hard to extend to support heterogeneous batch.
"""


# reference: torch_geometric.transforms.line_graph
def linegraph_of_digraph(
    edges: torch.Tensor,
    edge_indices: torch.Tensor,
    num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # nodes: (N, Dn)
    # edges: (E, De)
    # edge_indices: (2, E)
    # return: nodes, edge_indices of the line graph

    edge_indices, edges = coalesce(edge_indices, edges, num_nodes=num_nodes)
    row, col = edge_indices
    i = torch.arange(row.size(0), dtype=torch.long, device=row.device)
    count = scatter(torch.ones_like(row), row, dim=0, dim_size=num_nodes, reduce="sum")
    ptr = cumsum(count)
    cols = [i[ptr[col[j]]:ptr[col[j] + 1]] for j in range(col.size(0))]
    rows = [row.new_full((c.numel(), ), j) for j, c in enumerate(cols)]
    row, col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)
    edge_indices = torch.stack([row, col], dim=0)
    return edges, edge_indices


def compute_connectivity(
    points: torch.Tensor,
    knn: int = 5,
) -> torch.Tensor:
    # points: (B, N, D)
    # knn: int, number of nearest neighbors
    # threshold_radius: float, maximum distance for two points to be connected
    # return: (2, E) where E is the number of edges

    device = points.device
    
    # compute knn
    dists, senders, _ = pytorch3d.ops.knn_points(
        p1=points, p2=points, K=knn + 1, 
        return_nn=False, return_sorted=False
    )
    senders[torch.allclose(dists, torch.tensor(0.))] = -1

    # convert to edge indices
    offset = torch.arange(points.size(0), device=device) * points.size(1)
    senders += offset[:, None, None]
    receivers = torch.arange(points.size(0) * points.size(1), device=device).repeat_interleave(knn + 1)
    senders = senders.view(-1)

    # remove invalid indices and self-loop
    mask = (senders != -1) & (senders != receivers)
    senders = senders[mask]
    receivers = receivers[mask]

    return torch.stack([senders, receivers])


def build_graph(
    point_coords: torch.Tensor,
    point_feats: Optional[torch.Tensor] = None,
    knn: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # point_coords: (B, N, 3)
    # point_feats: (B, N, Ds), optional, to distinguish different types of nodes
    # knn: int, number of nearest neighbors
    # return: nodes (B * N, Dn), edges (E, De), edge_indices (2, E)

    device = point_coords.device
    dtype = point_coords.dtype

    edge_indices = compute_connectivity(point_coords, knn)
    senders, receivers = edge_indices

    B, N, _ = point_coords.shape
    point_coords = point_coords.view(B * N, -1)
    if point_feats is not None:
        point_feats = point_feats.view(B * N, -1)

    # TODO: experiment with different node/edge features
    nodes = []
    if point_feats is not None:
        nodes.append(point_feats)

    edges = []
    edges.append(point_coords[senders] - point_coords[receivers])
    if point_feats is not None:
        edges.append(point_feats[senders])
        edges.append(point_feats[receivers])
    edges = torch.cat(edges, dim=-1)
    
    if len(nodes) == 0:     # add dummy node feature
        nodes = torch.zeros((B * N, 1), device=device, dtype=dtype)
    else:
        nodes = torch.cat(nodes, dim=-1)

    return nodes, edges, edge_indices


def build_graph_encoder(
    node_input_dim, 
    node_output_dim,
    edge_input_dim,
    num_propagation_steps: int = 3,
):
    import warnings
    warnings.warn("remove node encoder, propagator", DeprecationWarning)
    
    class MLP(ModuleAttrMixin):
        def __init__(self,
            input_dim,
            output_dim,
            num_layers,
            last_act: bool = True,
        ):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, output_dim))
            for _ in range(num_layers - 1):
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Linear(output_dim, output_dim))
            if last_act:
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Identity())
            
        def forward(self, x, residual=None):
            for layer in self.layers[:-1]:
                x = layer(x)
            if residual is not None:
                x += residual
            return self.layers[-1](x)
        
    class GraphEncoder(MessagePassing):
        def __init__(self, 
            node_input_dim, 
            node_output_dim,
            edge_input_dim,
            num_propagation_steps: int = 3,
        ):
            super().__init__()
            self.node_enc = MLP(node_input_dim, node_output_dim, num_layers=3, last_act=True)
            self.edge_enc = MLP(edge_input_dim, node_output_dim, num_layers=3, last_act=True)
            self.node_prop = MLP(2*node_output_dim, node_output_dim, num_layers=1, last_act=True)
            self.edge_prop = MLP(3*node_output_dim, node_output_dim, num_layers=1, last_act=True)
            self.num_propagation_steps = num_propagation_steps

        def forward(self, nodes, edges, edge_indices):
            nodes = self.node_enc(nodes)
            edges = self.edge_enc(edges)
            message = nodes
            for _ in range(self.num_propagation_steps):
                message = self.node_prop(torch.cat([
                    nodes,
                    self.propagate(
                        edge_index=edge_indices,
                        x=message,
                        edges=edges
                    )
                ], dim=-1), residual=message)
            return message
        
        # override MessagePassing class method
        def message(self, x_i, x_j, edges):
            return self.edge_prop(torch.cat([x_i, x_j, edges], dim=-1))

    return GraphEncoder(node_input_dim, node_output_dim, edge_input_dim, num_propagation_steps)
