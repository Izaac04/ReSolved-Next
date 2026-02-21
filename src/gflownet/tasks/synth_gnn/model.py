import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class SynthProxyGNN(nn.Module):
    """
    Small multi-task GNN for predicting synthesizability.
    Outputs:
        - success_logit: probability (logit) of AiZynthFinder success
        - route_length: predicted route length (regression)
    """

    def __init__(
            self,
            node_feat_dim: int = 7,
            edge_feat_dim: int = 3,
            hidden_dim: int = 128,
            num_layers: int = 3,
            dropout: float = 0.2,
    ):
        super().__init__()

        # Encoders
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            ) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Heads
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)   # success logit
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)   # route length
        )

    def forward(self, data):
        """
        data.x:         [num_nodes, node_feat_dim]
        data.edge_attr: [num_edges, edge_feat_dim]
        data.edge_index:[2, num_edges]
        data.batch:     [num_nodes] graph ids
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h = self.node_encoder(x.float())
        e = self.edge_encoder(edge_attr.float())

        for conv in self.convs:
            h = conv(h, edge_index, e)
            h = F.relu(h)
            h = self.dropout(h)

        g = global_mean_pool(h, batch)

        success_logit = self.class_head(g).squeeze(-1)
        route_length  = self.reg_head(g).squeeze(-1)
        return success_logit, route_length
