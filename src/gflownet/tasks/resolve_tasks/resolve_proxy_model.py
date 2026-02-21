import torch
from torch import nn

# Ensure these files are in the same directory
from .mpnn_layer import MPNNLayer
from .readout import SolvReadout

class MPNNModel(nn.Module):
    def __init__(
            self,
            num_layers,
            emb_dim,
            magic_number,
            magic_number_2,
            magic_number_3,
            magic_number_4,
            num_seed_points,
            heads,
            dim_atoms,
            dim_bond,
            emb_dielec,
            emb_refract,
            out_dim=1,
            num_atom_types=120,
            num_bond_types=5
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.edge_dim = emb_dim

        self.atom_type_embedding = nn.Embedding(num_atom_types, dim_atoms)
        self.bond_type_embedding = nn.Embedding(num_bond_types, dim_bond)

        self.lin_in_atoms = nn.Linear(8 + dim_atoms, emb_dim)
        self.lin_in_bonds = nn.Linear(3 + dim_bond, emb_dim)

        self.convs = nn.ModuleList([
            MPNNLayer(
                emb_dim,
                self.edge_dim,
                magic_number,
                magic_number_2,
                magic_number_3,
                magic_number_4,
                aggr="add",
            )
            for _ in range(num_layers)
        ])

        self.readout = SolvReadout(
            emb_dim=emb_dim,
            num_seed_points=num_seed_points,
            heads=heads,
            emb_dielec=emb_dielec,
            emb_refract=emb_refract,
            out_dim=out_dim,
        )

    def forward(self, data, data_dielec, data_ref):
        atom_embed = self.atom_type_embedding(data.x[:, 0].long())
        other_atom_feats = data.x[:, 1:]
        h = self.lin_in_atoms(torch.cat([atom_embed, other_atom_feats], dim=1))

        bond_embed = self.bond_type_embedding(data.edge_attr[:, 0].long())
        other_bond_feats = torch.cat(
            [data.edge_attr[:, :1], data.edge_attr[:, 2:5], data.edge_attr[:, 6:]], dim=1
        )
        msg_e = self.lin_in_bonds(torch.cat([bond_embed, other_bond_feats], dim=1))

        for conv in self.convs:
            h_next, msg_next = conv(h, data.edge_index, msg_e)
            h = h + h_next
            msg_e = msg_e + msg_next

        return self.readout(h, msg_e, data, data_dielec, data_ref)


class ResolveProxyWrapper(nn.Module):
    """
    Adapts MPNNModel(data, data_dielec, data_ref)
    → returns the solvent-corrected reduction potential (Column 1).
    """

    def __init__(self, mpnn: MPNNModel):
        super().__init__()
        self.mpnn = mpnn
        self.device = next(mpnn.parameters()).device

    def forward(self, batch, dielec=None, ref=None):
        num_graphs = batch.num_graphs

        # Ensure solvent tensors match the batch size
        if dielec is None:
            dielec = torch.zeros(num_graphs, device=self.device)
        else:
            if dielec.dim() == 0 or (dielec.dim() == 1 and dielec.size(0) == 1):
                dielec = dielec.expand(num_graphs)

        if ref is None:
            ref = torch.zeros(num_graphs, device=self.device)
        else:
            if ref.dim() == 0 or (ref.dim() == 1 and ref.size(0) == 1):
                ref = ref.expand(num_graphs)

        # Get raw model output: shape [Batch, 2]
        # Column 0: -EA (Gas Phase)
        # Column 1: -EA + Solv_Correction (In Solvent)
        raw_out = self.mpnn(batch, dielec, ref)

        # We return only Column 1 (the solvent-dependent reduction potential)
        if raw_out.dim() > 1 and raw_out.size(1) > 1:
            return raw_out[:, 1]

        return raw_out