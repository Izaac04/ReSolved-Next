import torch
from pathlib import Path
from .model import SynthProxyGNN


# Resolve path relative to THIS file
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CKPT = _THIS_DIR / "proxy_best.pt"


def load_synth_proxy(
        device: torch.device,
        checkpoint_path: Path = _DEFAULT_CKPT,
):
    ckpt = torch.load(checkpoint_path, map_location=device)

    assert "model_state" in ckpt
    assert "args" in ckpt

    args = ckpt["args"]

    model = SynthProxyGNN(
        node_feat_dim=args["node_feat_dim"],
        edge_feat_dim=args["edge_feat_dim"],
        hidden_dim=args["hidden_dim"],
        num_layers=args["num_layers"],
        dropout=args["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model
