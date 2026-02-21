import torch

from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Batch

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.tasks.synth_gnn.load import load_synth_proxy
from gflownet.tasks.synth_gnn.featurise import smiles_to_data
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward
from gflownet.utils.misc import get_worker_device

from .resolve_featurise import mol_to_graph
from .utils import tanimoto_dist, internal_diversity, load_resolve_proxy, load_my_fragments, remove_dummy_atoms


class MyFragmentsResolveTask(GFNTask):
    """
    GFlowNet task for fragment-based molecule generation with multi-objective rewards.
    Combines property optimisation (ResolveGNN), diversity incentives, and
    synthesizability estimation (SynthProxyGNN).
    """

    def __init__(
            self,
            cfg,
            fragments,
            checkpoint_path: str,
            target_value: float,
            wrap_model=None,
            clip_min: float = 1e-4,
            clip_max: float = 100.0,
            synth_checkpoint_path: str | None = None,
            dielectric: float = 78.4,
            refractive: float = 1.333,
    ):
        """
        Initialise the fragment-based ReSolve task.
        Sets up conditioning, reward weights, diversity tracking, and loads
        frozen property and synthesizability proxy models.
        """
        self.cfg = cfg
        self._wrap_model = wrap_model or (lambda x: x)

        # Conditioning
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        self.fragments = fragments

        # Gaussian reward config
        self.target_value = float(target_value)
        self.reward_sigma = 0.5

        # Reward config
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Multi-objective weights
        self.w_prop = 1.0
        self.w_div = 0.3
        self.w_syn = 0.7

        # Diversity params
        # Note: internal diversity is batch-level; scaffold novelty is per-molecule.
        self.div_alpha = 0.5  # scaffold novelty weight
        self.div_beta = 0.5  # internal diversity weight
        self.seen_scaffolds: set[str] = set()  # populate from training set elsewhere if you want true "novel vs train"
        self.train_fps = []  # store train fingerprints for novelty-vs-train molecule bonus

        # Synthesizability params
        self.syn_success_bonus = 1.0
        self.syn_route_penalty = 0.15

        # Solvent (explicit conditioning)
        self.dielectric = float(dielectric)
        self.refractive = float(refractive)

        print(
            f"[ResolveTask] Solvent conditioning: "
            f"ε={self.dielectric}, n={self.refractive}"
        )

        # ---- Load proxies (frozen) ----
        dev = get_worker_device()
        print(f"[Info] Proxy device: {dev} (env/trainer/dataloader may stay on CPU)")

        # Property proxy (ResolveGNN)
        self.proxy = load_resolve_proxy(
            checkpoint_path=checkpoint_path,
            device=dev,
            wrap_model=self._wrap_model,
        )
        self.proxy.eval()

        # Synthesizability proxy (SynthProxyGNN)
        # Your load.py signature is: load_synth_proxy(device, checkpoint_path=_DEFAULT_CKPT)
        if synth_checkpoint_path is None:
            self.synth_proxy = load_synth_proxy(device=dev)  # uses synth_gnn/proxy_best.pt by default
            print("[Info] Using default synth proxy checkpoint (synth_gnn/proxy_best.pt)")
        else:
            self.synth_proxy = load_synth_proxy(device=dev, checkpoint_path=Path(synth_checkpoint_path))
            print(f"[Info] Using synth proxy checkpoint: {synth_checkpoint_path}")
        self.synth_proxy.eval()

        # (Optional) keep a handle for MurckoScaffold if you want, not required
        self._murcko = MurckoScaffold

    def sample_conditional_information(self, n: int, train_it: int):
        """
        Sample conditional information for temperature-based conditioning.
        Used to control exploration during GFlowNet training.
        """
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info, flat_reward: ObjectProperties):
        """
        Transform scalar rewards into conditional log-rewards
        using temperature-based conditioning.
        """
        return LogScalar(
            self.temperature_conditional.transform(cond_info, to_logreward(flat_reward))
        )

    def compute_obj_properties(self, mols):
        """
        Compute per-molecule rewards from generated molecules.
        Evaluates property quality, diversity, and synthesizability,
        then combines them into a clipped final reward.
        """

        # 0. Basic validity check
        cleaned_mols = mols
        valid_after_cleanup = torch.tensor(
            [m is not None for m in mols],
            dtype=torch.bool
        )

        if not valid_after_cleanup.any():
            return ObjectProperties(torch.zeros((len(mols), 1))), valid_after_cleanup

        # 1. Valid molecule filtering
        graphs = [mol_to_graph(m) if m is not None else None for m in cleaned_mols]
        graph_valid = torch.tensor([g is not None for g in graphs], dtype=torch.bool)

        # FINAL validity = survived dummy removal AND graph construction
        is_valid = valid_after_cleanup & graph_valid

        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid

        valid_graphs = [g for g, v in zip(graphs, is_valid) if v]
        valid_mols = [m for m, v in zip(cleaned_mols, is_valid) if v]

        device = self.proxy.device

        # Length Prior
        traj_lens = torch.tensor(
            [m.GetNumAtoms() for m in valid_mols],
            device=device,
            dtype=torch.float,
        )

        # 2. Property: ResolveGNN
        batch = Batch.from_data_list(valid_graphs).to(device)

        with torch.no_grad():
            num_graphs = batch.num_graphs

            dielec = torch.full(
                (num_graphs,),
                self.dielectric,
                device=device,
            )

            ref = torch.full(
                (num_graphs,),
                self.refractive,
                device=device,
            )

            out = self.proxy(batch, dielec, ref)

            if isinstance(out, (tuple, list)):
                pred = out[0]
            elif isinstance(out, dict):
                for k in ["pred", "y", "output", "redox"]:
                    if k in out:
                        pred = out[k]
                        break
                else:
                    raise RuntimeError("Could not find regression output in proxy dict.")
            else:
                pred = out

            value = pred.squeeze()

            if torch.rand(1).item() < 0.01:
                print(
                    f"[Resolve] mean={value.mean():.3f} "
                    f"(eps={self.dielectric}, ref={self.refractive})"
                )

            # Map regression → reward
            P = torch.exp(
                -((value - self.target_value) ** 2)
                / (2 * self.reward_sigma ** 2)
            )

        # 3. Diversity
        # Fingerprints
        fps = [Chem.RDKFingerprint(m) for m in valid_mols]

        # (a) Internal diversity (batch-level)
        if len(fps) > 1:
            dists = []
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    dists.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            D_internal = torch.full(
                (len(valid_mols),),
                float(sum(dists) / len(dists)),
                device=device,
            )
        else:
            D_internal = torch.zeros(len(valid_mols), device=device)

        # (b) Scaffold novelty
        scaffolds = []
        for m in valid_mols:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(m)
            if scaffold_mol is None:
                scaffolds.append("")
            else:
                scaffolds.append(Chem.MolToSmiles(scaffold_mol))

        D_scaffold = torch.tensor(
            [1.0 if s not in self.seen_scaffolds else 0.0 for s in scaffolds],
            device=device,
        )

        # Combine diversity
        D = self.div_alpha * D_scaffold + self.div_beta * D_internal

        # 4. Synthesizability
        synth_graphs = []
        for m in valid_mols:
            smi = Chem.MolToSmiles(m)
            g = smiles_to_data(smi)
            if g is None:
                g = smiles_to_data("")  # fallback dummy (won't crash batch)
            synth_graphs.append(g)

        synth_batch = Batch.from_data_list(synth_graphs).to(self.proxy.device)

        with torch.no_grad():
            success_logit, route_len = self.synth_proxy(synth_batch)

        success_prob = torch.sigmoid(success_logit).squeeze()
        route_len = route_len.squeeze()

        S = (
                self.syn_success_bonus * success_prob
                - self.syn_route_penalty * route_len
        )

        L_target = 18.0
        L_sigma = 6.0

        L_prior = torch.exp(
            -((traj_lens - L_target) ** 2)
            / (2 * L_sigma ** 2)
        )

        # 5. Final Reward
        R = (
                self.w_prop * P
                + self.w_div * D
                + self.w_syn * S
        )

        R = R * L_prior

        R = R.clip(self.clip_min, self.clip_max)

        # 6. Pack return
        n_valid = int(is_valid.sum().item())

        if R.shape[0] != n_valid:
            raise RuntimeError(
                f"[compute_obj_properties] shape mismatch: "
                f"R has {R.shape[0]} rows but is_valid.sum() = {n_valid}"
            )

        if R.shape[0] != int(is_valid.sum().item()):
            raise RuntimeError(
                f"[compute_obj_properties] mismatch: "
                f"R={R.shape[0]} vs is_valid.sum()={int(is_valid.sum())}"
            )

        # 7. Debug
        if torch.rand(1).item() < 0.01:
            print(
                f"[Reward] P={P.mean():.3f} | "
                f"D={D.mean():.3f} | "
                f"S={S.mean():.3f} | "
                f"R={R.mean():.3f}"
            )

        return ObjectProperties(R.unsqueeze(1)), is_valid

    def make_env_context(self):
        """
        Create the fragment-based molecule-building environment context.
        Defines fragment vocabulary, conditioning dimensions, and size limits.
        """
        return FragMolBuildingEnvContext(
            fragments=self.fragments,
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.num_cond_dim,
        )
