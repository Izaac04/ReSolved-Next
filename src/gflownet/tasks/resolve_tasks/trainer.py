import socket

from gflownet.config import Config
from gflownet.online_trainer import StandardOnlineTrainer

from gflownet.tasks.resolve_tasks.task import MyFragmentsResolveTask
from .utils import load_my_fragments


class MyFragmentsResolveTrainer(StandardOnlineTrainer):
    """
    Trainer for fragment-based ReSolve GFlowNet experiments.
    Configures training hyperparameters and binds the custom fragment task.
    """

    def __init__(self, cfg: Config, fragments_csv: str, checkpoint_path: str, dielectric: float,
                 refractive: float, target_value: float):
        # Experiment inputs
        self._fragments_csv = fragments_csv
        self._checkpoint_path = checkpoint_path
        self._reward_mode = "gaussian"

        # Regression reward parameters
        self.target_index = 0
        self.target_value = target_value
        self.reward_sigma = 0.15
        self.reward_scale = 1.0

        # store solvent
        self.dielectric = dielectric
        self.refractive = refractive

        super().__init__(cfg)

    def set_default_hps(self, cfg: Config):
        """Set default hyperparameters for GFlowNet training."""
        cfg.hostname = socket.gethostname()

        # Parallelism
        cfg.num_workers = 0

        # Optimisation
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.clip_grad_param = 10.0

        # GFlowNet algorithm
        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 12
        cfg.algo.sampling_tau = 0.99
        cfg.algo.illegal_action_logreward = -75

        # Policy network (NOT your proxy)
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

    def setup_task(self):
        """Load custom fragments and initialize the fragment-based ReSolve task."""
        frags = load_my_fragments(self._fragments_csv)

        self.task = MyFragmentsResolveTask(
            cfg=self.cfg,
            fragments=frags,
            checkpoint_path=self._checkpoint_path,
            wrap_model=self._wrap_for_mp,
            dielectric=self.dielectric,
            refractive=self.refractive,
            target_value=self.target_value,
        )

        print(
            f"[DEBUG] Gaussian reward → "
            f"target_value={self.target_value}, "
            f"sigma={self.reward_sigma}"
        )

    def setup_env_context(self):
        """Environment context is created directly by the task."""
        pass

    def setup(self):
        """Finalize trainer setup and start the GFlowNet training loop."""
        self.setup_task()
        self.ctx = self.task.make_env_context()
        super().setup()

        print(f"TRAINING STARTED with YOUR {len(self.task.fragments)} custom fragments!")
        print(f"[Info] Reward proxy checkpoint: {self._checkpoint_path}")
        print(f"[Info] Reward mode: {self._reward_mode}")
