import gc
import pathlib
import socket
import time

import torch

from gflownet.config import Config
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import cycle
from gflownet.utils.misc import create_logger

from gflownet.tasks.resolve_tasks.task import MyFragmentsResolveTask
from .utils import load_my_fragments


class MyFragmentsResolveTrainer(StandardOnlineTrainer):
    """
    Trainer for fragment-based ReSolve GFlowNet experiments.
    Configures training hyperparameters and binds the custom fragment task.
    """

    def __init__(
            self,
            cfg: Config,
            fragments_csv: str,
            checkpoint_path: str,
            dielectric: float,
            refractive: float,
            target_value: float,
            best_checkpoint_metric: str = "sampled_reward_avg",
            best_checkpoint_mode: str = "auto",
    ):
        # Experiment inputs
        self._fragments_csv = fragments_csv
        self._checkpoint_path = checkpoint_path
        self._reward_mode = "gaussian"
        self.best_checkpoint_metric = best_checkpoint_metric
        self.best_checkpoint_mode = self._resolve_metric_mode(best_checkpoint_metric, best_checkpoint_mode)
        self.best_checkpoint_value = None
        self.best_checkpoint_step = None

        # Regression reward parameters
        self.target_index = 0
        self.target_value = target_value
        self.reward_sigma = 0.15
        self.reward_scale = 1.0

        # store solvent
        self.dielectric = dielectric
        self.refractive = refractive

        super().__init__(cfg)

    @staticmethod
    def _resolve_metric_mode(metric_name: str, metric_mode: str) -> str:
        if metric_mode in {"min", "max"}:
            return metric_mode

        metric_name = metric_name.lower()
        maximize_markers = ("reward", "acc", "auc", "success", "precision", "recall", "f1")
        minimize_markers = ("loss", "error", "mae", "mse", "rmse", "nll")

        if any(marker in metric_name for marker in maximize_markers):
            return "max"
        if any(marker in metric_name for marker in minimize_markers):
            return "min"
        return "max"

    def _is_better_checkpoint(self, candidate_value: float) -> bool:
        if self.best_checkpoint_value is None:
            return True
        if self.best_checkpoint_mode == "max":
            return candidate_value > self.best_checkpoint_value
        return candidate_value < self.best_checkpoint_value

    def _save_best_state(self, it: int, metric_value: float):
        state = {
            "models_state_dict": [self.model.state_dict()],
            "cfg": self.cfg,
            "step": it,
            "best_checkpoint_metric": self.best_checkpoint_metric,
            "best_checkpoint_mode": self.best_checkpoint_mode,
            "best_checkpoint_value": metric_value,
        }
        if self.sampling_model is not self.model:
            state["sampling_model_state_dict"] = [self.sampling_model.state_dict()]

        fn = pathlib.Path(self.cfg.log_dir) / "best_model_state.pt"
        with open(fn, "wb") as fd:
            torch.save(state, fd)

    @staticmethod
    def _mean_validation_info(infos: list[dict]) -> dict:
        if not infos:
            return {}

        summary = {}
        keys = set().union(*(info.keys() for info in infos))
        for key in keys:
            values = [info[key] for info in infos if isinstance(info.get(key), (int, float))]
            if values:
                summary[key] = float(sum(values) / len(values))
        return summary

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
        print(
            f"[Info] Best checkpoint metric: {self.best_checkpoint_metric} "
            f"({self.best_checkpoint_mode})"
        )

    def run(self, logger=None):
        """Train ReSolved-Next and save a best checkpoint using a configurable validation metric."""
        if logger is None:
            logger = create_logger(logfile=self.cfg.log_dir + "/train.log")
        self.model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 1)
        valid_freq = self.cfg.validate_every
        ckpt_freq = self.cfg.checkpoint_every if self.cfg.checkpoint_every is not None else valid_freq
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()
        if self.cfg.num_final_gen_steps:
            final_dl = self.build_final_data_loader()
        callbacks = self.build_callbacks()
        start = self.cfg.start_at_step + 1
        num_training_steps = self.cfg.num_training_steps
        logger.info("Starting training")
        start_time = time.time()

        for it, batch in zip(range(start, 1 + num_training_steps), cycle(train_dl)):
            if it % 1024 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            if self.replay_buffer is not None and len(self.replay_buffer) < self.replay_buffer.warmup:
                logger.info(
                    f"iteration {it} : warming up replay buffer {len(self.replay_buffer)}/{self.replay_buffer.warmup}"
                )
                continue

            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx, it)
            info["time_spent"] = time.time() - start_time
            start_time = time.time()
            self.log(info, it, "train")
            if it % self.print_every == 0:
                logger.info(f"iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))

            if valid_freq > 0 and it % valid_freq == 0:
                valid_infos = []
                for valid_batch in valid_dl:
                    valid_info = self.evaluate_batch(valid_batch.to(self.device), epoch_idx, batch_idx)
                    valid_infos.append(valid_info)
                    self.log(valid_info, it, "valid")
                    logger.info(
                        f"validation - iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in valid_info.items())
                    )

                valid_summary = self._mean_validation_info(valid_infos)
                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, "on_validation_end"):
                        c.on_validation_end(end_metrics)
                self.log(end_metrics, it, "valid_end")

                metric_value = valid_summary.get(self.best_checkpoint_metric)
                if metric_value is None:
                    logger.warning(
                        f"Best-checkpoint metric '{self.best_checkpoint_metric}' was not found in validation metrics."
                    )
                elif self._is_better_checkpoint(metric_value):
                    self.best_checkpoint_value = metric_value
                    self.best_checkpoint_step = it
                    self._save_best_state(it, metric_value)
                    logger.info(
                        f"Saved best checkpoint at iteration {it} "
                        f"with {self.best_checkpoint_metric}={metric_value:.4f}"
                    )

            if ckpt_freq > 0 and it % ckpt_freq == 0:
                self._save_state(it)

        self._save_state(num_training_steps)

        num_final_gen_steps = self.cfg.num_final_gen_steps
        final_info = {}
        if num_final_gen_steps:
            logger.info(f"Generating final {num_final_gen_steps} batches ...")
            for it, batch in zip(
                    range(num_training_steps + 1, num_training_steps + num_final_gen_steps + 1),
                    cycle(final_dl),
            ):
                if hasattr(batch, "extra_info"):
                    for k, v in batch.extra_info.items():
                        if k not in final_info:
                            final_info[k] = []
                        if hasattr(v, "item"):
                            v = v.item()
                        final_info[k].append(v)
                if it % self.print_every == 0:
                    logger.info(f"Generating objs {it - num_training_steps}/{num_final_gen_steps}")
            final_info = {k: sum(v) / len(v) for k, v in final_info.items()}

            logger.info("Final generation steps completed - " + " ".join(f"{k}:{v:.2f}" for k, v in final_info.items()))
            self.log(final_info, num_training_steps, "final")

        del train_dl
        del valid_dl
        if self.cfg.num_final_gen_steps:
            del final_dl