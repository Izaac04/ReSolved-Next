import argparse
import os
import datetime
import torch
import torch.multiprocessing as mp

from gflownet.config import Config, init_empty

from .trainer import MyFragmentsResolveTrainer


"""
Example run command:
python -m gflownet.tasks.resolve_tasks.run_my_fragments_gfn \
  --fragments-csv gflownet/tasks/resolve_tasks/unique_fragments.csv \
  --checkpoint gflownet/tasks/resolve_tasks/best_model.pth \
  --dielectric 78.4 \
  --refractive 1.333 \
  --target-value 3.8 \
  --steps 50000
"""


def main():
    print(">>> LOADING MyFragmentsResolveTask FROM:", __file__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--fragments-csv", type=str, default="unique_fragments.csv")
    ap.add_argument("--checkpoint", type=str, default="best_model.pth")
    ap.add_argument("--log-dir", type=str, default=None)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument(
        "--dielectric",
        type=float,
        default=78.4,
        help="Solvent dielectric constant (e.g. water=78.4, MeCN≈37.5)",
    )

    ap.add_argument(
        "--refractive",
        type=float,
        default=1.333,
        help="Solvent refractive index (e.g. water=1.333)",
    )

    ap.add_argument(
        "--target-value",
        type=float,
        required=True,
        help="Target property value for Gaussian reward"
    )
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    cfg = init_empty(Config())

    cfg.log_dir = args.log_dir or f"./logs/resolve_my_fragments_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    cfg.overwrite_existing_exp = True
    cfg.print_every = 50
    cfg.validate_every = 100
    cfg.num_training_steps = int(args.steps)
    cfg.num_final_gen_steps = 512
    cfg.num_validation_gen_steps = 64
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    mp.set_start_method("spawn", force=True)

    # Temperature conditioning (same style as the paper)
    cfg.cond.temperature.sample_dist = "uniform"
    cfg.cond.temperature.dist_params = [0.0, 64.0]
    print("Training started.")

    trainer = MyFragmentsResolveTrainer(
        cfg=cfg,
        fragments_csv=args.fragments_csv,
        checkpoint_path=args.checkpoint,
        dielectric=args.dielectric,
        refractive=args.refractive,
        target_value=args.target_value,
    )
    trainer.run()


if __name__ == "__main__":
    main()