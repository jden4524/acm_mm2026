import argparse
import re
from pathlib import Path

import pandas as pd
import wandb
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log all CSV metrics from eval_results/<run_name> to Weights & Biases."
    )
    parser.add_argument(
        "checkpoint_name",
        help="Folder name under eval_results (XXX in ./eval_results/XXX).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "eval_results",
        help="Root folder that contains evaluation run folders.",
    )
    parser.add_argument(
        "--project",
        default="attn_ft",
        help="Weights & Biases project name.",
    )
    return parser.parse_args()


def log_csv_file(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    csv_fn = csv_path.stem
    
    if "HallusionBench" in csv_fn:
        df = df.set_index("split")
        return {"HallusionBench": df.loc["Overall"].mean()}
    if "MME" in csv_fn:
        return {"MME": df["perception"][0]}
    if "MMVP" in csv_fn:
        return {"MMVP": df["Overall"][0]}
    if "VisOnlyQA" in csv_fn:
        df = df.set_index("split")
        return {"VisOnlyQA_Real": df.loc["Eval_Real","Overall"],
                "VisOnlyQA_Synthetic": df.loc["Eval_Synthetic","Overall"]}
    if "VStarBench" in csv_fn:
        return {"VStarBench": df["Overall"][0]}
    if "POPE" in csv_fn:
        df = df.set_index("split")
        return {"POPE": df.loc["Overall","Overall"]}

    return {}


def log_eval_results(checkpoint_name,
                     run_name=None, 
                     results_root: Path=Path(__file__).resolve().parent / "eval_results", 
                     project: str="attn_ft") -> None:
    if run_name is None:
        run_name = "_".join(checkpoint_name.split("_")[:-1])
    target_dir = results_root / checkpoint_name

    if not target_dir.exists() or not target_dir.is_dir():
        raise FileNotFoundError(f"Evaluation folder not found: {target_dir}")

    csv_files = sorted(target_dir.rglob("*.csv"))
    # get run id from json if exists, else create new run
    wandb_run_id_path = results_root / "wandb_run_id.json"
    if wandb_run_id_path.exists():
        with open(wandb_run_id_path, "r") as f:
            run_id_dict = json.load(f)
        run_id = run_id_dict.get(run_name)
    else:
        run_id = None
    wandb.init(
        project=project,
        name=run_name,
        id=run_id,
        resume="allow",
        job_type="eval",
    )
    metrics_to_log = {}
    for csv_file in csv_files:
        metrics_to_log.update(log_csv_file(csv_file))

    step = checkpoint_name.split("_")[-1]
    wandb.log(metrics_to_log, step=int(step))
    run_id = wandb.run.id
    wandb.finish()
    print(f"Logged {len(csv_files)} CSV files from {target_dir} to wandb")
    # load and update json for future reference

    if wandb_run_id_path.exists():
        run_id_dict = json.load(open(wandb_run_id_path))
    else:
        run_id_dict = {}
    run_id_dict[run_name] = run_id
    json.dump(run_id_dict, open(wandb_run_id_path, "w"), indent=2)


if __name__ == "__main__":
    args = parse_args()
    log_eval_results(
        checkpoint_name=args.checkpoint_name,
        results_root=args.results_root,
        project=args.project
    )
