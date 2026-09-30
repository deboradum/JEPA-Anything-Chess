import copy
import time
import torch
import wandb
import argparse
from tqdm import tqdm  # type: ignore
from pprint import pprint
import torch.optim as optim
from typing import DefaultDict
from collections import defaultdict
from torch.utils.data import DataLoader

from jepa_anything_core.losses import jepa_anything_objective

from model import ChessOPFModel
from config import TrainConfig, load_config
from utils import get_net, get_optimizer, get_dataloaders

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

def evaluate(
    net: ChessOPFModel,
    conf: TrainConfig,
    loader: DataLoader,
):
    val_metrics_sums: DefaultDict[str, float] = defaultdict(float)
    total_samples = 0

    net.eval()
    with torch.no_grad():
        for batch_idx, (context_tokens, target_tokens) in tqdm(
            enumerate(loader),
            total=len(loader),
            desc="Evaluating",
            leave=False,
        ):
            context_tokens, target_tokens = context_tokens.to(device), target_tokens.to(device)
            bs = context_tokens.shape[0]

            context_state, pred_factors, target_state, target_factors = net(
                context_tokens, target_tokens
            )

            objective = jepa_anything_objective(
                predicted_factors=pred_factors,
                target_factors=target_factors,
                analysis_basis=net.projection.analysis_basis(),
                context_states=context_state,
                valid_context_mask=None,
                orthogonality_weight=conf.orthogonality_weight,
                factor_activity_weight=conf.factor_activity_weight,
                encoder_variance_weight=conf.encoder_variance_weight,
                factor_min_std=0.1,
                encoder_min_std=0.1,
                eps=1e-6,
            )

            val_metrics_sums["prediction_loss"] += objective.prediction.item() * bs
            val_metrics_sums["encoder_variance_loss"] += objective.encoder_variance.item() * bs
            val_metrics_sums["factor_activity_loss"] += objective.factor_activity.item() * bs
            val_metrics_sums["orthogonality_loss"] += objective.orthogonality.item() * bs
            val_metrics_sums["total_loss"] += objective.total.item() * bs
            # TODO: Calculate other metrics like accuracy or something like that

            total_samples += bs

    final_metrics_avg = {}
    if total_samples > 0:
        for key, total_sum in val_metrics_sums.items():
            final_metrics_avg[key] = total_sum / total_samples

    return final_metrics_avg

def train(
    net: ChessOPFModel,
    optimizer: optim.Optimizer,
    conf: TrainConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
):
    best_state_dict = copy.deepcopy(net.state_dict())
    early_stop_counter = 0
    global_step = 0
    num_train_batches = len(train_loader)

    # Evaluate before training
    start = time.perf_counter()
    net.eval()
    val_metrics = evaluate(net, conf, test_loader)
    net.train()
    taken = time.perf_counter() - start
    wandb.log({
        "epoch": 0,
        "val/total_loss": val_metrics.get("total_loss", 0.0),
        "val/prediction_loss": val_metrics.get("prediction_loss", 0.0),
        "val/encoder_variance_loss": val_metrics.get("encoder_variance_loss", 0.0),
        "val/factor_activity_loss": val_metrics.get("factor_activity_loss", 0.0),
        "val/orthogonality_loss": val_metrics.get("orthogonality_loss", 0.0),
    })
    print(
        f"[Eval] Epoch 0 (Initial), Time: {taken:.2f}s\n"
        f"  Total Loss:       {val_metrics.get('total_loss', 0.0):.4f}\n"
        f"  Prediction Loss:  {val_metrics.get('prediction_loss', 0.0):.4f}\n"
        f"  Enc Variance:     {val_metrics.get('encoder_variance_loss', 0.0):.4f}\n"
        f"  Factor Activity:  {val_metrics.get('factor_activity_loss', 0.0):.4f}\n"
        f"  Orthogonality:    {val_metrics.get('orthogonality_loss', 0.0):.4f}\n"
    )

    start = time.perf_counter()
    for epoch in range(conf.epochs):
        running_metrics_sums: DefaultDict[str, float] = defaultdict(float)
        samples_since_log = 0
        total_grad_norm_before = 0.
        total_grad_norm_after = 0.

        net.train()
        for batch_idx, (context_tokens, target_tokens) in enumerate(train_loader):
            context_tokens, target_tokens = context_tokens.to(device), target_tokens.to(device)
            bs = context_tokens.shape[0]

            samples_since_log += bs

            optimizer.zero_grad()

            context_state, pred_factors, target_state, target_factors = net(
                context_tokens, target_tokens
            )

            objective = jepa_anything_objective(
                predicted_factors=pred_factors,
                target_factors=target_factors,
                analysis_basis=net.projection.analysis_basis(),
                context_states=context_state,
                valid_context_mask=None,
                orthogonality_weight=conf.orthogonality_weight,
                factor_activity_weight=conf.factor_activity_weight,
                encoder_variance_weight=conf.encoder_variance_weight,
                factor_min_std=0.1,
                encoder_min_std=0.1,
                eps=1e-6,
            )
            total_loss = objective.total

            total_loss.backward()

            running_metrics_sums["prediction_loss"] += objective.prediction.item() * bs
            running_metrics_sums["encoder_variance_loss"] += objective.encoder_variance.item() * bs
            running_metrics_sums["factor_activity_loss"] += objective.factor_activity.item() * bs
            running_metrics_sums["orthogonality_loss"] += objective.orthogonality.item() * bs
            running_metrics_sums["total_loss"] += objective.total.item() * bs
            # TODO: Calculate other metrics like accuracy or something like that

            if conf.gradient_clipping_norm != 0.0:
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), conf.gradient_clipping_norm
                )
                grad_norm_after = torch.sqrt(sum(p.grad.norm()**2 for p in net.parameters() if p.grad is not None)) # type: ignore
                total_grad_norm_before += grad_norm_before.item()
                total_grad_norm_after += grad_norm_after.item()

            optimizer.step()
            net.update_target_encoder()

            global_step += bs
            if (batch_idx+1) % conf.log_interval == 0:
                taken = time.perf_counter() - start
                ips = samples_since_log / taken

                train_metrics = {}
                for key, sum_of_avgs in running_metrics_sums.items():
                    train_metrics[key] = sum_of_avgs / samples_since_log

                wandb.log({
                    "epoch": epoch,
                    "batch": batch_idx+1,
                    "examples": global_step,
                    "train/iterations_per_second": ips,
                    "train/total_loss": train_metrics.get("total_loss", 0.0),
                    "train/prediction_loss": train_metrics.get("prediction_loss", 0.0),
                    "train/encoder_variance_loss": train_metrics.get("encoder_variance_loss", 0.0),
                    "train/factor_activity_loss": train_metrics.get("factor_activity_loss", 0.0),
                    "train/orthogonality_loss": train_metrics.get("orthogonality_loss", 0.0),
                    "train/grad_norm_before": total_grad_norm_before / conf.log_interval,
                    "train/grad_norm_after": total_grad_norm_after / conf.log_interval,
                })

                print(
                    f"Epoch {epoch + 1}, Step {batch_idx+1}/{num_train_batches} (Global {global_step}), Time: {taken:.2f}s ({ips:.2f} samples/s)\n"
                    f"  Total Loss:       {train_metrics.get('total_loss', 0.0):.4f}\n"
                    f"  Prediction Loss:  {train_metrics.get('prediction_loss', 0.0):.4f}\n"
                    f"  Enc Variance:     {train_metrics.get('encoder_variance_loss', 0.0):.4f}\n"
                    f"  Factor Activity:  {train_metrics.get('factor_activity_loss', 0.0):.4f}\n"
                    f"  Orthogonality:    {train_metrics.get('orthogonality_loss', 0.0):.4f}"
                )

                running_metrics_sums = defaultdict(float)
                samples_since_log = 0
                total_grad_norm_before = 0.
                total_grad_norm_after = 0.
                start = time.perf_counter()

        # Evaluate
        start = time.perf_counter()
        net.eval()
        val_metrics = evaluate(net, conf, test_loader)
        net.train()
        taken = time.perf_counter() - start
        wandb.log({
            "epoch": epoch + 1,
            "val/total_loss": val_metrics.get("total_loss", 0.0),
            "val/prediction_loss": val_metrics.get("prediction_loss", 0.0),
            "val/encoder_variance_loss": val_metrics.get("encoder_variance_loss", 0.0),
            "val/factor_activity_loss": val_metrics.get("factor_activity_loss", 0.0),
            "val/orthogonality_loss": val_metrics.get("orthogonality_loss", 0.0),
            "examples": global_step,
        })
        print(
            f"\n[Eval] Epoch {epoch + 1}/{conf.epochs}, Time: {taken:.2f}s\n"
            f"  Total Loss:       {val_metrics.get('total_loss', 0.0):.4f}\n"
            f"  Prediction Loss:  {val_metrics.get('prediction_loss', 0.0):.4f}\n"
            f"  Enc Variance:     {val_metrics.get('encoder_variance_loss', 0.0):.4f}\n"
            f"  Factor Activity:  {val_metrics.get('factor_activity_loss', 0.0):.4f}\n"
            f"  Orthogonality:    {val_metrics.get('orthogonality_loss', 0.0):.4f}\n"
        )

    return val_metrics

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to train config", required=True
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile model before training"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    conf = load_config(args.config)
    config_dict = {**vars(conf)}

    torch.manual_seed(conf.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Training model with config:", end=" ")
    pprint(conf)

    print("Setting up model")
    net = get_net(conf, device)

    num_params = sum(p.numel() for p in net.parameters()) # type: ignore
    print(f"Model parameters {num_params:,}")
    config_dict["num_params"] = num_params

    print("Setting up optimizer")
    optimizer = get_optimizer(net, conf)

    print("Setting up dataloaders")
    train_loader, test_loader = get_dataloaders(conf)

    wandb.init(
        project="chess-jepa-anything",
        name=conf.run_name,
        config=config_dict,
        tags=[],
        settings=wandb.Settings(x_disable_stats=True),
    )

    print("Starting training on device:", device)
    test_metrics = train(
        net=net,
        optimizer=optimizer,
        conf=conf,
        train_loader=train_loader,
        test_loader=test_loader,
    )
