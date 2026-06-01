# extra_logging_utils.py
# ============================================================
# Diagnostic logging utilities for SAE feature-recovery experiments.
#
# Purpose:
#   Track why mixed-reconstruction SAEs fail to recover target directions
#   even when target directions are present and decoder gradients may exist.
#
# Expected cached assets:
#   H_bg_z.pt
#   weak_dirs_z_orth.pt
#   strong_dirs_z_orth.pt
#   random_dirs_orth.pt
#   nuisance_dirs_z_orth.pt
#
# Training conditions:
#   1. Standard L1 SAE on mixed activations
#   2. TopK SAE on mixed activations
#   3. Delta SAE on paired deltas
#   4. Random-pair SAE on random mixed differences
#
# Logged quantities:
#   - decoder-gradient projection onto target directions
#   - target-gradient / total-gradient ratio
#   - best latent index over training
#   - decoder cosine allocation
#   - target residual explained
#   - latent firing selectivity
# ============================================================

import argparse
from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--asset-dir", type=str, required=True,
                   help="Layer asset directory, e.g. results/main_comparison/layer_8")
    p.add_argument("--save-dir", type=str, default="results/diagnostics")

    p.add_argument("--group", type=str, default="weak",
                   choices=["weak", "strong", "random"])
    p.add_argument("--layer", type=int, default=8)

    p.add_argument("--scale", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sae-latents", type=int, default=512)
    p.add_argument("--topk-k", type=int, default=20)

    p.add_argument("--l1-coef", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=512)

    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--n-probe", type=int, default=3000)

    p.add_argument("--log-every", type=int, default=50)

    # Must match run_comparison.py.
    p.add_argument("--common-nuisance-scale", type=float, default=0.8)
    p.add_argument("--common-nuisance-prob", type=float, default=0.25)
    p.add_argument("--contrast-low", type=float, default=0.8)
    p.add_argument("--contrast-high", type=float, default=1.2)

    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


args = parse_args()

DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
ASSET = Path(args.asset_dir)
SAVE = Path(args.save_dir)
SAVE.mkdir(parents=True, exist_ok=True)

with open(SAVE / "diagnostic_config.json", "w") as f:
    json.dump(vars(args), f, indent=2)

print("Device:", DEVICE)
print("Asset dir:", ASSET)
print("Save dir:", SAVE)


# ============================================================
# LOAD ASSETS
# ============================================================

def normalize_rows(x, eps=1e-8):
    return x / x.norm(dim=1, keepdim=True).clamp(min=eps)


def load_assets():
    h_path = ASSET / "H_bg_z.pt"
    nuisance_path = ASSET / "nuisance_dirs_z_orth.pt"

    if args.group == "weak":
        dirs_path = ASSET / "weak_dirs_z_orth.pt"
        meta_path = ASSET / "weak_meta.csv"
    elif args.group == "strong":
        dirs_path = ASSET / "strong_dirs_z_orth.pt"
        meta_path = ASSET / "strong_meta.csv"
    else:
        dirs_path = ASSET / "random_dirs_orth.pt"
        meta_path = ASSET / "random_meta.csv"

    missing = [p for p in [h_path, nuisance_path, dirs_path, meta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(str(p) for p in missing))

    H_bg = torch.load(h_path, map_location="cpu").float()
    nuisance_dirs = torch.load(nuisance_path, map_location="cpu").float()
    target_dirs = torch.load(dirs_path, map_location="cpu").float()
    meta = pd.read_csv(meta_path)

    nuisance_dirs = normalize_rows(nuisance_dirs)
    target_dirs = normalize_rows(target_dirs)

    concept_col = "target" if "target" in meta.columns else "concept"
    concepts = meta[concept_col].astype(str).tolist()

    return H_bg, nuisance_dirs, target_dirs, meta, concepts


H_BG, NUISANCE_DIRS, TARGET_DIRS, META, CONCEPTS = load_assets()
D_MODEL = H_BG.shape[1]
N_DIRS = TARGET_DIRS.shape[0]

print("H_bg:", tuple(H_BG.shape))
print("nuisance_dirs:", tuple(NUISANCE_DIRS.shape))
print("target_dirs:", tuple(TARGET_DIRS.shape))
print("concepts:", CONCEPTS)


# ============================================================
# DATA GENERATION
# ============================================================

def sample_background(H_bg, n, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randint(0, H_bg.shape[0], (n,), generator=g)
    return H_bg[idx]


def add_nuisance(h, nuisance_dirs, seed):
    torch.manual_seed(seed)

    n, nc = h.shape[0], nuisance_dirs.shape[0]
    mask = (torch.rand(n, nc) < args.common_nuisance_prob).float()
    coeff = mask * (torch.rand(n, nc) * 2 - 1) * args.common_nuisance_scale

    return h + coeff @ nuisance_dirs


def generate_mixed_dataset(H_bg, target_dirs, nuisance_dirs, n, signal_scale, seed):
    """
    Returns:
      X: mixed activation
      S: known injected target signal only
      chosen: concept id for injected direction
      Y: one-hot concept label
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    background = sample_background(H_bg, n, seed)
    background = add_nuisance(background, nuisance_dirs, seed + 100)

    chosen = torch.randint(0, target_dirs.shape[0], (n,))
    coeff = (
        args.contrast_low
        + (args.contrast_high - args.contrast_low) * torch.rand(n)
    ) * signal_scale

    signal = target_dirs[chosen] * coeff[:, None]
    X = background + signal

    Y = torch.zeros(n, target_dirs.shape[0])
    Y[torch.arange(n), chosen] = 1.0

    return X.float(), signal.float(), chosen.long(), Y.float()


def generate_delta_dataset(target_dirs, n, signal_scale, seed):
    """
    Pure paired-delta data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    chosen = torch.randint(0, target_dirs.shape[0], (n,))
    coeff = (
        args.contrast_low
        + (args.contrast_high - args.contrast_low) * torch.rand(n)
    ) * signal_scale

    X = target_dirs[chosen] * coeff[:, None]

    Y = torch.zeros(n, target_dirs.shape[0])
    Y[torch.arange(n), chosen] = 1.0

    return X.float(), X.float(), chosen.long(), Y.float()


def make_random_pair_deltas(X, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    return (X[rng.permutation(n)] - X[rng.permutation(n)]).float()


# ============================================================
# SAE MODELS
# ============================================================

class StandardSAE(nn.Module):
    def __init__(self, input_dim, n_latents):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_latents)
        self.decoder = nn.Linear(n_latents, input_dim, bias=False)

        nn.init.normal_(self.encoder.weight, std=0.02)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, std=0.02)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    def normalize_decoder(self):
        with torch.no_grad():
            W = self.decoder.weight.data
            self.decoder.weight.data = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)


class TopKSAE(nn.Module):
    def __init__(self, input_dim, n_latents, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, n_latents)
        self.decoder = nn.Linear(n_latents, input_dim, bias=False)

        nn.init.normal_(self.encoder.weight, std=0.02)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, std=0.02)

    def forward(self, x):
        z_pre = self.encoder(x)
        topk_v, topk_i = torch.topk(z_pre, self.k, dim=-1)

        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_i, torch.relu(topk_v))

        x_hat = self.decoder(z)
        return x_hat, z

    def normalize_decoder(self):
        with torch.no_grad():
            W = self.decoder.weight.data
            self.decoder.weight.data = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)


# ============================================================
# METRICS
# ============================================================

def decoder_matrix_latents(sae):
    """
    Returns decoder columns as [n_latents, d_model].
    For nn.Linear(n_latents -> d_model), decoder.weight is [d_model, n_latents].
    """
    W = sae.decoder.weight.detach().cpu().float().T
    W = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return W


def decoder_grad_latents(sae):
    """
    Returns decoder gradient columns as [n_latents, d_model].
    """
    G = sae.decoder.weight.grad
    if G is None:
        return None

    G = G.detach().cpu().float().T
    return G


def compute_decoder_allocation(sae, target_dirs):
    """
    Returns:
      cos_matrix: [n_dirs, n_latents]
      max_cos: [n_dirs]
      best_latent: [n_dirs]
    """
    W = decoder_matrix_latents(sae)
    dirs = target_dirs.cpu().float()
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)

    cos = torch.abs(dirs @ W.T)
    max_cos, best_latent = cos.max(dim=1)

    return cos, max_cos, best_latent


def compute_decoder_gradient_stats(sae, target_dirs):
    """
    Computes decoder-gradient projection and target/total gradient ratio.

    For each target direction v:
      max_grad_proj = max_j |grad_decoder_j · v|
      mean_grad_proj = mean_j |grad_decoder_j · v|
      max_grad_norm = max_j ||grad_decoder_j||
      mean_grad_norm = mean_j ||grad_decoder_j||
      target_total_grad_ratio = max_grad_proj / mean_grad_norm
    """
    G = decoder_grad_latents(sae)
    if G is None:
        return None

    dirs = target_dirs.cpu().float()
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)

    grad_norms = G.norm(dim=1).clamp(min=1e-12)

    proj = torch.abs(dirs @ G.T)  # [n_dirs, n_latents]

    max_proj, best_grad_latent = proj.max(dim=1)
    mean_proj = proj.mean(dim=1)

    max_grad_norm = grad_norms.max().item()
    mean_grad_norm = grad_norms.mean().item()

    rows = []
    for ci in range(dirs.shape[0]):
        rows.append({
            "concept_id": ci,
            "max_dec_grad_proj": float(max_proj[ci].item()),
            "mean_dec_grad_proj": float(mean_proj[ci].item()),
            "best_grad_latent": int(best_grad_latent[ci].item()),
            "max_dec_grad_norm": float(max_grad_norm),
            "mean_dec_grad_norm": float(mean_grad_norm),
            "target_total_grad_ratio": float(max_proj[ci].item() / (mean_grad_norm + 1e-12)),
        })

    return pd.DataFrame(rows)


def compute_probe_diagnostics(sae, X_probe, S_probe, chosen_probe, target_dirs):
    """
    Functional diagnostics on a fixed probe set.

    For each concept:
      - best decoder latent
      - max decoder cosine
      - target residual explained
      - latent firing selectivity

    target_residual_explained:
      Uses known injected target signal S_probe.
      For concept c:
        true target projection = S_probe @ v
        residual projection    = (X_probe - x_hat) @ v
        explained = 1 - Var(residual_proj) / Var(true_target_proj)

      This is cleanest for delta data.
      For mixed data, residual includes background reconstruction error projected onto v,
      which is exactly the point: background can swamp the target component.
    """
    sae.eval()

    X = X_probe.to(DEVICE)
    with torch.no_grad():
        x_hat, z = sae(X)

    x_hat = x_hat.detach().cpu()
    z = z.detach().cpu()

    X_cpu = X_probe.cpu()
    S_cpu = S_probe.cpu()
    chosen_cpu = chosen_probe.cpu()

    _, max_cos, best_latent = compute_decoder_allocation(sae, target_dirs)

    rows = []

    for ci in range(target_dirs.shape[0]):
        v = target_dirs[ci].cpu().float()
        v = v / v.norm().clamp(min=1e-8)

        mask = chosen_cpu == ci
        n_pos = int(mask.sum().item())

        j = int(best_latent[ci].item())

        if n_pos >= 2:
            true_proj = S_cpu[mask] @ v
            residual_proj = (X_cpu[mask] - x_hat[mask]) @ v

            true_var = true_proj.var(unbiased=False).item()
            residual_var = residual_proj.var(unbiased=False).item()

            target_explained = 1.0 - residual_var / (true_var + 1e-12)
        else:
            target_explained = np.nan
            true_var = np.nan
            residual_var = np.nan

        z_j = z[:, j]
        if n_pos >= 1 and (~mask).sum().item() >= 1:
            pos_mean = z_j[mask].mean().item()
            neg_mean = z_j[~mask].mean().item()

            # bounded-ish selectivity score
            selectivity = (pos_mean - neg_mean) / (abs(pos_mean) + abs(neg_mean) + 1e-8)
            ratio = pos_mean / (neg_mean + 1e-8)
        else:
            pos_mean = np.nan
            neg_mean = np.nan
            selectivity = np.nan
            ratio = np.nan

        rows.append({
            "concept_id": ci,
            "concept": CONCEPTS[ci] if ci < len(CONCEPTS) else f"concept_{ci}",
            "best_decoder_latent": j,
            "max_dec_cos": float(max_cos[ci].item()),
            "n_positive_probe": n_pos,

            "target_signal_variance": true_var,
            "target_residual_variance": residual_var,
            "target_residual_explained": float(target_explained),

            "best_latent_pos_activation": float(pos_mean),
            "best_latent_neg_activation": float(neg_mean),
            "best_latent_selectivity": float(selectivity),
            "best_latent_pos_neg_ratio": float(ratio),
        })

    return pd.DataFrame(rows)


def finalize_corr_recovery(Y, Z):
    """
    Same recovery metric as the main comparison script:
      for each concept label, max absolute correlation with any latent.
    """
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)

    Yc = Y - Y.mean(axis=0, keepdims=True)
    Zc = Z - Z.mean(axis=0, keepdims=True)

    yn = np.sqrt((Yc ** 2).sum(axis=0)) + 1e-12
    zn = np.sqrt((Zc ** 2).sum(axis=0)) + 1e-12

    corr = np.abs((Yc.T @ Zc) / (yn[:, None] * zn[None, :]))
    recovery = corr.max(axis=1)
    best_latent = corr.argmax(axis=1)

    return recovery, best_latent


def get_codes(sae, X, batch_size=1024):
    sae.eval()

    out = []
    X_dev = X.to(DEVICE)

    with torch.no_grad():
        for s in range(0, X_dev.shape[0], batch_size):
            _, z = sae(X_dev[s:s + batch_size])
            out.append(z.detach().cpu())

    return torch.cat(out, dim=0).numpy()


# ============================================================
# TRAINING WITH DIAGNOSTIC LOGGING
# ============================================================

def train_with_diagnostics(
    condition_name,
    sae,
    X_train,
    X_probe,
    S_probe,
    chosen_probe,
    Y_probe,
    target_dirs,
    l1_coef=1e-3,
    is_topk=False,
):
    print("\n" + "=" * 100)
    print(f"Training condition: {condition_name}")
    print("=" * 100)

    sae = sae.to(DEVICE)
    sae.normalize_decoder()

    opt = optim.Adam(sae.parameters(), lr=args.lr)

    X_train_dev = X_train.to(DEVICE)
    n = X_train_dev.shape[0]

    logs = []

    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch_size,), device=DEVICE)
        batch = X_train_dev[idx]

        x_hat, z = sae(batch)

        recon_loss = ((x_hat - batch) ** 2).mean()

        if is_topk:
            loss = recon_loss
            l1_loss = torch.tensor(0.0, device=DEVICE)
        else:
            l1_loss = z.abs().mean()
            loss = recon_loss + l1_coef * l1_loss

        opt.zero_grad()
        loss.backward()

        if step % args.log_every == 0 or step == args.steps - 1:
            grad_df = compute_decoder_gradient_stats(sae, target_dirs)
            probe_df = compute_probe_diagnostics(
                sae=sae,
                X_probe=X_probe,
                S_probe=S_probe,
                chosen_probe=chosen_probe,
                target_dirs=target_dirs,
            )

            merged = probe_df.merge(grad_df, on="concept_id", how="left")

            merged["condition"] = condition_name
            merged["step"] = step
            merged["loss"] = float(loss.item())
            merged["recon_loss"] = float(recon_loss.item())
            merged["l1_loss"] = float(l1_loss.item())
            merged["l1_coef"] = float(l1_coef)
            merged["group"] = args.group
            merged["layer"] = args.layer
            merged["scale"] = args.scale
            merged["seed"] = args.seed
            merged["sae_latents"] = args.sae_latents

            logs.append(merged)

            mean_alloc = merged["max_dec_cos"].mean()
            mean_select = merged["best_latent_selectivity"].mean()
            mean_expl = merged["target_residual_explained"].mean()
            mean_grad_ratio = merged["target_total_grad_ratio"].mean()

            print(
                f"[{condition_name}] step={step:4d} "
                f"loss={loss.item():.4e} "
                f"alloc={mean_alloc:.3f} "
                f"select={mean_select:.3f} "
                f"target_expl={mean_expl:.3f} "
                f"grad_ratio={mean_grad_ratio:.3e}"
            )

        opt.step()
        sae.normalize_decoder()

    log_df = pd.concat(logs, ignore_index=True)

    # final recovery
    Z_probe = get_codes(sae, X_probe)
    recovery, best_rec_latent = finalize_corr_recovery(Y_probe.numpy(), Z_probe)

    rec_df = pd.DataFrame({
        "condition": condition_name,
        "concept_id": np.arange(len(recovery)),
        "concept": CONCEPTS[:len(recovery)],
        "recovery_score": recovery,
        "best_recovery_latent": best_rec_latent,
        "group": args.group,
        "layer": args.layer,
        "scale": args.scale,
        "seed": args.seed,
        "sae_latents": args.sae_latents,
    })

    return sae, log_df, rec_df


# ============================================================
# PLOTS
# ============================================================

def plot_diagnostic_timeseries(all_logs):
    df = all_logs.copy()

    condition_order = ["std_L1_mixed", "topK_mixed", "paired_delta", "random_pair"]
    colors = {
        "std_L1_mixed": "#D85A30",
        "topK_mixed": "#378ADD",
        "paired_delta": "#1D9E75",
        "random_pair": "#888780",
    }

    labels = {
        "std_L1_mixed": "L1 mixed",
        "topK_mixed": "TopK mixed",
        "paired_delta": "Paired delta",
        "random_pair": "Random-pair",
    }

    step_df = (
        df.groupby(["condition", "step"])
        .agg(
            max_dec_cos=("max_dec_cos", "mean"),
            target_total_grad_ratio=("target_total_grad_ratio", "mean"),
            target_residual_explained=("target_residual_explained", "mean"),
            best_latent_selectivity=("best_latent_selectivity", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    panels = [
        ("max_dec_cos", "Decoder allocation\nmean max cosine"),
        ("target_total_grad_ratio", "Target / total decoder-gradient ratio"),
        ("target_residual_explained", "Target residual explained"),
        ("best_latent_selectivity", "Best-latent firing selectivity"),
    ]

    for ax, (metric, ylabel) in zip(axes, panels):
        for cond in condition_order:
            d = step_df[step_df["condition"] == cond].sort_values("step")
            if d.empty:
                continue

            ax.plot(
                d["step"],
                d[metric],
                lw=2.4,
                color=colors[cond],
                label=labels[cond],
            )

        if metric == "target_total_grad_ratio":
            ax.set_yscale("log")

        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.set_title(metric, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(
        f"Diagnostic logging: why mixed SAE fails\n"
        f"group={args.group}, layer={args.layer}, scale={args.scale}, K={args.sae_latents}",
        fontweight="bold",
    )

    plt.tight_layout()

    out = SAVE / f"diagnostic_timeseries_{args.group}_layer{args.layer}_scale{args.scale}_K{args.sae_latents}.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


def plot_final_diagnostic_bars(all_logs, all_recovery):
    df = all_logs.copy()

    max_step = df["step"].max()
    late = df[df["step"] == max_step].copy()

    summary = (
        late.groupby("condition")
        .agg(
            final_allocation=("max_dec_cos", "mean"),
            final_grad_ratio=("target_total_grad_ratio", "mean"),
            final_target_explained=("target_residual_explained", "mean"),
            final_selectivity=("best_latent_selectivity", "mean"),
        )
        .reset_index()
    )

    rec_summary = (
        all_recovery.groupby("condition")
        .agg(mean_recovery=("recovery_score", "mean"))
        .reset_index()
    )

    summary = summary.merge(rec_summary, on="condition", how="left")

    order = ["std_L1_mixed", "topK_mixed", "paired_delta", "random_pair"]
    summary["order"] = summary["condition"].apply(lambda x: order.index(x) if x in order else 99)
    summary = summary.sort_values("order")

    labels = {
        "std_L1_mixed": "L1 mixed",
        "topK_mixed": "TopK mixed",
        "paired_delta": "Paired delta",
        "random_pair": "Random-pair",
    }

    colors = {
        "std_L1_mixed": "#D85A30",
        "topK_mixed": "#378ADD",
        "paired_delta": "#1D9E75",
        "random_pair": "#888780",
    }

    metrics = [
        ("mean_recovery", "Mean recovery"),
        ("final_allocation", "Final decoder allocation"),
        ("final_target_explained", "Target residual explained"),
        ("final_selectivity", "Latent firing selectivity"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    x = np.arange(len(summary))

    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.bar(
            x,
            summary[metric],
            color=[colors[c] for c in summary["condition"]],
            alpha=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] for c in summary["condition"]], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(metric, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Final diagnostic summary\n"
        f"group={args.group}, layer={args.layer}, scale={args.scale}, K={args.sae_latents}",
        fontweight="bold",
    )

    plt.tight_layout()

    out = SAVE / f"diagnostic_final_bars_{args.group}_layer{args.layer}_scale{args.scale}_K{args.sae_latents}.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()

    print("Saved:", out)

    summary.to_csv(
        SAVE / f"diagnostic_final_summary_{args.group}_layer{args.layer}_scale{args.scale}_K{args.sae_latents}.csv",
        index=False,
    )


def plot_best_latent_stability(all_logs):
    df = all_logs.copy()

    condition_order = ["std_L1_mixed", "topK_mixed", "paired_delta", "random_pair"]
    labels = {
        "std_L1_mixed": "L1 mixed",
        "topK_mixed": "TopK mixed",
        "paired_delta": "Paired delta",
        "random_pair": "Random-pair",
    }

    fig, axes = plt.subplots(len(condition_order), 1, figsize=(12, 3 * len(condition_order)), sharex=True)

    if len(condition_order) == 1:
        axes = [axes]

    for ax, cond in zip(axes, condition_order):
        d = df[df["condition"] == cond].copy()
        if d.empty:
            continue

        # Heatmap: rows=concepts, cols=steps, value=best latent id
        pivot = d.pivot_table(
            index="concept_id",
            columns="step",
            values="best_decoder_latent",
            aggfunc="first",
        )

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(f"{labels[cond]}: best decoder latent over training", fontweight="bold")
        ax.set_ylabel("Concept id")
        ax.set_yticks(np.arange(len(CONCEPTS)))
        ax.set_yticklabels(CONCEPTS, fontsize=8)

        steps = list(pivot.columns)
        tick_positions = np.linspace(0, len(steps) - 1, min(6, len(steps))).astype(int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(steps[i]) for i in tick_positions])

    axes[-1].set_xlabel("Training step")

    plt.suptitle(
        "Best-latent stability diagnostic\n"
        "Stable horizontal colors mean the same latent keeps claiming the direction",
        fontweight="bold",
    )

    plt.tight_layout()

    out = SAVE / f"best_latent_stability_{args.group}_layer{args.layer}_scale{args.scale}_K{args.sae_latents}.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


# ============================================================
# MAIN
# ============================================================

def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\nGenerating datasets...")

    H_tr, S_tr, chosen_tr, Y_tr = generate_mixed_dataset(
        H_BG, TARGET_DIRS, NUISANCE_DIRS,
        args.n_train, args.scale, args.seed + 1,
    )

    H_probe, S_probe, chosen_probe, Y_probe = generate_mixed_dataset(
        H_BG, TARGET_DIRS, NUISANCE_DIRS,
        args.n_probe, args.scale, args.seed + 2,
    )

    D_tr, DS_tr, dchosen_tr, DY_tr = generate_delta_dataset(
        TARGET_DIRS, args.n_train, args.scale, args.seed + 3,
    )

    D_probe, DS_probe, dchosen_probe, DY_probe = generate_delta_dataset(
        TARGET_DIRS, args.n_probe, args.scale, args.seed + 4,
    )

    RP_tr = make_random_pair_deltas(H_tr, args.seed + 5)
    RP_probe = make_random_pair_deltas(H_probe, args.seed + 6)

    # For random-pair, labels are not meaningful. We keep mixed labels only
    # for diagnostic shape compatibility, but recovery/selectivity should be
    # interpreted as a negative control.
    RP_S_probe = S_probe
    RP_chosen_probe = chosen_probe
    RP_Y_probe = Y_probe

    print("H_tr:", tuple(H_tr.shape))
    print("D_tr:", tuple(D_tr.shape))
    print("RP_tr:", tuple(RP_tr.shape))

    results = []

    # Standard L1 mixed
    std_sae = StandardSAE(D_MODEL, args.sae_latents)
    _, log_std, rec_std = train_with_diagnostics(
        condition_name="std_L1_mixed",
        sae=std_sae,
        X_train=H_tr,
        X_probe=H_probe,
        S_probe=S_probe,
        chosen_probe=chosen_probe,
        Y_probe=Y_probe,
        target_dirs=TARGET_DIRS,
        l1_coef=args.l1_coef,
        is_topk=False,
    )
    results.append((log_std, rec_std))

    del std_sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # TopK mixed
    topk_sae = TopKSAE(D_MODEL, args.sae_latents, args.topk_k)
    _, log_topk, rec_topk = train_with_diagnostics(
        condition_name="topK_mixed",
        sae=topk_sae,
        X_train=H_tr,
        X_probe=H_probe,
        S_probe=S_probe,
        chosen_probe=chosen_probe,
        Y_probe=Y_probe,
        target_dirs=TARGET_DIRS,
        l1_coef=0.0,
        is_topk=True,
    )
    results.append((log_topk, rec_topk))

    del topk_sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Paired delta
    delta_sae = StandardSAE(D_MODEL, args.sae_latents)
    _, log_delta, rec_delta = train_with_diagnostics(
        condition_name="paired_delta",
        sae=delta_sae,
        X_train=D_tr,
        X_probe=D_probe,
        S_probe=DS_probe,
        chosen_probe=dchosen_probe,
        Y_probe=DY_probe,
        target_dirs=TARGET_DIRS,
        l1_coef=args.l1_coef,
        is_topk=False,
    )
    results.append((log_delta, rec_delta))

    del delta_sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Random pair
    rp_sae = StandardSAE(D_MODEL, args.sae_latents)
    _, log_rp, rec_rp = train_with_diagnostics(
        condition_name="random_pair",
        sae=rp_sae,
        X_train=RP_tr,
        X_probe=RP_probe,
        S_probe=RP_S_probe,
        chosen_probe=RP_chosen_probe,
        Y_probe=RP_Y_probe,
        target_dirs=TARGET_DIRS,
        l1_coef=args.l1_coef,
        is_topk=False,
    )
    results.append((log_rp, rec_rp))

    del rp_sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_logs = pd.concat([r[0] for r in results], ignore_index=True)
    all_recovery = pd.concat([r[1] for r in results], ignore_index=True)

    all_logs_path = SAVE / f"ALL_DIAGNOSTIC_LOGS_{args.group}_layer{args.layer}_scale{args.scale}_K{args.sae_latents}.csv"
    all_rec_path = SAVE / f"ALL_DIAGNOSTIC_RECOVERY_{args.group}_layer{args.layer}_scale{args.scale}_K{args.sae_latents}.csv"

    all_logs.to_csv(all_logs_path, index=False)
    all_recovery.to_csv(all_rec_path, index=False)

    print("\nSaved:")
    print(" ", all_logs_path)
    print(" ", all_rec_path)

    print("\nFinal recovery:")
    print(
        all_recovery.groupby("condition")
        .agg(
            mean_recovery=("recovery_score", "mean"),
            frac_recovered=("recovery_score", lambda x: float((x >= 0.30).mean())),
        )
        .reset_index()
        .to_string(index=False)
    )

    plot_diagnostic_timeseries(all_logs)
    plot_final_diagnostic_bars(all_logs, all_recovery)
    plot_best_latent_stability(all_logs)

    print("\nDone. Diagnostic outputs saved to:", SAVE)


if __name__ == "__main__":
    main()