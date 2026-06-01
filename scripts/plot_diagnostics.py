# plot_diagnostics.py
# ============================================================
# Plotting utilities for the SAE feature-recovery artifact.
#
# Purpose:
#   Generate diagnostic figures showing the relationship between
#   target/background variance, decoder allocation, gradient signal,
#   selectivity, and functional recovery.
#
# Main story:
#   target/background variance ratio
#          downarrow
#   decoder latent allocation
#          downarrow
#   decoder-gradient signal
#          downarrow
#   functional recovery or failure
# ============================================================

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--asset-dir", type=str, required=True)
    p.add_argument("--save-dir", type=str, default="figures")

    p.add_argument("--group", type=str, default="weak",
                   choices=["weak", "strong", "random"])
    p.add_argument("--layer", type=int, default=8)

    p.add_argument("--scale-for-capacity", type=float, default=0.05)
    p.add_argument("--early-frac", type=float, default=0.10)
    p.add_argument("--late-frac", type=float, default=0.10)

    p.add_argument("--variance-samples", type=int, default=8000)
    p.add_argument("--variance-seeds", type=int, nargs="+", default=[0, 1, 2])

    # Must match run_comparison.py / dataset-generation settings.
    p.add_argument("--common-nuisance-scale", type=float, default=0.8)
    p.add_argument("--common-nuisance-prob", type=float, default=0.25)
    p.add_argument("--contrast-low", type=float, default=0.8)
    p.add_argument("--contrast-high", type=float, default=1.2)

    p.add_argument("--allocation-threshold", type=float, default=0.30)

    return p.parse_args()


args = parse_args()

OUT = Path(args.out_dir)
ASSET = Path(args.asset_dir)
SAVE = Path(args.save_dir)
SAVE.mkdir(parents=True, exist_ok=True)


# ============================================================
# STYLE
# ============================================================

COND_LABELS = {
    "standard": "L1 mixed",
    "topk": "TopK mixed",
    "delta": "Paired delta",
    "random_pair": "Random-pair control",
    "random_label": "Random-label control",
}

COND_COLORS = {
    "standard": "#D85A30",
    "topk": "#378ADD",
    "delta": "#1D9E75",
    "random_pair": "#888780",
    "random_label": "#B07AA1",
}

COND_MARKERS = {
    "standard": "o",
    "topk": "s",
    "delta": "^",
    "random_pair": "D",
    "random_label": "x",
}


def condition_family(condition):
    c = str(condition)
    if c.startswith("standard_L1"):
        return "standard"
    if c.startswith("topK"):
        return "topk"
    if c.startswith("delta_K"):
        return "delta"
    if c.startswith("random_pair"):
        return "random_pair"
    if c.startswith("random_label"):
        return "random_label"
    return "other"


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():
    agg_path = OUT / "AGGREGATE_RESULTS.csv"
    all_path = OUT / "ALL_SUMMARY.csv"

    if not agg_path.exists():
        raise FileNotFoundError(f"Missing {agg_path}")
    if not all_path.exists():
        raise FileNotFoundError(f"Missing {all_path}")

    agg = pd.read_csv(agg_path)
    all_summary = pd.read_csv(all_path)

    agg["family"] = agg["condition"].apply(condition_family)
    all_summary["family"] = all_summary["condition"].apply(condition_family)

    return agg, all_summary


agg, all_summary = load_results()


# ============================================================
# LOAD ASSETS
# ============================================================

def load_assets():
    h_path = ASSET / "H_bg_z.pt"
    nuisance_path = ASSET / "nuisance_dirs_z_orth.pt"

    if args.group == "weak":
        dirs_path = ASSET / "weak_dirs_z_orth.pt"
    elif args.group == "strong":
        dirs_path = ASSET / "strong_dirs_z_orth.pt"
    else:
        dirs_path = ASSET / "random_dirs_orth.pt"

    missing = [p for p in [h_path, nuisance_path, dirs_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing asset files:\n" + "\n".join(str(p) for p in missing)
        )

    H_bg = torch.load(h_path, map_location="cpu").float()
    nuisance_dirs = torch.load(nuisance_path, map_location="cpu").float()
    target_dirs = torch.load(dirs_path, map_location="cpu").float()

    nuisance_dirs = nuisance_dirs / nuisance_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    target_dirs = target_dirs / target_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)

    return H_bg, nuisance_dirs, target_dirs


H_BG, NUISANCE_DIRS, TARGET_DIRS = load_assets()


# ============================================================
# VARIANCE RATIO COMPUTATION — UPDATED
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


def total_variance(X):
    Xc = X - X.mean(dim=0, keepdim=True)
    return Xc.var(dim=0, unbiased=False).sum().item() + 1e-12


def projected_variance(X, target_dirs):
    Xc = X - X.mean(dim=0, keepdim=True)
    vals = []

    for j in range(target_dirs.shape[0]):
        v = target_dirs[j]
        v = v / v.norm().clamp(min=1e-8)
        proj = Xc @ v
        vals.append(proj.var(unbiased=False).item())

    return float(np.mean(vals))


def generate_background_nuisance(H_bg, nuisance_dirs, n, seed):
    h_base = sample_background(H_bg, n, seed)
    h_base = add_nuisance(h_base, nuisance_dirs, seed + 100)
    return h_base.float()


def generate_target_signal(target_dirs, n, signal_scale, seed):
    torch.manual_seed(seed)

    chosen = torch.randint(0, target_dirs.shape[0], (n,))
    coeff = (
        args.contrast_low
        + (args.contrast_high - args.contrast_low) * torch.rand(n)
    ) * signal_scale

    target_signal = target_dirs[chosen] * coeff[:, None]
    return target_signal.float()


def generate_mixed_dataset_for_metric(H_bg, target_dirs, nuisance_dirs, n, signal_scale, seed):
    background = generate_background_nuisance(H_bg, nuisance_dirs, n, seed)
    target_signal = generate_target_signal(target_dirs, n, signal_scale, seed + 3)
    mixed = background + target_signal
    return background, target_signal, mixed


def make_random_pair(X, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    return (X[rng.permutation(n)] - X[rng.permutation(n)]).float()


def compute_variance_metric_table():
    """
    Computes two metrics:

    1. old_within_dataset_share:
       Var(X @ v) / Var(X)
       Useful for showing delta has high internal direction share.

    2. target_background_var_ratio:
       Var(target_signal) / Var(background+nuisance)
       This is the important threshold x-axis.
    """

    scales = sorted(
        agg.loc[
            (agg["direction_group"] == args.group)
            & (agg["layer"] == args.layer),
            "signal_scale",
        ].dropna().unique()
    )

    rows = []

    for scale in scales:
        for seed in args.variance_seeds:
            background, target_signal, mixed = generate_mixed_dataset_for_metric(
                H_BG,
                TARGET_DIRS,
                NUISANCE_DIRS,
                args.variance_samples,
                scale,
                seed + 1,
            )

            delta = target_signal
            random_pair = make_random_pair(mixed, seed + 50)

            background_var = total_variance(background)
            target_var_abs = projected_variance(target_signal, TARGET_DIRS)

            # This is the new threshold x-axis.
            target_bg_ratio = target_var_abs / background_var

            for dataset, X in [
                ("mixed", mixed),
                ("delta", delta),
                ("random_pair", random_pair),
            ]:
                dataset_total_var = total_variance(X)
                dataset_proj_var = projected_variance(X, TARGET_DIRS)

                old_share = dataset_proj_var / dataset_total_var

                rows.append({
                    "direction_group": args.group,
                    "layer": args.layer,
                    "signal_scale": scale,
                    "seed": seed,
                    "dataset": dataset,

                    # Old metric.
                    "within_dataset_variance_share": old_share,

                    # New threshold metric.
                    # Same target/background ratio for mixed and delta at a given scale,
                    # because both contain the same injected target signal magnitude.
                    # Random-pair gets the same x-coordinate for comparison.
                    "target_background_var_ratio": target_bg_ratio,

                    "background_total_variance": background_var,
                    "target_projected_variance": target_var_abs,
                    "dataset_total_variance": dataset_total_var,
                    "dataset_projected_variance": dataset_proj_var,
                })

    raw = pd.DataFrame(rows)

    summary = (
        raw.groupby(["direction_group", "layer", "signal_scale", "dataset"])
        .agg(
            within_dataset_variance_share_mean=("within_dataset_variance_share", "mean"),
            within_dataset_variance_share_std=("within_dataset_variance_share", "std"),
            target_background_var_ratio_mean=("target_background_var_ratio", "mean"),
            target_background_var_ratio_std=("target_background_var_ratio", "std"),
            background_total_variance_mean=("background_total_variance", "mean"),
            target_projected_variance_mean=("target_projected_variance", "mean"),
        )
        .reset_index()
    )

    raw.to_csv(SAVE / f"variance_metrics_raw_{args.group}_layer{args.layer}.csv", index=False)
    summary.to_csv(SAVE / f"variance_metrics_by_scale_{args.group}_layer{args.layer}.csv", index=False)

    return summary


variance_df = compute_variance_metric_table()


def metric_for_condition(row, metric_name):
    fam = row["family"]

    if fam in ["standard", "topk"]:
        dataset = "mixed"
    elif fam == "delta":
        dataset = "delta"
    elif fam == "random_pair":
        dataset = "random_pair"
    else:
        dataset = "mixed"

    match = variance_df[
        (variance_df["direction_group"] == row["direction_group"])
        & (variance_df["layer"] == row["layer"])
        & (np.isclose(variance_df["signal_scale"], row["signal_scale"]))
        & (variance_df["dataset"] == dataset)
    ]

    if match.empty:
        return np.nan

    return float(match[metric_name].iloc[0])


agg["within_dataset_variance_share"] = agg.apply(
    lambda r: metric_for_condition(r, "within_dataset_variance_share_mean"),
    axis=1,
)

agg["target_background_var_ratio"] = agg.apply(
    lambda r: metric_for_condition(r, "target_background_var_ratio_mean"),
    axis=1,
)

agg.to_csv(SAVE / f"aggregate_with_variance_metrics_{args.group}_layer{args.layer}.csv", index=False)


# ============================================================
# GRAD LOG COLLECTION
# ============================================================

def parse_run_dir_name(name):
    # expected: weak_l8_s0_scale0.05_k512
    m = re.match(
        r"(?P<group>\w+)_l(?P<layer>\d+)_s(?P<seed>\d+)_scale(?P<scale>[\d.]+)_k(?P<k>\d+)",
        name,
    )

    if not m:
        return None

    return {
        "direction_group": m.group("group"),
        "layer": int(m.group("layer")),
        "seed": int(m.group("seed")),
        "signal_scale": float(m.group("scale")),
        "sae_latents": int(m.group("k")),
    }


def collect_grad_logs():
    rows = []

    pattern = f"{args.group}_l{args.layer}_s*_scale*_k*"

    for run_dir in sorted(OUT.glob(pattern)):
        if not run_dir.is_dir():
            continue

        meta = parse_run_dir_name(run_dir.name)
        if meta is None:
            continue

        files = [
            ("grad_log_std_L1.csv", "standard"),
            ("grad_log_topK20.csv", "topk"),
            ("grad_log_delta.csv", "delta"),
            ("grad_log_rand_pair.csv", "random_pair"),
        ]

        for cond_file, fam in files:
            path = run_dir / cond_file
            if not path.exists():
                continue

            df = pd.read_csv(path)
            df["family"] = fam
            df["condition_pretty"] = COND_LABELS[fam]

            for k, v in meta.items():
                df[k] = v

            rows.append(df)

    if not rows:
        raise RuntimeError(
            f"No grad logs found for pattern {pattern}. Check --out-dir, --group, --layer."
        )

    logs = pd.concat(rows, ignore_index=True)

    # Backward compatibility.
    if "max_dec_grad_raw_proj" not in logs.columns:
        logs["max_dec_grad_raw_proj"] = logs["max_grad_proj"]
    if "mean_dec_grad_raw_proj" not in logs.columns:
        logs["mean_dec_grad_raw_proj"] = logs["mean_grad_proj"]

    return logs


grad_logs = collect_grad_logs()


def summarize_grad_logs(logs):
    summary_rows = []

    group_cols = ["family", "signal_scale", "sae_latents"]

    for keys, df in logs.groupby(group_cols):
        fam, scale, k = keys

        step_vals = sorted(df["step"].unique())
        n_steps = len(step_vals)

        n_early = max(1, int(n_steps * args.early_frac))
        n_late = max(1, int(n_steps * args.late_frac))

        early_steps = set(step_vals[:n_early])
        late_steps = set(step_vals[-n_late:])

        early = df[df["step"].isin(early_steps)]
        late = df[df["step"].isin(late_steps)]

        early_grad = early["max_grad_proj"].mean()
        late_grad = late["max_grad_proj"].mean()

        early_alloc = early["max_dec_cos"].mean()
        late_alloc = late["max_dec_cos"].mean()

        summary_rows.append({
            "direction_group": args.group,
            "layer": args.layer,
            "family": fam,
            "condition_pretty": COND_LABELS.get(fam, fam),
            "signal_scale": scale,
            "sae_latents": k,

            "early_grad": early_grad,
            "late_grad": late_grad,
            "early_alloc": early_alloc,
            "late_alloc": late_alloc,

            "allocation_gain": late_alloc - early_alloc,
            "gradient_collapse_ratio": late_grad / (early_grad + 1e-12),
        })

    summary = pd.DataFrame(summary_rows)

    summary["within_dataset_variance_share"] = summary.apply(
        lambda r: metric_for_condition(
            {
                "family": r["family"],
                "direction_group": args.group,
                "layer": args.layer,
                "signal_scale": r["signal_scale"],
            },
            "within_dataset_variance_share_mean",
        ),
        axis=1,
    )

    summary["target_background_var_ratio"] = summary.apply(
        lambda r: metric_for_condition(
            {
                "family": r["family"],
                "direction_group": args.group,
                "layer": args.layer,
                "signal_scale": r["signal_scale"],
            },
            "target_background_var_ratio_mean",
        ),
        axis=1,
    )

    summary.to_csv(SAVE / f"grad_mechanism_summary_{args.group}_layer{args.layer}.csv", index=False)

    return summary


grad_summary = summarize_grad_logs(grad_logs)


# ============================================================
# FIGURE 1 — RECOVERY VS SCALE
# ============================================================

def fig1_recovery_vs_scale():
    df = agg[
        (agg["direction_group"] == args.group)
        & (agg["layer"] == args.layer)
        & (agg["family"].isin(["standard", "topk", "delta", "random_pair"]))
    ].copy()

    if df.empty:
        print("Figure 1 skipped: empty data.")
        return

    ks = sorted(df["sae_latents"].unique())

    fig, axes = plt.subplots(1, len(ks), figsize=(6 * len(ks), 5), squeeze=False)

    for ax, k in zip(axes.flatten(), ks):
        df_k = df[df["sae_latents"] == k]

        for fam in ["standard", "topk", "delta", "random_pair"]:
            df_f = df_k[df_k["family"] == fam].sort_values("signal_scale")

            if df_f.empty:
                continue

            ax.plot(
                df_f["signal_scale"],
                df_f["mean_recovery_mean"],
                marker=COND_MARKERS[fam],
                linewidth=2.2,
                color=COND_COLORS[fam],
                label=COND_LABELS[fam],
            )

            ax.fill_between(
                df_f["signal_scale"],
                df_f["mean_recovery_mean"] - df_f["mean_recovery_std"].fillna(0),
                df_f["mean_recovery_mean"] + df_f["mean_recovery_std"].fillna(0),
                alpha=0.15,
                color=COND_COLORS[fam],
            )

        ax.set_title(f"{args.group} | layer={args.layer} | K={k}", fontweight="bold")
        ax.set_xlabel("Signal scale")
        ax.set_ylabel("Mean recovery")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(
        "Figure 1: Recovery vs Signal Scale\n"
        "Paired deltas recover directions while mixed L1/TopK remain near baseline",
        fontweight="bold",
    )

    plt.tight_layout()

    out = SAVE / f"fig1_recovery_vs_scale_{args.group}_layer{args.layer}.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


# ============================================================
# FIGURE 2 — CAPACITY COMPARISON
# ============================================================

def fig2_capacity_comparison():
    df = agg[
        (agg["direction_group"] == args.group)
        & (agg["layer"] == args.layer)
        & (np.isclose(agg["signal_scale"], args.scale_for_capacity))
        & (agg["family"].isin(["standard", "topk", "delta", "random_pair"]))
    ].copy()

    if df.empty:
        print("Figure 2 skipped: empty data.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # Use categorical x-axis to avoid ugly log tick labels.
    ks = sorted(df["sae_latents"].unique())
    x_pos = np.arange(len(ks))
    k_to_x = {k: i for i, k in enumerate(ks)}

    for fam in ["standard", "topk", "delta", "random_pair"]:
        df_f = df[df["family"] == fam].sort_values("sae_latents")

        if df_f.empty:
            continue

        xs = [k_to_x[k] for k in df_f["sae_latents"]]

        ax.plot(
            xs,
            df_f["mean_recovery_mean"],
            marker=COND_MARKERS[fam],
            linewidth=2.5,
            color=COND_COLORS[fam],
            label=COND_LABELS[fam],
        )

        ax.fill_between(
            xs,
            df_f["mean_recovery_mean"] - df_f["mean_recovery_std"].fillna(0),
            df_f["mean_recovery_mean"] + df_f["mean_recovery_std"].fillna(0),
            alpha=0.15,
            color=COND_COLORS[fam],
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(k)) for k in ks])

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("SAE latents")
    ax.set_ylabel("Mean recovery")

    ax.set_title(
        f"Figure 2: Capacity Does Not Rescue Mixed SAEs\n"
        f"{args.group}, layer={args.layer}, scale={args.scale_for_capacity}",
        fontweight="bold",
    )

    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    out = SAVE / f"fig2_capacity_comparison_{args.group}_layer{args.layer}.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


# ============================================================
# FIGURE 3 — FIXED THRESHOLD PLOT
# ============================================================

def fig3_target_background_threshold():
    rec_df = agg[
        (agg["direction_group"] == args.group)
        & (agg["layer"] == args.layer)
        & (agg["family"].isin(["standard", "topk", "delta", "random_pair"]))
    ].copy()

    alloc_df = grad_summary[
        grad_summary["family"].isin(["standard", "topk", "delta", "random_pair"])
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ------------------------------------------------------------
    # Panel A: target/background ratio vs recovery
    # ------------------------------------------------------------
    ax = axes[0]

    for fam in ["standard", "topk", "delta", "random_pair"]:
        df_f = rec_df[rec_df["family"] == fam]

        if df_f.empty:
            continue

        ax.scatter(
            df_f["target_background_var_ratio"],
            df_f["mean_recovery_mean"],
            s=85,
            marker=COND_MARKERS[fam],
            color=COND_COLORS[fam],
            label=COND_LABELS[fam],
            alpha=0.9,
        )

    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Target/background variance ratio")
    ax.set_ylabel("Mean recovery")
    ax.set_title("A. Recovery turns on with target/background ratio", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ------------------------------------------------------------
    # Panel B: target/background ratio vs decoder allocation
    # ------------------------------------------------------------
    ax = axes[1]

    for fam in ["standard", "topk", "delta", "random_pair"]:
        df_f = alloc_df[alloc_df["family"] == fam]

        if df_f.empty:
            continue

        ax.scatter(
            df_f["target_background_var_ratio"],
            df_f["late_alloc"],
            s=85,
            marker=COND_MARKERS[fam],
            color=COND_COLORS[fam],
            label=COND_LABELS[fam],
            alpha=0.9,
        )

    ax.axhline(
        args.allocation_threshold,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"allocation threshold={args.allocation_threshold}",
    )

    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Target/background variance ratio")
    ax.set_ylabel("Final decoder allocation\n(late max decoder cosine)")
    ax.set_title("B. Decoder allocation threshold", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.suptitle(
        "Figure 3: Allocation Turns On Above a Target-Signal Variance Threshold\n"
        "Capacity alone does not rescue low-salience mixed directions",
        fontweight="bold",
    )

    plt.tight_layout()

    out = SAVE / f"fig3_target_background_threshold_{args.group}_layer{args.layer}.png"
    plt.savefig(out, dpi=260, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


# ============================================================
# FIGURE 4 — DECODER-GRADIENT SIGNAL WITHOUT ALLOCATION
# ============================================================

def fig4_decoder_gradient_vs_allocation():
    scale = args.scale_for_capacity
    k = 512

    df = grad_logs[
        (np.isclose(grad_logs["signal_scale"], scale))
        & (grad_logs["sae_latents"] == k)
        & (grad_logs["family"].isin(["standard", "topk", "delta", "random_pair"]))
    ].copy()

    if df.empty:
        print("Figure 4 skipped: empty data.")
        return

    step_summary = (
        df.groupby(["family", "step"])
        .agg(
            grad=("max_grad_proj", "mean"),
            grad_std=("max_grad_proj", "std"),
            alloc=("max_dec_cos", "mean"),
            alloc_std=("max_dec_cos", "std"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]

    for fam in ["standard", "topk", "delta", "random_pair"]:
        df_f = step_summary[step_summary["family"] == fam].sort_values("step")

        if df_f.empty:
            continue

        ax.plot(
            df_f["step"],
            df_f["grad"],
            color=COND_COLORS[fam],
            linewidth=2.2,
            label=COND_LABELS[fam],
        )

    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Decoder-gradient projection\n(raw projection onto target direction)")
    ax.set_title("A. Residual decoder-gradient pressure", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]

    for fam in ["standard", "topk", "delta", "random_pair"]:
        df_f = step_summary[step_summary["family"] == fam].sort_values("step")

        if df_f.empty:
            continue

        ax.plot(
            df_f["step"],
            df_f["alloc"],
            color=COND_COLORS[fam],
            linewidth=2.2,
            label=COND_LABELS[fam],
        )

    ax.axhline(
        args.allocation_threshold,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"allocation threshold={args.allocation_threshold}",
    )

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Decoder allocation\n(max decoder cosine)")
    ax.set_title("B. Decoder latent allocation", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.suptitle(
        f"Figure 4: Decoder-Gradient Signal Without Allocation\n"
        f"{args.group}, layer={args.layer}, scale={scale}, K={k}",
        fontweight="bold",
    )

    plt.tight_layout()

    out = SAVE / f"fig4_decoder_gradient_vs_allocation_{args.group}_layer{args.layer}.png"
    plt.savefig(out, dpi=260, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


# ============================================================
# FIGURE 5 — ALLOCATION GAIN + GRADIENT COLLAPSE
# ============================================================

def fig5_allocation_gain_gradient_collapse():
    scale = args.scale_for_capacity

    df = grad_summary[
        (np.isclose(grad_summary["signal_scale"], scale))
        & (grad_summary["family"].isin(["standard", "topk", "delta", "random_pair"]))
    ].copy()

    if df.empty:
        print("Figure 5 skipped: empty data.")
        return

    bar_df = (
        df.groupby("family")
        .agg(
            allocation_gain=("allocation_gain", "mean"),
            allocation_gain_std=("allocation_gain", "std"),
            gradient_collapse_ratio=("gradient_collapse_ratio", "mean"),
            gradient_collapse_ratio_std=("gradient_collapse_ratio", "std"),
        )
        .reset_index()
    )

    fam_order = ["standard", "topk", "delta", "random_pair"]
    bar_df["order"] = bar_df["family"].apply(lambda x: fam_order.index(x))
    bar_df = bar_df.sort_values("order")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(bar_df))

    ax = axes[0]

    ax.bar(
        x,
        bar_df["allocation_gain"],
        yerr=bar_df["allocation_gain_std"].fillna(0),
        capsize=4,
        color=[COND_COLORS[f] for f in bar_df["family"]],
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[f] for f in bar_df["family"]],
                       rotation=25, ha="right")
    ax.set_ylabel("Allocation gain\n(late allocation - early allocation)")
    ax.set_title("A. Delta produces large allocation gain", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]

    ax.bar(
        x,
        bar_df["gradient_collapse_ratio"],
        yerr=bar_df["gradient_collapse_ratio_std"].fillna(0),
        capsize=4,
        color=[COND_COLORS[f] for f in bar_df["family"]],
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[f] for f in bar_df["family"]],
                       rotation=25, ha="right")
    ax.set_ylabel("Decoder-gradient collapse ratio\n(late grad / early grad)")
    ax.set_title("B. Successful allocation leaves less residual gradient",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Figure 5: Allocation Followed by Decoder-Gradient Collapse\n"
        f"{args.group}, layer={args.layer}, scale={scale}",
        fontweight="bold",
    )

    plt.tight_layout()

    out = SAVE / f"fig5_allocation_gain_gradient_collapse_{args.group}_layer{args.layer}.png"
    plt.savefig(out, dpi=260, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nLoaded:")
    print(f"  Results: {OUT}")
    print(f"  Assets:  {ASSET}")
    print(f"  Save:    {SAVE}")
    print(f"  Group:   {args.group}")
    print(f"  Layer:   {args.layer}")

    fig1_recovery_vs_scale()
    fig2_capacity_comparison()
    fig3_target_background_threshold()
    fig4_decoder_gradient_vs_allocation()
    fig5_allocation_gain_gradient_collapse()

    print("\nDone. Figures saved to:", SAVE)


if __name__ == "__main__":
    main()