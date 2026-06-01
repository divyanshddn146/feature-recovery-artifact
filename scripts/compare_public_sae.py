# compare_public_sae.py
# ============================================================
# Public SAE allocation comparison for the SAE feature-recovery artifact.
#
# Goal:
#   Test whether public/pretrained GPT-2-small residual-stream SAE decoders
#   allocate latents to the controlled target directions used in this project.
#
# Main metric:
#   allocation_score(v) = max_j |cos(public_decoder_j, target_direction_v)|
#
# Compares:
#   - Internal SAE conditions:
#       standard L1 mixed reconstruction
#       TopK mixed reconstruction
#       paired-delta training
#       random-pair delta control
#
#   - Public SAE:
#       SAELens release="gpt2-small-res-jb"
#
# Layer note:
#   The experiment uses blocks.L.hook_resid_post.
#   Public JB SAEs are usually trained on blocks.K.hook_resid_pre.
#
#   In TransformerLens, blocks.L.hook_resid_post is closest to
#   blocks.(L+1).hook_resid_pre.
#
#   Therefore the default mapping is:
#       experiment layer L -> public SAE layer L+1
#
# Example:
#   python scripts/compare_public_sae.py \
#     --out-dir results/main_comparison \
#     --save-dir results/public_sae_comparison
# ============================================================

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--out-dir", type=str, required=True,
                   help="Main experiment output folder, e.g. results/main_comparison")
    p.add_argument("--save-dir", type=str, default="results/public_sae_comparison")

    p.add_argument("--layers", type=int, nargs="+", default=[4, 8, 10])
    p.add_argument("--groups", type=str, nargs="+", default=["weak", "strong"],
                   choices=["weak", "strong", "random"])

    p.add_argument("--scale", type=float, default=0.05,
                   help="Signal scale used for internal SAE allocation comparison")
    p.add_argument("--late-frac", type=float, default=0.10,
                   help="Use final fraction of grad logs for late allocation")

    p.add_argument("--release", type=str, default="gpt2-small-res-jb",
                   help="SAELens pretrained SAE release")
    p.add_argument("--sae-id-template", type=str,
                   default="blocks.{sae_layer}.hook_resid_pre",
                   help="SAELens SAE id template")
    p.add_argument("--sae-layer-offset", type=int, default=1,
                   help="Public SAE layer = experiment layer + offset. Use 1 for resid_post -> next resid_pre.")

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--top-n", type=int, default=10)

    return p.parse_args()


args = parse_args()

OUT = Path(args.out_dir)
SAVE = Path(args.save_dir)
SAVE.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMPORT SAELENS
# ============================================================

try:
    from sae_lens import SAE
except ImportError as e:
    raise ImportError(
        "Missing sae_lens. Install with:\n"
        "  pip install sae-lens\n"
        "Then rerun this script."
    ) from e


# ============================================================
# HELPERS
# ============================================================

COND_LABELS = {
    "std_L1": "L1 mixed",
    "topK20": "TopK mixed",
    "delta": "Paired delta",
    "rand_pair": "Random-pair control",
    "public_sae": "Public JB SAE",
}

COLORS = {
    "std_L1": "#D85A30",
    "topK20": "#378ADD",
    "delta": "#1D9E75",
    "rand_pair": "#888780",
    "public_sae": "#9B59B6",
}


def normalize_rows(x, eps=1e-8):
    return x / x.norm(dim=1, keepdim=True).clamp(min=eps)


def load_target_dirs_and_meta(layer, group):
    layer_dir = OUT / f"layer_{layer}"

    if group == "weak":
        dirs_path = layer_dir / "weak_dirs_z_orth.pt"
        meta_path = layer_dir / "weak_meta.csv"
    elif group == "strong":
        dirs_path = layer_dir / "strong_dirs_z_orth.pt"
        meta_path = layer_dir / "strong_meta.csv"
    else:
        dirs_path = layer_dir / "random_dirs_orth.pt"
        meta_path = layer_dir / "random_meta.csv"

    if not dirs_path.exists():
        raise FileNotFoundError(f"Missing target dirs: {dirs_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    dirs = torch.load(dirs_path, map_location="cpu").float()
    dirs = normalize_rows(dirs)

    meta = pd.read_csv(meta_path)

    concept_col = "target" if "target" in meta.columns else "concept"
    concepts = meta[concept_col].astype(str).tolist()

    return dirs, meta, concepts


def get_public_decoder_matrix(sae, target_dim):
    """
    Return decoder matrix as [n_latents, d_model].

    SAELens usually exposes sae.W_dec with shape [d_sae, d_in].
    This function handles common variants robustly.
    """

    W = None

    if hasattr(sae, "W_dec"):
        W = sae.W_dec
    elif hasattr(sae, "decoder") and hasattr(sae.decoder, "weight"):
        # PyTorch Linear decoder sometimes stores [d_in, d_sae]
        W = sae.decoder.weight
    elif hasattr(sae, "W_dec_DF"):
        W = sae.W_dec_DF
    else:
        raise AttributeError(
            "Could not find decoder weights. Tried sae.W_dec, sae.decoder.weight, sae.W_dec_DF."
        )

    W = W.detach().float().cpu()

    # Want [n_latents, d_model].
    if W.shape[1] == target_dim:
        dec = W
    elif W.shape[0] == target_dim:
        dec = W.T
    else:
        raise ValueError(
            f"Decoder shape {tuple(W.shape)} does not match target_dim={target_dim}."
        )

    dec = normalize_rows(dec)
    return dec


def load_public_sae_for_layer(exp_layer, target_dim):
    sae_layer = exp_layer + args.sae_layer_offset
    sae_id = args.sae_id_template.format(layer=exp_layer, sae_layer=sae_layer)

    print(f"\nLoading public SAE: release={args.release}, sae_id={sae_id}")

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=args.release,
        sae_id=sae_id,
        device=args.device,
    )

    dec = get_public_decoder_matrix(sae, target_dim=target_dim)

    print(f"  Public SAE decoder: {tuple(dec.shape)}")
    return dec, sae_id


def compute_public_allocation(target_dirs, concepts, public_dec, layer, group, sae_id):
    """
    target_dirs: [n_dirs, d_model]
    public_dec: [n_latents, d_model]
    """

    cos = torch.abs(target_dirs @ public_dec.T)  # [n_dirs, n_latents]
    max_cos, best_latent = cos.max(dim=1)

    rows = []

    for i, concept in enumerate(concepts):
        rows.append({
            "source": "public_sae",
            "condition": "public_sae",
            "condition_label": COND_LABELS["public_sae"],
            "direction_group": group,
            "layer": layer,
            "public_sae_id": sae_id,
            "concept_id": i,
            "concept": concept,
            "allocation_score": float(max_cos[i].item()),
            "best_public_latent": int(best_latent[i].item()),
        })

    return pd.DataFrame(rows)


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


def collect_internal_late_allocations(layer, group, concepts):
    """
    Collect late decoder allocation from internal SAE diagnostic logs:
      max_dec_cos averaged over final late-frac of logged steps.

    Returns per-concept rows for:
      std_L1, topK20, delta, rand_pair
      K=512 and K=2048 if available.
    """

    rows = []

    pattern = f"{group}_l{layer}_s*_scale{args.scale}_k*"

    run_dirs = sorted(OUT.glob(pattern))
    if not run_dirs:
        print(f"  Warning: no internal run dirs found for pattern: {pattern}")
        return pd.DataFrame()

    file_map = {
        "std_L1": "grad_log_std_L1.csv",
        "topK20": "grad_log_topK20.csv",
        "delta": "grad_log_delta.csv",
        "rand_pair": "grad_log_rand_pair.csv",
    }

    for run_dir in run_dirs:
        meta = parse_run_dir_name(run_dir.name)
        if meta is None:
            continue

        for cond, filename in file_map.items():
            path = run_dir / filename
            if not path.exists():
                continue

            df = pd.read_csv(path)
            if df.empty:
                continue

            steps = sorted(df["step"].unique())
            n_late = max(1, int(len(steps) * args.late_frac))
            late_steps = set(steps[-n_late:])
            late = df[df["step"].isin(late_steps)].copy()

            # average late max_dec_cos per concept
            per_concept = (
                late.groupby(["concept_id", "concept"], as_index=False)
                .agg(allocation_score=("max_dec_cos", "mean"))
            )

            for _, r in per_concept.iterrows():
                rows.append({
                    "source": "internal",
                    "condition": cond,
                    "condition_label": COND_LABELS[cond],
                    "direction_group": group,
                    "layer": layer,
                    "seed": meta["seed"],
                    "signal_scale": meta["signal_scale"],
                    "sae_latents": meta["sae_latents"],
                    "concept_id": int(r["concept_id"]),
                    "concept": str(r["concept"]),
                    "allocation_score": float(r["allocation_score"]),
                })

    return pd.DataFrame(rows)


def summarize_allocations(df):
    group_cols = ["source", "condition", "condition_label", "direction_group", "layer"]

    optional_cols = []
    if "sae_latents" in df.columns:
        optional_cols.append("sae_latents")
    if "signal_scale" in df.columns:
        optional_cols.append("signal_scale")
    if "public_sae_id" in df.columns:
        optional_cols.append("public_sae_id")

    # Use only columns present and meaningful.
    group_cols = group_cols + [
        c for c in optional_cols
        if c in df.columns and not df[c].isna().all()
    ]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            mean_allocation=("allocation_score", "mean"),
            median_allocation=("allocation_score", "median"),
            min_allocation=("allocation_score", "min"),
            max_allocation=("allocation_score", "max"),
            frac_above_03=("allocation_score", lambda x: float((x >= 0.30).mean())),
            n=("allocation_score", "count"),
        )
        .reset_index()
    )

    return summary


# ============================================================
# PLOTTING
# ============================================================

def plot_public_vs_internal(summary, layer, group):
    """
    Bar plot:
      internal K512/K2048 conditions + public SAE
    """

    # Build compact labels
    plot_rows = []

    for _, r in summary.iterrows():
        cond = r["condition"]

        if cond == "public_sae":
            label = "Public JB SAE"
            order = 100
        else:
            k = r.get("sae_latents", np.nan)
            if pd.isna(k):
                label = r["condition_label"]
            else:
                label = f"{r['condition_label']} K={int(k)}"

            order_map = {
                "std_L1": 0,
                "topK20": 10,
                "rand_pair": 20,
                "delta": 30,
            }
            order = order_map.get(cond, 50) + (0 if pd.isna(k) else int(k) / 10000)

        plot_rows.append({
            "label": label,
            "condition": cond,
            "mean_allocation": r["mean_allocation"],
            "median_allocation": r["median_allocation"],
            "frac_above_03": r["frac_above_03"],
            "order": order,
        })

    plot_df = pd.DataFrame(plot_rows).sort_values("order")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(plot_df))
    colors = [COLORS.get(c, "#555555") for c in plot_df["condition"]]

    ax = axes[0]
    ax.bar(x, plot_df["mean_allocation"], color=colors, alpha=0.9)
    ax.axhline(0.30, color="black", linestyle="--", linewidth=1.2, label="allocation threshold=0.30")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=35, ha="right")
    ax.set_ylabel("Mean allocation score\nmax |cos(decoder, direction)|")
    ax.set_title(f"{group}, layer {layer}: public SAE allocation comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x, plot_df["frac_above_03"], color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=35, ha="right")
    ax.set_ylabel("Fraction of directions above 0.30")
    ax.set_ylim(0, 1.05)
    ax.set_title("How many directions are allocated?", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    out = SAVE / f"public_vs_internal_allocation_{group}_layer{layer}.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


def plot_public_per_concept(public_df, layer, group):
    df = public_df.sort_values("allocation_score", ascending=True).copy()

    fig_h = max(5, 0.35 * len(df))
    fig, ax = plt.subplots(figsize=(9, fig_h))

    y = np.arange(len(df))
    ax.barh(y, df["allocation_score"], color=COLORS["public_sae"], alpha=0.9)
    ax.axvline(0.30, color="black", linestyle="--", linewidth=1.2, label="0.30 threshold")

    ax.set_yticks(y)
    ax.set_yticklabels(df["concept"])
    ax.set_xlabel("Public SAE allocation score\nmax |cos(public decoder, target direction)|")
    ax.set_title(f"Public JB SAE per-concept allocation\n{group}, experiment layer {layer}",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()

    out = SAVE / f"public_per_concept_allocation_{group}_layer{layer}.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


def print_top_bottom(public_df, layer, group):
    print("\n" + "=" * 90)
    print(f"PUBLIC SAE ALLOCATION | group={group} | experiment layer={layer}")
    print("=" * 90)

    top = public_df.sort_values("allocation_score", ascending=False).head(args.top_n)
    bottom = public_df.sort_values("allocation_score", ascending=True).head(args.top_n)

    print("\nTop allocated directions:")
    print(top[["concept", "allocation_score", "best_public_latent"]].to_string(index=False))

    print("\nLowest allocated directions:")
    print(bottom[["concept", "allocation_score", "best_public_latent"]].to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main():
    all_public = []
    all_internal = []
    all_combined = []
    all_summary = []

    print("\nPublic SAE allocation comparison")
    print(f"Experiment out-dir: {OUT}")
    print(f"Save dir:           {SAVE}")
    print(f"SAELens release:    {args.release}")
    print(f"Layer offset:       public SAE layer = experiment layer + {args.sae_layer_offset}")
    print(f"Internal scale:     {args.scale}")

    for layer in args.layers:
        for group in args.groups:
            print("\n" + "#" * 100)
            print(f"Layer={layer} | group={group}")
            print("#" * 100)

            target_dirs, meta, concepts = load_target_dirs_and_meta(layer, group)

            public_dec, sae_id = load_public_sae_for_layer(
                exp_layer=layer,
                target_dim=target_dirs.shape[1],
            )

            public_df = compute_public_allocation(
                target_dirs=target_dirs,
                concepts=concepts,
                public_dec=public_dec,
                layer=layer,
                group=group,
                sae_id=sae_id,
            )

            internal_df = collect_internal_late_allocations(
                layer=layer,
                group=group,
                concepts=concepts,
            )

            combined = pd.concat([internal_df, public_df], ignore_index=True)
            summary = summarize_allocations(combined)

            # Save per-layer/group files
            public_path = SAVE / f"public_allocation_{group}_layer{layer}.csv"
            internal_path = SAVE / f"internal_late_allocation_{group}_layer{layer}.csv"
            combined_path = SAVE / f"combined_allocation_{group}_layer{layer}.csv"
            summary_path = SAVE / f"allocation_summary_{group}_layer{layer}.csv"

            public_df.to_csv(public_path, index=False)
            internal_df.to_csv(internal_path, index=False)
            combined.to_csv(combined_path, index=False)
            summary.to_csv(summary_path, index=False)

            print("Saved:", public_path)
            print("Saved:", internal_path)
            print("Saved:", combined_path)
            print("Saved:", summary_path)

            print_top_bottom(public_df, layer, group)

            if not combined.empty:
                plot_public_vs_internal(summary, layer, group)
            plot_public_per_concept(public_df, layer, group)

            all_public.append(public_df)
            all_internal.append(internal_df)
            all_combined.append(combined)
            all_summary.append(summary)

    # Save global files
    if all_public:
        pd.concat(all_public, ignore_index=True).to_csv(
            SAVE / "ALL_PUBLIC_SAE_ALLOCATION.csv", index=False
        )

    if all_internal:
        pd.concat(all_internal, ignore_index=True).to_csv(
            SAVE / "ALL_INTERNAL_LATE_ALLOCATION.csv", index=False
        )

    if all_combined:
        pd.concat(all_combined, ignore_index=True).to_csv(
            SAVE / "ALL_COMBINED_ALLOCATION.csv", index=False
        )

    if all_summary:
        pd.concat(all_summary, ignore_index=True).to_csv(
            SAVE / "ALL_ALLOCATION_SUMMARY.csv", index=False
        )

    print("\nDone.")
    print("Main outputs:")
    print("  ALL_PUBLIC_SAE_ALLOCATION.csv")
    print("  ALL_INTERNAL_LATE_ALLOCATION.csv")
    print("  ALL_COMBINED_ALLOCATION.csv")
    print("  ALL_ALLOCATION_SUMMARY.csv")
    print("  public_vs_internal_allocation_<group>_layer<layer>.png")
    print("  public_per_concept_allocation_<group>_layer<layer>.png")


if __name__ == "__main__":
    main()