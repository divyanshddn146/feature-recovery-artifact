# find_weak_directions.py
# ============================================================
# Direction-discovery diagnostic for SAE feature-recovery experiments.
#
# Purpose:
#   Identify candidate concepts with weak or low-SNR residual-stream
#   directions that can be used in controlled SAE recovery experiments.
#
# Strategy:
#   For each candidate concept pair, compute a delta direction and measure:
#     1. delta norm
#     2. SNR estimate against background activations
#     3. cosine similarity to top background PCs
#
# Low-SNR candidates are directions with small delta norm and low alignment
# with dominant background principal components.
#
# Example:
#   python scripts/find_weak_directions.py \
#     --model-name gpt2 \
#     --layer 8 \
#     --out-dir results/weak_direction_scan
# ============================================================

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformer_lens import HookedTransformer
except ImportError:
    raise ImportError("pip install transformer-lens einops")


# ============================================================
# ARGS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--layer", type=int, default=8,
                   help="Residual stream layer to probe")
    p.add_argument("--position-strategy", type=str, default="final",
                   choices=["final", "mean"])
    p.add_argument("--n-background", type=int, default=500,
                   help="Background prompts for SNR baseline")
    p.add_argument("--n-templates", type=int, default=6,
                   help="Templates averaged per concept direction")
    p.add_argument("--n-pcs", type=int, default=20,
                   help="Top background PCs to measure alignment against")
    p.add_argument("--out-dir", type=str, default="results/weak_direction_scan")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--top-k-weak", type=int, default=20,
                   help="How many weakest directions to print")
    return p.parse_args()


args = parse_args()
DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_DIR / "config.json", "w") as f:
    json.dump(vars(args), f, indent=2)


# ============================================================
# CANDIDATE CONCEPT PAIRS
# ============================================================
# Format: (target, baseline, category)
# The direction = h(target prompt) - h(baseline prompt)
# We want pairs where the DIFFERENCE is meaningful but weak.
#
# Designed to cover:
#   - Domain-specific rare terms vs generic descriptions
#   - Compound/relational concepts vs single concepts
#   - Subtle distinctions vs obvious ones

candidates = [
    # --- Medical rare vs general ---
    ("myocarditis", "heart inflammation", "medical_rare"),
    ("glioblastoma", "brain tumor", "medical_rare"),
    ("bradycardia", "slow heartbeat", "medical_rare"),
    ("telomerase", "chromosome enzyme", "medical_rare"),
    ("prion", "misfolded protein", "medical_rare"),
    ("apoptosis", "cell death", "medical_rare"),
    ("anaphylaxis", "severe allergic reaction", "medical_rare"),
    ("thrombocytopenia", "low platelet count", "medical_rare"),

    # --- Software rare vs general ---
    ("WebSocket", "web connection", "software_rare"),
    ("Kubernetes", "container system", "software_rare"),
    ("OAuth", "login protocol", "software_rare"),
    ("PostgreSQL", "database system", "software_rare"),
    ("GraphQL", "API language", "software_rare"),
    ("Redis", "data store", "software_rare"),
    ("gRPC", "remote procedure protocol", "software_rare"),
    ("Terraform", "infrastructure tool", "software_rare"),

    # --- Physics/math rare vs general ---
    ("eigenvalue", "matrix scaling number", "math_rare"),
    ("quasar", "distant bright object", "astro_rare"),
    ("phonon", "lattice vibration", "physics_rare"),
    ("fermion", "matter particle", "physics_rare"),
    ("perovskite", "crystal structure", "materials_rare"),
    ("renormalization", "physics calculation method", "physics_rare"),

    # --- Finance/law rare vs general ---
    ("subrogation", "insurance transfer", "law_rare"),
    ("fiduciary", "trusted representative", "law_rare"),
    ("arbitrage", "price difference trade", "finance_rare"),
    ("escrow", "third party account", "finance_rare"),

    # --- Compound/relational (harder for models) ---
    ("sarcasm about economics", "sarcasm", "compound"),
    ("uncertainty about medical diagnosis", "uncertainty", "compound"),
    ("irony in political speech", "irony", "compound"),
    ("metaphor about memory loss", "metaphor", "compound"),
    ("nostalgia for childhood food", "nostalgia", "compound"),
    ("regret about financial decisions", "regret", "compound"),

    # --- Subtle semantic distinctions ---
    ("imply", "state directly", "subtle"),
    ("infer", "conclude", "subtle"),
    ("refute", "disagree with", "subtle"),
    ("mitigate", "reduce", "subtle"),
    ("exacerbate", "worsen", "subtle"),
    ("ameliorate", "improve", "subtle"),
    ("corroborate", "confirm", "subtle"),
    ("substantiate", "prove", "subtle"),

    # --- Common concepts (expected to be STRONG — good baseline) ---
    ("science", "ordinary topic", "common_baseline"),
    ("technology", "ordinary topic", "common_baseline"),
    ("food", "ordinary topic", "common_baseline"),
    ("weather", "ordinary topic", "common_baseline"),
    ("sports", "ordinary topic", "common_baseline"),
    ("music", "ordinary topic", "common_baseline"),
]

templates = [
    "The topic is {term}.",
    "This text is about {term}.",
    "The article discusses {term}.",
    "A specialist mentioned {term}.",
    "The report is focused on {term}.",
    "The paragraph explains {term}.",
    "The textbook describes {term}.",
    "The document covers {term}.",
][:args.n_templates]

background_prompts = [
    "The article discusses science, technology, and society in simple language.",
    "A researcher wrote a short explanation about a recent discovery.",
    "The report describes several facts and gives a neutral summary.",
    "In the following paragraph, the author explains an ordinary topic.",
    "The document contains background information about common events.",
    "A student read a passage about history, engineering, and mathematics.",
    "The lecture notes introduce a concept and provide examples.",
    "The news summary describes economic, political, and cultural topics.",
    "This paragraph is written in a clear and informative style.",
    "The textbook section explains an idea using several definitions.",
    "A technical document describes a system and its components.",
    "The conversation includes questions about learning and problem solving.",
    "The passage discusses health, education, software, and science.",
    "The user asked for a concise explanation of a familiar topic.",
    "The assistant provides a careful answer with relevant details.",
    "The overview covers key ideas in a structured and readable format.",
    "The notes summarize the main points of a lecture on general topics.",
    "The essay introduces a subject and provides supporting evidence.",
    "The review describes the content of a book about everyday life.",
    "The tutorial walks through a common task step by step.",
]


# ============================================================
# LOAD MODEL
# ============================================================

model = HookedTransformer.from_pretrained(
    "gemma-2b",
    device=DEVICE,
)
n_layers = model.cfg.n_layers
assert args.layer < n_layers, (
    f"Layer {args.layer} out of range — model only has {n_layers} layers (0-{n_layers-1})"
)
HOOK = f"blocks.{args.layer}.hook_resid_post"
print(f"n_layers={n_layers}, probing hook={HOOK}")
model.eval()
D_MODEL = model.cfg.d_model
HOOK = f"blocks.{args.layer}.hook_resid_post"
print(f"d_model={D_MODEL}, probing hook={HOOK}")


# ============================================================
# EXTRACTION
# ============================================================

def extract_activation(prompt):
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)
        h = cache[HOOK]
        if args.position_strategy == "final":
            vec = h[0, -1, :].detach().cpu().float()
        else:
            vec = h[0].mean(dim=0).detach().cpu().float()
        del cache, h
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vec


def extract_background(prompts):
    acts = []
    print(f"\nExtracting {len(prompts)} background activations...")
    for i, p in enumerate(prompts):
        acts.append(extract_activation(p))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(prompts)}")
    return torch.stack(acts).numpy()


def extract_delta(target, baseline):
    deltas = []
    for t in templates:
        h_t = extract_activation(t.format(term=target))
        h_b = extract_activation(t.format(term=baseline))
        deltas.append((h_t - h_b).numpy())
    return np.stack(deltas).mean(axis=0)


# ============================================================
# BACKGROUND STATS
# ============================================================

bg_raw = extract_background(background_prompts[:args.n_background])

scaler = StandardScaler()
bg_z = scaler.fit_transform(bg_raw)

bg_std = bg_z.std(axis=0)
bg_global_std = float(np.std(bg_z))

print(f"\nBackground global std (standardized): {bg_global_std:.4f}")
print(f"Background mean activation norm: {np.linalg.norm(bg_raw, axis=1).mean():.4f}")

pca = PCA(n_components=args.n_pcs)
pca.fit(bg_z)
bg_pcs = pca.components_

print(f"\nTop {args.n_pcs} PCs explain {pca.explained_variance_ratio_.sum()*100:.1f}% of background variance")


# ============================================================
# SCAN EACH CANDIDATE
# ============================================================

rows = []

print(f"\nScanning {len(candidates)} candidate concept pairs...")
print("=" * 70)

for i, (target, baseline, category) in enumerate(candidates):
    delta = extract_delta(target, baseline)

    delta_norm = float(np.linalg.norm(delta))

    delta_unit = delta / (delta_norm + 1e-8)

    pc_cosines = np.abs(bg_pcs @ delta_unit)
    max_pc_cos = float(pc_cosines.max())
    mean_pc_cos = float(pc_cosines.mean())

    delta_z = scaler.transform(delta.reshape(1, -1))[0]
    delta_z_norm = float(np.linalg.norm(delta_z))
    snr_estimate = delta_z_norm / (bg_global_std * np.sqrt(D_MODEL) + 1e-8)

    rows.append({
        "target": target,
        "baseline": baseline,
        "category": category,
        "delta_norm": delta_norm,
        "delta_norm_z": delta_z_norm,
        "snr_estimate": snr_estimate,
        "max_pc_cosine": max_pc_cos,
        "mean_pc_cos_top20": mean_pc_cos,
        "weakness_score": snr_estimate * (1 + max_pc_cos),
    })

    print(
        f"{i:02d} {target:35s} "
        f"norm={delta_norm:.3f} "
        f"snr={snr_estimate:.3f} "
        f"max_pc_cos={max_pc_cos:.3f} "
        f"[{category}]"
    )

df = pd.DataFrame(rows)
df = df.sort_values("weakness_score")
df.to_csv(OUT_DIR / "direction_scan_results.csv", index=False)


# ============================================================
# REPORT WEAK CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print(f"TOP {args.top_k_weak} WEAKEST DIRECTIONS (sorted by weakness_score)")
print("These are best candidates for contrastive SAE testing")
print("=" * 70)

weak = df.head(args.top_k_weak)
for _, row in weak.iterrows():
    print(
        f"  {row['target']:35s} | "
        f"snr={row['snr_estimate']:.4f} | "
        f"max_pc={row['max_pc_cosine']:.3f} | "
        f"score={row['weakness_score']:.4f} | "
        f"[{row['category']}]"
    )

print("\n" + "=" * 70)
print("TOP STRONG DIRECTIONS (expected to work on both SAEs)")
print("=" * 70)

strong = df.tail(10).iloc[::-1]
for _, row in strong.iterrows():
    print(
        f"  {row['target']:35s} | "
        f"snr={row['snr_estimate']:.4f} | "
        f"max_pc={row['max_pc_cosine']:.3f} | "
        f"[{row['category']}]"
    )


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

categories = df["category"].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
cat_color = {c: col for c, col in zip(categories, colors)}

ax = axes[0]
for _, row in df.iterrows():
    ax.scatter(
        row["snr_estimate"],
        row["max_pc_cosine"],
        color=cat_color[row["category"]],
        alpha=0.8,
        s=60,
    )
    ax.annotate(
        row["target"],
        (row["snr_estimate"], row["max_pc_cosine"]),
        fontsize=6,
        alpha=0.7,
        xytext=(3, 2),
        textcoords="offset points",
    )

ax.axvline(df["snr_estimate"].quantile(0.25), color="red", linestyle="--",
           alpha=0.4, label="25th pct SNR")
ax.axhline(df["max_pc_cosine"].quantile(0.75), color="orange", linestyle="--",
           alpha=0.4, label="75th pct PC alignment")

ax.set_xlabel("SNR estimate (lower = weaker signal)")
ax.set_ylabel("Max cosine with top BG PCs (higher = more aligned with noise)")
ax.set_title("Direction weakness map\n(bottom-left = best contrastive SAE candidates)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cat_color[c], label=c) for c in categories]
ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

ax = axes[1]
df_plot = df.head(30).copy()
bars = ax.barh(
    range(len(df_plot)),
    df_plot["weakness_score"],
    color=[cat_color[c] for c in df_plot["category"]],
    alpha=0.8,
)
ax.set_yticks(range(len(df_plot)))
ax.set_yticklabels(df_plot["target"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Weakness score (lower = weaker)")
ax.set_title("Top 30 weakest directions\n(best candidates for contrastive SAE test)")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "weakness_map.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {OUT_DIR}/weakness_map.png")


# ============================================================
# GENERATE PROMPT PAIRS FOR NEXT EXPERIMENT
# ============================================================

print("\n" + "=" * 70)
print("SUGGESTED PROMPT PAIRS FOR CONTRASTIVE SAE EXPERIMENT")
print("Candidate directions can be used in the main comparison script.")
print("=" * 70)

weak_pairs = df.head(15)
print("\nweak_candidates = [")
for _, row in weak_pairs.iterrows():
    print(f'    {{"target": "{row["target"]}", "baseline": "{row["baseline"]}", '
          f'"domain": "{row["category"]}", '
          f'"snr": {row["snr_estimate"]:.4f}}},')
print("]")

print(f"\nFull results saved to {OUT_DIR}/direction_scan_results.csv")