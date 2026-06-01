# make_dataset.py
# ============================================================
# Dataset creation for controlled SAE feature-recovery experiments.
#
# Creates and saves:
#   - background activations
#   - weak target directions
#   - strong target directions
#   - random orthogonal directions
#   - nuisance directions
#   - mixed activation datasets
#   - paired-delta datasets
#   - random-pair delta datasets
#   - metadata and manifest files
#
# Important:
#   This script creates datasets only.
#   It does not train SAEs.
#
# Example:
#   python scripts/make_dataset.py \
#     --model-name gpt2 \
#     --layers 4 8 11 \
#     --out-dir results/dataset \
#     --n-background 5000 \
#     --n-train 20000 \
#     --n-test 5000 \
#     --seeds 0 1 2 \
#     --cache
# ============================================================

import os
import gc
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

try:
    from transformer_lens import HookedTransformer
except ImportError as e:
    raise ImportError("Missing dependency. Install with: pip install transformer-lens einops") from e


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# ARGS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--layers", type=int, nargs="+", default=[4, 8, 11])
    p.add_argument("--out-dir", type=str, default="results/dataset")

    p.add_argument("--n-background", type=int, default=5000)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-test", type=int, default=5000)

    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--signal-scales", type=float, nargs="+", default=[0.02, 0.05, 0.10])

    p.add_argument("--n-templates", type=int, default=8)
    p.add_argument("--n-nuisance", type=int, default=20)
    p.add_argument("--n-random-dirs", type=int, default=30)

    p.add_argument("--contrast-low", type=float, default=0.8)
    p.add_argument("--contrast-high", type=float, default=1.2)

    p.add_argument("--nuisance-scale", type=float, default=0.8)
    p.add_argument("--nuisance-prob", type=float, default=0.25)

    p.add_argument("--position", type=str, default="final", choices=["final", "mean"])
    p.add_argument("--device", type=str, default=None)

    p.add_argument(
        "--cache",
        action="store_true",
        help="Skip extraction/generation if cached files exist.",
    )

    return p.parse_args()


args = parse_args()

DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONCEPT SETS
# ============================================================

weak_concepts = [
    # Medical rare
    {"target": "myocarditis", "baseline": "heart inflammation", "domain": "medical_rare"},
    {"target": "glioblastoma", "baseline": "brain tumor", "domain": "medical_rare"},
    {"target": "bradycardia", "baseline": "slow heartbeat", "domain": "medical_rare"},
    {"target": "telomerase", "baseline": "chromosome enzyme", "domain": "medical_rare"},
    {"target": "prion", "baseline": "misfolded protein", "domain": "medical_rare"},
    {"target": "apoptosis", "baseline": "cell death", "domain": "medical_rare"},
    {"target": "anaphylaxis", "baseline": "severe allergic reaction", "domain": "medical_rare"},

    # Software rare
    {"target": "Kubernetes", "baseline": "container system", "domain": "software_rare"},
    {"target": "PostgreSQL", "baseline": "database system", "domain": "software_rare"},
    {"target": "Redis", "baseline": "data store", "domain": "software_rare"},
    {"target": "GraphQL", "baseline": "API language", "domain": "software_rare"},
    {"target": "Terraform", "baseline": "infrastructure tool", "domain": "software_rare"},
    {"target": "WebSocket", "baseline": "web connection", "domain": "software_rare"},

    # Finance / law rare
    {"target": "arbitrage", "baseline": "price difference trade", "domain": "finance_rare"},
    {"target": "fiduciary", "baseline": "trusted representative", "domain": "law_rare"},
    {"target": "subrogation", "baseline": "insurance transfer", "domain": "law_rare"},
    {"target": "escrow", "baseline": "third party account", "domain": "finance_rare"},

    # Physics / math rare
    {"target": "eigenvalue", "baseline": "matrix scaling number", "domain": "math_rare"},
    {"target": "phonon", "baseline": "lattice vibration", "domain": "physics_rare"},
    {"target": "fermion", "baseline": "matter particle", "domain": "physics_rare"},

    # Compound concepts
    {"target": "metaphor about memory loss", "baseline": "metaphor", "domain": "compound"},
    {"target": "regret about financial decisions", "baseline": "regret", "domain": "compound"},
    {"target": "irony in political speech", "baseline": "irony", "domain": "compound"},
    {"target": "nostalgia for childhood food", "baseline": "nostalgia", "domain": "compound"},

    # Subtle distinctions
    {"target": "mitigate", "baseline": "reduce", "domain": "subtle"},
    {"target": "exacerbate", "baseline": "worsen", "domain": "subtle"},
    {"target": "ameliorate", "baseline": "improve", "domain": "subtle"},
    {"target": "substantiate", "baseline": "prove", "domain": "subtle"},
    {"target": "refute", "baseline": "disagree with", "domain": "subtle"},
    {"target": "infer", "baseline": "conclude", "domain": "subtle"},
]

# Neutral common topics for strong/common baseline directions.
strong_concepts = [
    {"target": "sports", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "technology", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "science", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "music", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "weather", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "food", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "history", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "gardening", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "education", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "health", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "economics", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "travel", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "animals", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "cooking", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "art", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "literature", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "cinema", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "environment", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "language", "baseline": "ordinary topic", "domain": "common_baseline"},
    {"target": "mathematics", "baseline": "ordinary topic", "domain": "common_baseline"},
]

assert len(weak_concepts) == 30, f"Expected 30 weak concepts, got {len(weak_concepts)}"
assert len(strong_concepts) == 20, f"Expected 20 strong concepts, got {len(strong_concepts)}"


ALL_DIRECTION_TEMPLATES = [
    "The topic is {term}.",
    "This sentence is about {term}.",
    "The article discusses {term}.",
    "A specialist mentioned {term}.",
    "The report focused on {term}.",
    "The technical phrase is {term}.",
    "The textbook describes {term}.",
    "The paragraph explains {term}.",
    "The document covers {term}.",
    "The passage is related to {term}.",
    "The concept being discussed is {term}.",
    "The explanation concerns {term}.",
    "The notes mention {term}.",
    "The summary refers to {term}.",
    "The key term is {term}.",
    "The subject of the text is {term}.",
]

direction_templates = ALL_DIRECTION_TEMPLATES[:args.n_templates]

nuisance_terms = [
    "science", "technology", "history", "economics", "education",
    "software", "health", "music", "sports", "weather",
    "school", "business", "language", "family", "internet",
    "research", "engineering", "mathematics", "culture", "travel",
    "art", "food", "animals", "cinema", "environment",
]

assert args.n_nuisance <= len(nuisance_terms), (
    f"--n-nuisance={args.n_nuisance} exceeds nuisance_terms length {len(nuisance_terms)}"
)

assert args.n_templates <= len(ALL_DIRECTION_TEMPLATES), (
    f"--n-templates={args.n_templates} exceeds available template count {len(ALL_DIRECTION_TEMPLATES)}"
)


# ============================================================
# UTILS
# ============================================================

def save_json(obj: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp(min=eps)


def orthogonalize_rows(dirs: torch.Tensor) -> torch.Tensor:
    """
    Orthogonalize row vectors using QR on transpose.
    Input:  [n_dirs, d_model]
    Output: [n_dirs, d_model]
    """
    if dirs.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(dirs.shape)}")

    n_dirs, d_model = dirs.shape
    if n_dirs > d_model:
        raise ValueError(
            f"Cannot orthogonalize {n_dirs} directions in dimension {d_model}"
        )

    A = dirs.T.detach().cpu().numpy()  # [d_model, n_dirs]
    Q, _ = np.linalg.qr(A)
    orth = torch.tensor(Q[:, :n_dirs].T, dtype=torch.float32)
    return normalize_rows(orth)


def make_background_prompts(n: int, seed: int) -> List[str]:
    templates = [
        "The article discusses science, technology, and society in simple language.",
        "A researcher wrote a short explanation about a recent discovery.",
        "The report describes several facts and gives a neutral summary.",
        "In the following paragraph, the author explains an ordinary topic.",
        "The document contains background information about common events.",
        "A student read a passage about history, engineering, and mathematics.",
        "The lecture notes introduce a concept and provide examples.",
        "The news summary describes economic, social, and cultural topics.",
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

    modifiers = [
        " It includes examples and definitions.",
        " The language is neutral and factual.",
        " The text avoids unusual rare terminology.",
        " Several common concepts are mentioned.",
        " The explanation is intended for a general audience.",
        " It contains ordinary background information.",
        " The writing style is clear and concise.",
        " The content is suitable for a general readership.",
    ]

    rng = random.Random(seed)
    return [rng.choice(templates) + rng.choice(modifiers) for _ in range(n)]


# ============================================================
# SAVE CONFIG EARLY
# ============================================================

save_json(vars(args), OUT_DIR / "config.json")


# ============================================================
# LOAD MODEL
# ============================================================

print(f"Device: {DEVICE}")
print(f"Model:  {args.model_name}")
print(f"Layers: {args.layers}")

print(f"\nLoading {args.model_name}...")
model = HookedTransformer.from_pretrained(args.model_name, device=DEVICE)
model.eval()

n_layers = model.cfg.n_layers
D_MODEL = model.cfg.d_model

print(f"n_layers={n_layers}, d_model={D_MODEL}")

for layer in args.layers:
    assert layer < n_layers, (
        f"Layer {layer} out of range. Model has layers 0 to {n_layers - 1}."
    )


# ============================================================
# EXTRACTION UTILITIES
# ============================================================

def extract_activation(prompt: str, hook_point: str) -> torch.Tensor:
    """
    Extract one residual-stream vector from one prompt at one hook point.
    Returns CPU float32 tensor [d_model].
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)
        h = cache[hook_point]

        if args.position == "final":
            vec = h[0, -1, :].detach().cpu().float()
        else:
            vec = h[0].mean(dim=0).detach().cpu().float()

        del cache, h

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vec


# ============================================================
# LAYER ASSET EXTRACTION
# ============================================================

def layer_paths(layer: int) -> Dict[str, Path]:
    layer_dir = OUT_DIR / f"layer_{layer}"
    return {
        # Main filenames
        "H_bg": layer_dir / "H_bg_z.pt",
        "bg_scale": layer_dir / "bg_scale.pt",
        "bg_mean": layer_dir / "bg_mean.pt",

        "weak_dirs": layer_dir / "weak_dirs.pt",
        "weak_meta": layer_dir / "weak_meta.csv",

        "strong_dirs": layer_dir / "strong_dirs.pt",
        "strong_meta": layer_dir / "strong_meta.csv",

        "random_dirs": layer_dir / "random_dirs.pt",
        "random_meta": layer_dir / "random_meta.csv",

        "nuisance_dirs": layer_dir / "nuisance_dirs.pt",

        # Backward-compatible filenames
        "weak_dirs_compat": layer_dir / "weak_dirs_z_orth.pt",
        "strong_dirs_compat": layer_dir / "strong_dirs_z_orth.pt",
        "random_dirs_compat": layer_dir / "random_dirs_orth.pt",
        "nuisance_dirs_compat": layer_dir / "nuisance_dirs_z_orth.pt",
    }


def required_asset_paths(paths: Dict[str, Path]) -> List[Path]:
    return [
        paths["H_bg"],
        paths["bg_scale"],
        paths["bg_mean"],
        paths["weak_dirs"],
        paths["weak_meta"],
        paths["strong_dirs"],
        paths["strong_meta"],
        paths["random_dirs"],
        paths["random_meta"],
        paths["nuisance_dirs"],
    ]


def load_cached_assets(paths: Dict[str, Path]) -> Dict[str, Any]:
    return {
        "H_bg": torch.load(paths["H_bg"], map_location="cpu"),
        "bg_scale": torch.load(paths["bg_scale"], map_location="cpu"),
        "bg_mean": torch.load(paths["bg_mean"], map_location="cpu"),
        "weak_dirs": torch.load(paths["weak_dirs"], map_location="cpu"),
        "weak_meta": pd.read_csv(paths["weak_meta"]),
        "strong_dirs": torch.load(paths["strong_dirs"], map_location="cpu"),
        "strong_meta": pd.read_csv(paths["strong_meta"]),
        "random_dirs": torch.load(paths["random_dirs"], map_location="cpu"),
        "random_meta": pd.read_csv(paths["random_meta"]),
        "nuisance_dirs": torch.load(paths["nuisance_dirs"], map_location="cpu"),
    }


def extract_layer_assets(layer: int) -> Dict[str, Any]:
    """
    Extract and save all assets for one layer:
      - background activations + scaler
      - weak concept directions
      - strong concept directions
      - random directions
      - nuisance directions
    """
    hook_point = f"blocks.{layer}.hook_resid_post"
    layer_dir = OUT_DIR / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    paths = layer_paths(layer)

    if args.cache and all(p.exists() for p in required_asset_paths(paths)):
        print(f"  Loading cached assets for layer {layer}")
        return load_cached_assets(paths)

    print(f"\n{'=' * 80}")
    print(f"Extracting assets for layer {layer} | hook={hook_point}")
    print(f"{'=' * 80}")

    # ------------------------------------------------------------
    # Background activations
    # ------------------------------------------------------------
    print(f"\nBackground activations ({args.n_background} prompts)...")
    bg_prompts = make_background_prompts(args.n_background, seed=1234 + layer)

    bg_acts = []
    for prompt in tqdm(bg_prompts, desc=f"Layer {layer} background"):
        bg_acts.append(extract_activation(prompt, hook_point))

    H_bg_raw = torch.stack(bg_acts).float()  # [n_background, d_model]

    scaler = StandardScaler()
    H_bg_z = torch.tensor(scaler.fit_transform(H_bg_raw.numpy()), dtype=torch.float32)
    bg_scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    bg_mean = torch.tensor(scaler.mean_, dtype=torch.float32)

    print(
        f"  H_bg_z: {tuple(H_bg_z.shape)} | "
        f"mean={H_bg_z.mean().item():.4f} std={H_bg_z.std().item():.4f}"
    )

    del bg_acts, H_bg_raw
    gc.collect()

    # ------------------------------------------------------------
    # Concept directions
    # ------------------------------------------------------------
    def extract_concept_directions(
        concepts: List[Dict[str, str]],
        label: str,
    ) -> Tuple[torch.Tensor, pd.DataFrame]:
        directions = []
        rows = []

        print(f"\n{label} directions ({len(concepts)} concepts, {len(direction_templates)} templates)...")

        for ci, concept in enumerate(tqdm(concepts, desc=f"Layer {layer} {label} dirs")):
            raw_deltas = []

            for template in direction_templates:
                h_t = extract_activation(template.format(term=concept["target"]), hook_point)
                h_b = extract_activation(template.format(term=concept["baseline"]), hook_point)
                raw_deltas.append(h_t - h_b)

            v_raw = torch.stack(raw_deltas).mean(dim=0).float()
            raw_norm = float(v_raw.norm().item())

            # Transform to standardized background coordinate space.
            v_z_raw = v_raw / (bg_scale + 1e-8)
            z_norm_before_normalization = float(v_z_raw.norm().item())

            # Unit-normalize before orthogonalization.
            v_z = v_z_raw / (v_z_raw.norm() + 1e-8)

            directions.append(v_z.float())
            rows.append({
                "concept_id": ci,
                "target": concept["target"],
                "baseline": concept["baseline"],
                "domain": concept.get("domain", "unknown"),
                "group": label,
                "raw_norm": raw_norm,
                "z_norm_before_normalization": z_norm_before_normalization,
                "n_templates": len(direction_templates),
                "layer": layer,
                "hook_point": hook_point,
                "position": args.position,
            })

        dirs_tensor = torch.stack(directions).float()
        dirs_orth = orthogonalize_rows(dirs_tensor)
        meta_df = pd.DataFrame(rows)

        return dirs_orth, meta_df

    weak_dirs, weak_meta = extract_concept_directions(weak_concepts, "weak")
    strong_dirs, strong_meta = extract_concept_directions(strong_concepts, "strong")

    # ------------------------------------------------------------
    # Random orthogonal directions
    # ------------------------------------------------------------
    print(f"\nRandom directions ({args.n_random_dirs})...")

    rng = np.random.default_rng(9000 + layer)
    A = rng.normal(size=(D_MODEL, args.n_random_dirs)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    random_dirs = torch.tensor(Q[:, :args.n_random_dirs].T, dtype=torch.float32)
    random_dirs = normalize_rows(random_dirs)

    random_meta = pd.DataFrame([
        {
            "concept_id": i,
            "target": f"random_{i}",
            "baseline": "none",
            "domain": "random",
            "group": "random",
            "raw_norm": 1.0,
            "z_norm_before_normalization": 1.0,
            "n_templates": 0,
            "layer": layer,
            "hook_point": hook_point,
            "position": args.position,
        }
        for i in range(args.n_random_dirs)
    ])

    # ------------------------------------------------------------
    # Nuisance directions
    # ------------------------------------------------------------
    print(f"\nNuisance directions ({args.n_nuisance} terms)...")

    h_base = extract_activation("The topic is ordinary information.", hook_point)
    nuisance = []

    for term in tqdm(nuisance_terms[:args.n_nuisance], desc=f"Layer {layer} nuisance"):
        h = extract_activation(f"The topic is {term}.", hook_point)
        v_raw = h - h_base
        v_z_raw = v_raw / (bg_scale + 1e-8)
        v_z = v_z_raw / (v_z_raw.norm() + 1e-8)
        nuisance.append(v_z.float())

    nuisance_dirs = orthogonalize_rows(torch.stack(nuisance).float())

    # ------------------------------------------------------------
    # Save assets
    # ------------------------------------------------------------
    torch.save(H_bg_z, paths["H_bg"])
    torch.save(bg_scale, paths["bg_scale"])
    torch.save(bg_mean, paths["bg_mean"])

    torch.save(weak_dirs, paths["weak_dirs"])
    torch.save(strong_dirs, paths["strong_dirs"])
    torch.save(random_dirs, paths["random_dirs"])
    torch.save(nuisance_dirs, paths["nuisance_dirs"])

    weak_meta.to_csv(paths["weak_meta"], index=False)
    strong_meta.to_csv(paths["strong_meta"], index=False)
    random_meta.to_csv(paths["random_meta"], index=False)

    # Backward-compatible copies for older scripts.
    torch.save(weak_dirs, paths["weak_dirs_compat"])
    torch.save(strong_dirs, paths["strong_dirs_compat"])
    torch.save(random_dirs, paths["random_dirs_compat"])
    torch.save(nuisance_dirs, paths["nuisance_dirs_compat"])

    print(f"\nSaved assets to {layer_dir}")
    print(f"  H_bg_z:        {tuple(H_bg_z.shape)}")
    print(f"  weak_dirs:     {tuple(weak_dirs.shape)}")
    print(f"  strong_dirs:   {tuple(strong_dirs.shape)}")
    print(f"  random_dirs:   {tuple(random_dirs.shape)}")
    print(f"  nuisance_dirs: {tuple(nuisance_dirs.shape)}")

    return {
        "H_bg": H_bg_z,
        "bg_scale": bg_scale,
        "bg_mean": bg_mean,
        "weak_dirs": weak_dirs,
        "weak_meta": weak_meta,
        "strong_dirs": strong_dirs,
        "strong_meta": strong_meta,
        "random_dirs": random_dirs,
        "random_meta": random_meta,
        "nuisance_dirs": nuisance_dirs,
    }


# ============================================================
# DATASET GENERATION
# ============================================================

def sample_background(H_bg: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randint(0, H_bg.shape[0], (n,), generator=g)
    return H_bg[idx]


def add_nuisance(h: torch.Tensor, nuisance_dirs: torch.Tensor, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)

    n, nc = h.shape[0], nuisance_dirs.shape[0]
    mask = (torch.rand(n, nc) < args.nuisance_prob).float()
    coeff = mask * (torch.rand(n, nc) * 2 - 1) * args.nuisance_scale

    return h + coeff @ nuisance_dirs


def generate_mixed_dataset(
    H_bg: torch.Tensor,
    target_dirs: torch.Tensor,
    nuisance_dirs: torch.Tensor,
    n: int,
    signal_scale: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mixed activation dataset:
      X = background + nuisance + target_signal

    Returns:
      H_mix: [n, d_model]
      Y:     [n, n_dirs] one-hot concept labels
      chosen:[n] concept IDs
    """
    torch.manual_seed(seed)

    h_base = sample_background(H_bg, n, seed)
    h_base = add_nuisance(h_base, nuisance_dirs, seed + 100)

    chosen = torch.randint(0, target_dirs.shape[0], (n,))
    coeff = (
        args.contrast_low
        + (args.contrast_high - args.contrast_low) * torch.rand(n)
    ) * signal_scale

    h_mix = h_base + target_dirs[chosen] * coeff[:, None]

    Y = torch.zeros(n, target_dirs.shape[0])
    Y[torch.arange(n), chosen] = 1.0

    return h_mix.float(), Y.float(), chosen.long()


def generate_delta_dataset(
    target_dirs: torch.Tensor,
    n: int,
    signal_scale: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure paired-delta dataset:
      D = target_signal

    Returns:
      D:      [n, d_model]
      chosen: [n] concept IDs
    """
    torch.manual_seed(seed)

    chosen = torch.randint(0, target_dirs.shape[0], (n,))
    coeff = (
        args.contrast_low
        + (args.contrast_high - args.contrast_low) * torch.rand(n)
    ) * signal_scale

    D = target_dirs[chosen] * coeff[:, None]

    return D.float(), chosen.long()


def make_random_pair_deltas(
    X_tr: torch.Tensor,
    X_te: torch.Tensor,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random-pair delta control:
      RP = X[random_perm_1] - X[random_perm_2]

    This is not a matched target-baseline delta.
    It tests whether arbitrary pairwise differencing is enough.
    """
    rng = np.random.default_rng(seed)

    n_tr = X_tr.shape[0]
    n_te = X_te.shape[0]

    p1_tr = torch.tensor(rng.permutation(n_tr), dtype=torch.long)
    p2_tr = torch.tensor(rng.permutation(n_tr), dtype=torch.long)
    p1_te = torch.tensor(rng.permutation(n_te), dtype=torch.long)
    p2_te = torch.tensor(rng.permutation(n_te), dtype=torch.long)

    RP_tr = X_tr[p1_tr] - X_tr[p2_tr]
    RP_te = X_te[p1_te] - X_te[p2_te]

    return RP_tr.float(), RP_te.float()


def generate_and_save_datasets(
    layer: int,
    assets: Dict[str, Any],
    signal_scale: float,
    seed: int,
) -> None:
    """
    Generate and save mixed + delta + random-pair datasets for one
    layer / signal_scale / seed combination.
    """
    H_bg = assets["H_bg"]
    nuisance_dirs = assets["nuisance_dirs"]

    group_to_dirs = {
        "weak": assets["weak_dirs"],
        "strong": assets["strong_dirs"],
        "random": assets["random_dirs"],
    }

    scale_dir = OUT_DIR / f"layer_{layer}" / f"scale_{signal_scale}" / f"seed_{seed}"
    scale_dir.mkdir(parents=True, exist_ok=True)

    if args.cache and (scale_dir / "done.txt").exists():
        print(f"  Skipping cached datasets: layer={layer} scale={signal_scale} seed={seed}")
        return

    print(f"\n  Generating datasets: layer={layer} scale={signal_scale} seed={seed}")

    for group_name, target_dirs in group_to_dirs.items():
        group_dir = scale_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        H_tr, Y_tr, ch_tr = generate_mixed_dataset(
            H_bg=H_bg,
            target_dirs=target_dirs,
            nuisance_dirs=nuisance_dirs,
            n=args.n_train,
            signal_scale=signal_scale,
            seed=seed + 1,
        )

        H_te, Y_te, ch_te = generate_mixed_dataset(
            H_bg=H_bg,
            target_dirs=target_dirs,
            nuisance_dirs=nuisance_dirs,
            n=args.n_test,
            signal_scale=signal_scale,
            seed=seed + 2,
        )

        D_tr, D_ch_tr = generate_delta_dataset(
            target_dirs=target_dirs,
            n=args.n_train,
            signal_scale=signal_scale,
            seed=seed + 3,
        )

        D_te, D_ch_te = generate_delta_dataset(
            target_dirs=target_dirs,
            n=args.n_test,
            signal_scale=signal_scale,
            seed=seed + 4,
        )

        RP_tr, RP_te = make_random_pair_deltas(H_tr, H_te, seed=seed + 5)

        # Mixed datasets
        torch.save(H_tr, group_dir / "H_tr.pt")
        torch.save(Y_tr, group_dir / "Y_tr.pt")
        torch.save(ch_tr, group_dir / "ch_tr.pt")

        torch.save(H_te, group_dir / "H_te.pt")
        torch.save(Y_te, group_dir / "Y_te.pt")
        torch.save(ch_te, group_dir / "ch_te.pt")

        # Delta datasets
        torch.save(D_tr, group_dir / "D_tr.pt")
        torch.save(D_ch_tr, group_dir / "D_ch_tr.pt")

        torch.save(D_te, group_dir / "D_te.pt")
        torch.save(D_ch_te, group_dir / "D_ch_te.pt")

        # Random-pair deltas
        torch.save(RP_tr, group_dir / "RP_tr.pt")
        torch.save(RP_te, group_dir / "RP_te.pt")

        # Compact group metadata
        group_info = {
            "group": group_name,
            "layer": layer,
            "signal_scale": signal_scale,
            "seed": seed,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "n_dirs": int(target_dirs.shape[0]),
            "d_model": int(target_dirs.shape[1]),
            "files": {
                "H_tr.pt": "mixed train activations",
                "Y_tr.pt": "mixed train one-hot concept labels",
                "ch_tr.pt": "mixed train chosen concept ids",
                "H_te.pt": "mixed test activations",
                "Y_te.pt": "mixed test one-hot concept labels",
                "ch_te.pt": "mixed test chosen concept ids",
                "D_tr.pt": "paired-delta train activations",
                "D_ch_tr.pt": "paired-delta train chosen concept ids",
                "D_te.pt": "paired-delta test activations",
                "D_ch_te.pt": "paired-delta test chosen concept ids",
                "RP_tr.pt": "random-pair delta train activations",
                "RP_te.pt": "random-pair delta test activations",
            },
        }
        save_json(group_info, group_dir / "group_info.json")

        print(
            f"    {group_name:>6s}: "
            f"H_tr={tuple(H_tr.shape)} H_te={tuple(H_te.shape)} "
            f"D_tr={tuple(D_tr.shape)} RP_tr={tuple(RP_tr.shape)}"
        )

        del H_tr, Y_tr, ch_tr, H_te, Y_te, ch_te, D_tr, D_ch_tr, D_te, D_ch_te, RP_tr, RP_te
        gc.collect()

    with open(scale_dir / "done.txt", "w") as f:
        f.write(f"layer={layer} scale={signal_scale} seed={seed}\n")


# ============================================================
# MANIFEST
# ============================================================

def estimate_dataset_size_gb() -> float:
    """
    Approximate activation tensor size only.
    Counts:
      - groups: weak/strong/random = 3
      - activation types: mixed, delta, random-pair = 3
      - train + test samples
      - float32
    Labels and metadata are small compared with activations.
    """
    n_layers_run = len(args.layers)
    n_scales = len(args.signal_scales)
    n_seeds = len(args.seeds)
    n_groups = 3
    n_activation_types = 3  # mixed + delta + random-pair
    n_samples = args.n_train + args.n_test
    bytes_per_float = 4

    total_bytes = (
        n_layers_run
        * n_scales
        * n_seeds
        * n_groups
        * n_activation_types
        * n_samples
        * D_MODEL
        * bytes_per_float
    )

    return total_bytes / 1e9


def save_manifest() -> None:
    manifest = {
        "model": args.model_name,
        "d_model": int(D_MODEL),
        "n_layers_model": int(n_layers),
        "layers": args.layers,
        "position": args.position,

        "n_background": args.n_background,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "seeds": args.seeds,
        "signal_scales": args.signal_scales,

        "contrast_low": args.contrast_low,
        "contrast_high": args.contrast_high,
        "nuisance_scale": args.nuisance_scale,
        "nuisance_prob": args.nuisance_prob,

        "n_weak": len(weak_concepts),
        "n_strong": len(strong_concepts),
        "n_random": args.n_random_dirs,
        "n_nuisance": args.n_nuisance,
        "n_templates": len(direction_templates),
        "direction_templates": direction_templates,

        "weak_concepts": weak_concepts,
        "strong_concepts": strong_concepts,

        "estimated_activation_dataset_size_gb": estimate_dataset_size_gb(),

        "file_structure": {
            "layer_{L}/H_bg_z.pt": "standardized background activations [n_background, d_model]",
            "layer_{L}/bg_scale.pt": "StandardScaler scale [d_model]",
            "layer_{L}/bg_mean.pt": "StandardScaler mean [d_model]",

            "layer_{L}/weak_dirs.pt": "weak concept directions [30, d_model]",
            "layer_{L}/weak_dirs_z_orth.pt": "backward-compatible copy of weak_dirs.pt",
            "layer_{L}/weak_meta.csv": "weak concept metadata",

            "layer_{L}/strong_dirs.pt": "strong concept directions [20, d_model]",
            "layer_{L}/strong_dirs_z_orth.pt": "backward-compatible copy of strong_dirs.pt",
            "layer_{L}/strong_meta.csv": "strong concept metadata",

            "layer_{L}/random_dirs.pt": "random orthogonal directions [n_random, d_model]",
            "layer_{L}/random_dirs_orth.pt": "backward-compatible copy of random_dirs.pt",
            "layer_{L}/random_meta.csv": "random direction metadata",

            "layer_{L}/nuisance_dirs.pt": "nuisance directions [n_nuisance, d_model]",
            "layer_{L}/nuisance_dirs_z_orth.pt": "backward-compatible copy of nuisance_dirs.pt",

            "layer_{L}/scale_{S}/seed_{R}/{group}/H_tr.pt": "mixed train activations",
            "layer_{L}/scale_{S}/seed_{R}/{group}/Y_tr.pt": "mixed train one-hot labels",
            "layer_{L}/scale_{S}/seed_{R}/{group}/ch_tr.pt": "mixed train chosen concept ids",
            "layer_{L}/scale_{S}/seed_{R}/{group}/H_te.pt": "mixed test activations",
            "layer_{L}/scale_{S}/seed_{R}/{group}/Y_te.pt": "mixed test one-hot labels",
            "layer_{L}/scale_{S}/seed_{R}/{group}/ch_te.pt": "mixed test chosen concept ids",
            "layer_{L}/scale_{S}/seed_{R}/{group}/D_tr.pt": "paired-delta train activations",
            "layer_{L}/scale_{S}/seed_{R}/{group}/D_ch_tr.pt": "paired-delta train chosen concept ids",
            "layer_{L}/scale_{S}/seed_{R}/{group}/D_te.pt": "paired-delta test activations",
            "layer_{L}/scale_{S}/seed_{R}/{group}/D_ch_te.pt": "paired-delta test chosen concept ids",
            "layer_{L}/scale_{S}/seed_{R}/{group}/RP_tr.pt": "random-pair delta train activations",
            "layer_{L}/scale_{S}/seed_{R}/{group}/RP_te.pt": "random-pair delta test activations",
        },
    }

    save_json(manifest, OUT_DIR / "manifest.json")
    print(f"\nManifest saved to {OUT_DIR / 'manifest.json'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"\n{'=' * 80}")
    print("DATASET CREATION")
    print(f"{'=' * 80}")
    print(f"Model:        {args.model_name}")
    print(f"Device:       {DEVICE}")
    print(f"Layers:       {args.layers}")
    print(f"Position:     {args.position}")
    print(f"Concepts:     {len(weak_concepts)} weak, {len(strong_concepts)} strong, {args.n_random_dirs} random")
    print(f"Samples:      {args.n_background} background, {args.n_train} train, {args.n_test} test")
    print(f"Seeds:        {args.seeds}")
    print(f"Scales:       {args.signal_scales}")
    print(f"Templates:    {len(direction_templates)}")
    print(f"Output dir:   {OUT_DIR}")
    print(f"{'=' * 80}")

    size_gb = estimate_dataset_size_gb()
    print(f"\nEstimated activation dataset size: ~{size_gb:.2f} GB")
    print("This estimate includes mixed + delta + random-pair activation tensors.")
    print("Labels and metadata add a smaller amount.")

    for layer in args.layers:
        print(f"\n{'=' * 80}")
        print(f"LAYER {layer}")
        print(f"{'=' * 80}")

        assets = extract_layer_assets(layer)

        for scale in args.signal_scales:
            for seed in args.seeds:
                generate_and_save_datasets(
                    layer=layer,
                    assets=assets,
                    signal_scale=scale,
                    seed=seed,
                )

        del assets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_manifest()

    print(f"\n{'=' * 80}")
    print("DATASET CREATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Output directory: {OUT_DIR}")

    print("\nExample loading code:")
    print("""
from pathlib import Path
import torch

layer = 8
scale = 0.05
seed = 0
group = "weak"

base = Path("OUT_DIR_HERE") / f"layer_{layer}" / f"scale_{scale}" / f"seed_{seed}" / group

H_tr = torch.load(base / "H_tr.pt")
Y_tr = torch.load(base / "Y_tr.pt")
D_tr = torch.load(base / "D_tr.pt")
RP_tr = torch.load(base / "RP_tr.pt")

weak_dirs = torch.load(Path("OUT_DIR_HERE") / f"layer_{layer}" / "weak_dirs.pt")
""".replace("OUT_DIR_HERE", str(OUT_DIR)))


if __name__ == "__main__":
    main()
