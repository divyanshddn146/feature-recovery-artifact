# run_comparison.py
# ============================================================
# Main controlled comparison for SAE feature recovery.
#
# Conditions:
#   1. Standard L1 SAE on mixed activations
#   2. TopK SAE on mixed activations
#   3. Paired-delta SAE
#   4. Random-label control
#   5. Random-pair delta control
#
# Main mechanism tested:
#
#   distributional variance share
#           downarrow
#   decoder latent allocation
#           downarrow
#   decoder-gradient signal
#           downarrow
#   functional recovery / failure
#
# Logged metrics:
#   - decoder allocation to target directions
#   - decoder-gradient projection onto target directions
#   - recovery score
#   - selectivity
#   - per-concept and aggregate summaries
#
# Example:
#   python scripts/run_comparison.py \
#     --model-name gpt2 \
#     --layers 8 \
#     --out-dir results/main_comparison \
#     --seeds 0 1 2 \
#     --sae-latents 512 2048 \
#     --signal-scales 0.05 \
#     --cache-layer-assets
# ============================================================

import os
import gc
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

try:
    from transformer_lens import HookedTransformer
except ImportError:
    raise ImportError("pip install transformer-lens einops")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--out-dir", type=str, default="results/main_comparison")

    parser.add_argument("--layers", type=int, nargs="+", default=[8])
    parser.add_argument(
        "--direction-groups",
        type=str,
        nargs="+",
        default=["weak", "strong", "random"],
        choices=["weak", "strong", "random"],
    )

    parser.add_argument(
        "--rare-scales",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05, 0.10],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sae-latents", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--topk-k", type=int, default=20)
    parser.add_argument("--l1-list", type=float, nargs="+", default=[1e-3])

    parser.add_argument(
        "--position-strategy",
        type=str,
        default="final",
        choices=["final", "mean"],
    )

    parser.add_argument("--n-background-prompts", type=int, default=2000)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-test", type=int, default=3000)

    parser.add_argument("--sae-steps", type=int, default=2500)
    parser.add_argument("--sae-batch-size", type=int, default=512)
    parser.add_argument("--sae-lr", type=float, default=1e-3)
    parser.add_argument(
        "--grad-log-every",
        type=int,
        default=50,
        help="Log decoder-gradient/allocation metrics every N steps.",
    )

    parser.add_argument("--n-directions", type=int, default=10)
    parser.add_argument("--common-nuisance-scale", type=float, default=0.8)
    parser.add_argument("--common-nuisance-prob", type=float, default=0.25)
    parser.add_argument("--n-common-nuisance", type=int, default=20)
    parser.add_argument("--contrast-low", type=float, default=0.8)
    parser.add_argument("--contrast-high", type=float, default=1.2)
    parser.add_argument("--recovery-threshold", type=float, default=0.30)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-layer-assets", action="store_true")
    parser.add_argument("--no-plots", action="store_true")

    return parser.parse_args()


args = parse_args()

DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_DIR / "run_config.json", "w") as f:
    json.dump(vars(args), f, indent=2)


# ============================================================
# CONCEPT SETS
# ============================================================

weak_concepts = [
    {"target": "exacerbate", "baseline": "worsen", "domain": "subtle", "snr": 3.327},
    {"target": "mitigate", "baseline": "reduce", "domain": "subtle", "snr": 3.282},
    {"target": "ameliorate", "baseline": "improve", "domain": "subtle", "snr": 3.392},
    {"target": "Redis", "baseline": "data store", "domain": "software_rare", "snr": 3.405},
    {"target": "Kubernetes", "baseline": "container system", "domain": "software_rare", "snr": 3.478},
    {"target": "arbitrage", "baseline": "price difference trade", "domain": "finance_rare", "snr": 3.554},
    {"target": "PostgreSQL", "baseline": "database system", "domain": "software_rare", "snr": 3.470},
    {"target": "metaphor about memory loss", "baseline": "metaphor", "domain": "compound", "snr": 3.409},
    {"target": "telomerase", "baseline": "chromosome enzyme", "domain": "medical_rare", "snr": 3.544},
    {"target": "regret about financial decisions", "baseline": "regret", "domain": "compound", "snr": 3.359},
]

strong_concepts = [
    {"target": "sports", "baseline": "ordinary topic", "domain": "common_baseline", "snr": 3.734},
    {"target": "technology", "baseline": "ordinary topic", "domain": "common_baseline", "snr": 3.602},
    {"target": "science", "baseline": "ordinary topic", "domain": "common_baseline", "snr": 3.576},
    {"target": "music", "baseline": "ordinary topic", "domain": "common_baseline", "snr": 3.615},
    {"target": "weather", "baseline": "ordinary topic", "domain": "common_baseline", "snr": 3.662},
    {"target": "food", "baseline": "ordinary topic", "domain": "common_baseline", "snr": 3.682},
]

direction_templates = [
    "The topic is {term}.",
    "This sentence is about {term}.",
    "The article discusses {term}.",
    "A specialist mentioned {term}.",
    "The report focused on {term}.",
    "The technical phrase is {term}.",
    "The textbook describes {term}.",
    "The paragraph explains {term}.",
]

nuisance_terms = [
    "science", "technology", "history", "economics", "education",
    "software", "health", "politics", "music", "sports",
    "weather", "school", "business", "language", "family",
    "internet", "research", "engineering", "mathematics", "culture",
]


# ============================================================
# LOAD MODEL
# ============================================================

print(f"\nLoading model: {args.model_name}")
model = HookedTransformer.from_pretrained(args.model_name, device=DEVICE)
model.eval()

n_layers = model.cfg.n_layers
D_MODEL = model.cfg.d_model
print(f"n_layers={n_layers}, d_model={D_MODEL}")

for layer in args.layers:
    assert layer < n_layers, f"Layer {layer} out of range"


# ============================================================
# ACTIVATION EXTRACTION
# ============================================================

def extract_activation(prompt, hook_point):
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)
        h = cache[hook_point]

        if args.position_strategy == "final":
            vec = h[0, -1, :].detach().cpu()
        else:
            vec = h[0].mean(0).detach().cpu()

        del cache, h

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vec.float()


def make_background_prompts(n, seed):
    templates = [
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
    ]

    modifiers = [
        " It includes examples and definitions.",
        " The language is neutral and factual.",
        " The text avoids unusual rare terminology.",
        " Several common concepts are mentioned.",
        " The explanation is intended for a general audience.",
        " It contains ordinary background information.",
    ]

    rng = random.Random(seed)
    return [rng.choice(templates) + rng.choice(modifiers) for _ in range(n)]


def extract_background_activations_and_scaler(hook_point, seed):
    prompts = make_background_prompts(args.n_background_prompts, seed)
    acts = []

    for p in tqdm(prompts, desc="Background", unit="prompt"):
        acts.append(extract_activation(p, hook_point))

    H_raw = torch.stack(acts).float()

    scaler = StandardScaler()
    H_z = torch.tensor(scaler.fit_transform(H_raw.numpy()), dtype=torch.float32)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32)

    print(f"Background: {H_z.shape} | mean={H_z.mean():.4f} std={H_z.std():.4f}")
    return H_z, mean, scale


def transform_raw_delta_to_z_direction(v_raw, bg_scale):
    v_z = v_raw / (bg_scale + 1e-8)
    v_z = v_z / (v_z.norm() + 1e-8)
    return v_z.float()


def orthogonalize_directions(dirs):
    A = dirs.T.numpy()
    Q, _ = np.linalg.qr(A)
    dirs_orth = torch.tensor(Q[:, :dirs.shape[0]].T, dtype=torch.float32)
    dirs_orth = dirs_orth / dirs_orth.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return dirs_orth


def extract_prompt_directions(concepts, hook_point, label, bg_scale):
    rows = []
    directions = []

    for ci, concept in enumerate(tqdm(concepts[:args.n_directions], desc=f"{label} dirs")):
        raw_deltas = []

        for template in direction_templates:
            h_t = extract_activation(template.format(term=concept["target"]), hook_point)
            h_b = extract_activation(template.format(term=concept["baseline"]), hook_point)
            raw_deltas.append(h_t - h_b)

        v_raw = torch.stack(raw_deltas).mean(0)
        raw_norm = v_raw.norm().item()
        v_z = transform_raw_delta_to_z_direction(v_raw, bg_scale)

        directions.append(v_z)
        rows.append({
            "direction_group": label,
            "concept_id": ci,
            "target": concept["target"],
            "baseline": concept["baseline"],
            "domain": concept.get("domain", "unknown"),
            "snr": concept.get("snr", 0.0),
            "raw_direction_norm": raw_norm,
        })

    dirs = torch.stack(directions).float()
    dirs_orth = orthogonalize_directions(dirs)

    return dirs_orth, pd.DataFrame(rows)


def make_random_directions(n, d_model, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d_model, n)).astype(np.float32)
    Q, _ = np.linalg.qr(A)

    dirs = torch.tensor(Q[:, :n].T, dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)

    rows = [
        {
            "direction_group": "random",
            "concept_id": i,
            "target": f"random_{i}",
            "baseline": "none",
            "domain": "random",
            "snr": 0.0,
            "raw_direction_norm": 1.0,
        }
        for i in range(n)
    ]

    return dirs, pd.DataFrame(rows)


def extract_nuisance_directions(hook_point, bg_scale):
    h_base = extract_activation("The topic is ordinary information.", hook_point)
    dirs = []

    for term in tqdm(nuisance_terms[:args.n_common_nuisance], desc="Nuisance"):
        h = extract_activation(f"The topic is {term}.", hook_point)
        v_z = transform_raw_delta_to_z_direction(h - h_base, bg_scale)
        dirs.append(v_z)

    dirs_orth = orthogonalize_directions(torch.stack(dirs).float())
    return dirs_orth


def get_or_create_layer_assets(layer):
    hook_point = f"blocks.{layer}.hook_resid_post"
    layer_dir = OUT_DIR / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "H_bg": layer_dir / "H_bg_z.pt",
        "bg_mean": layer_dir / "bg_mean.pt",
        "bg_scale": layer_dir / "bg_scale.pt",
        "weak_dirs": layer_dir / "weak_dirs_z_orth.pt",
        "weak_meta": layer_dir / "weak_meta.csv",
        "strong_dirs": layer_dir / "strong_dirs_z_orth.pt",
        "strong_meta": layer_dir / "strong_meta.csv",
        "random_dirs": layer_dir / "random_dirs_orth.pt",
        "random_meta": layer_dir / "random_meta.csv",
        "nuisance": layer_dir / "nuisance_dirs_z_orth.pt",
    }

    if args.cache_layer_assets and all(p.exists() for p in paths.values()):
        print(f"Loading cached assets for layer {layer}")

        H_bg = torch.load(paths["H_bg"], map_location="cpu")
        weak_dirs = torch.load(paths["weak_dirs"], map_location="cpu")
        weak_meta = pd.read_csv(paths["weak_meta"])
        strong_dirs = torch.load(paths["strong_dirs"], map_location="cpu")
        strong_meta = pd.read_csv(paths["strong_meta"])
        random_dirs = torch.load(paths["random_dirs"], map_location="cpu")
        random_meta = pd.read_csv(paths["random_meta"])
        nuisance = torch.load(paths["nuisance"], map_location="cpu")

    else:
        H_bg, bg_mean, bg_scale = extract_background_activations_and_scaler(
            hook_point,
            seed=1234 + layer,
        )

        weak_dirs, weak_meta = extract_prompt_directions(
            weak_concepts,
            hook_point,
            "weak",
            bg_scale,
        )
        strong_dirs, strong_meta = extract_prompt_directions(
            strong_concepts,
            hook_point,
            "strong",
            bg_scale,
        )
        random_dirs, random_meta = make_random_directions(
            args.n_directions,
            D_MODEL,
            seed=9000 + layer,
        )
        nuisance = extract_nuisance_directions(hook_point, bg_scale)

        if args.cache_layer_assets:
            torch.save(H_bg, paths["H_bg"])
            torch.save(bg_mean, paths["bg_mean"])
            torch.save(bg_scale, paths["bg_scale"])
            torch.save(weak_dirs, paths["weak_dirs"])
            weak_meta.to_csv(paths["weak_meta"], index=False)
            torch.save(strong_dirs, paths["strong_dirs"])
            strong_meta.to_csv(paths["strong_meta"], index=False)
            torch.save(random_dirs, paths["random_dirs"])
            random_meta.to_csv(paths["random_meta"], index=False)
            torch.save(nuisance, paths["nuisance"])

    direction_assets = {
        "weak": (weak_dirs, weak_meta),
        "strong": (strong_dirs, strong_meta),
        "random": (random_dirs, random_meta),
    }

    return H_bg, nuisance, {
        k: v for k, v in direction_assets.items()
        if k in args.direction_groups
    }


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
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    return h_mix.float(), Y.float(), chosen.numpy()


def generate_delta_dataset(target_dirs, n, signal_scale, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    chosen = torch.randint(0, target_dirs.shape[0], (n,))
    coeff = (
        args.contrast_low
        + (args.contrast_high - args.contrast_low) * torch.rand(n)
    ) * signal_scale

    return (target_dirs[chosen] * coeff[:, None]).float(), chosen.numpy()


def make_random_pair_deltas(X_tr, X_te, seed):
    rng = np.random.default_rng(seed)

    n_tr = X_tr.shape[0]
    n_te = X_te.shape[0]

    RP_tr = X_tr[rng.permutation(n_tr)] - X_tr[rng.permutation(n_tr)]
    RP_te = X_te[rng.permutation(n_te)] - X_te[rng.permutation(n_te)]

    return RP_tr.float(), RP_te.float()


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
        return self.decoder(z), z

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
        return self.decoder(z), z

    def normalize_decoder(self):
        with torch.no_grad():
            W = self.decoder.weight.data
            self.decoder.weight.data = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)


# ============================================================
# DECODER-GRADIENT LOGGING
# ============================================================

def _normalize_rows(x, eps=1e-8):
    return x / x.norm(dim=1, keepdim=True).clamp(min=eps)


def compute_decoder_direction_metrics(sae, concept_dirs, concept_names):
    """
    Compute decoder-gradient and decoder-allocation metrics.

    Main metrics:
      - raw decoder-gradient projection:
          residual update pressure toward a target direction.

      - decoder-gradient cosine:
          direction-only gradient alignment.

      - decoder cosine:
          whether a decoder latent is allocated to the target direction.

    Backward-compatible columns:
      - max_grad_proj / mean_grad_proj are raw decoder-gradient projection.
      - max_dec_cos / mean_dec_cos are decoder allocation.
    """

    if sae.decoder.weight.grad is None:
        return []

    # Decoder columns are feature directions.
    # decoder.weight: [d_model, n_latents]
    # transpose:      [n_latents, d_model]
    dec_w = sae.decoder.weight.detach().T.cpu().float()
    dec_w = _normalize_rows(dec_w)

    # Negative gradient is the gradient-descent update direction.
    dec_grad_raw = (-sae.decoder.weight.grad.detach()).T.cpu().float()
    dec_grad_normed = _normalize_rows(dec_grad_raw)

    rows = []

    for ci, (direction, name) in enumerate(zip(concept_dirs, concept_names)):
        direction = direction.detach().cpu().float()
        direction = direction / direction.norm().clamp(min=1e-8)

        # Raw decoder-gradient projection: main gradient availability metric.
        raw_proj = (dec_grad_raw @ direction).abs()
        max_raw_proj = raw_proj.max().item()
        mean_raw_proj = raw_proj.mean().item()

        # Direction-only decoder-gradient cosine.
        grad_cos = (dec_grad_normed @ direction).abs()
        max_grad_cos = grad_cos.max().item()
        mean_grad_cos = grad_cos.mean().item()

        # Decoder alignment / latent allocation.
        dec_cos = (dec_w @ direction).abs()
        max_dec_cos = dec_cos.max().item()
        mean_dec_cos = dec_cos.mean().item()

        rows.append({
            "concept_id": ci,
            "concept": name,

            # Backward-compatible names used by plotting script.
            # These now refer to RAW DECODER-gradient projection.
            "max_grad_proj": max_raw_proj,
            "mean_grad_proj": mean_raw_proj,

            # Explicit decoder-gradient metrics.
            "max_dec_grad_raw_proj": max_raw_proj,
            "mean_dec_grad_raw_proj": mean_raw_proj,
            "max_dec_grad_cos": max_grad_cos,
            "mean_dec_grad_cos": mean_grad_cos,

            # Latent allocation metrics.
            "max_dec_cos": max_dec_cos,
            "mean_dec_cos": mean_dec_cos,
        })

    return rows


def train_sae_with_grad_log(
    X_train,
    sae,
    optimizer,
    l1_coef,
    label,
    concept_dirs,
    concept_names,
    is_topk=False,
):
    X = X_train.to(DEVICE)
    n = X.shape[0]
    grad_log = []

    for step in tqdm(range(args.sae_steps), desc=label, leave=False):
        idx = torch.randint(0, n, (args.sae_batch_size,), device=DEVICE)
        batch = X[idx]

        x_hat, z = sae(batch)

        mse_loss = ((x_hat - batch) ** 2).mean()

        if is_topk:
            l1_loss = torch.tensor(0.0, device=DEVICE)
            loss = mse_loss
        else:
            l1_loss = z.abs().mean()
            loss = mse_loss + l1_coef * l1_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if step % args.grad_log_every == 0 and concept_dirs is not None:
            metric_rows = compute_decoder_direction_metrics(
                sae=sae,
                concept_dirs=concept_dirs,
                concept_names=concept_names,
            )

            for row in metric_rows:
                row.update({
                    "step": step,
                    "label": label,
                    "loss": loss.item(),
                    "mse_loss": mse_loss.item(),
                    "l1_loss": l1_loss.item(),
                    "l1_coef": l1_coef,
                    "is_topk": bool(is_topk),
                    "sae_latents": sae.decoder.weight.shape[1],
                })
                grad_log.append(row)

        optimizer.step()
        sae.normalize_decoder()

        if step % 500 == 0 or step == args.sae_steps - 1:
            tqdm.write(
                f"[{label}] step={step:4d} "
                f"loss={loss.item():.4e} "
                f"mse={mse_loss.item():.4e} "
                f"l1={l1_loss.item():.4e}"
            )

    grad_log_df = pd.DataFrame(grad_log) if grad_log else pd.DataFrame()
    return sae, grad_log_df


def train_standard_sae(
    X_train,
    input_dim,
    n_latents,
    l1_coef,
    seed,
    label,
    concept_dirs=None,
    concept_names=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    sae = StandardSAE(input_dim, n_latents).to(DEVICE)
    opt = optim.Adam(sae.parameters(), lr=args.sae_lr)

    return train_sae_with_grad_log(
        X_train=X_train,
        sae=sae,
        optimizer=opt,
        l1_coef=l1_coef,
        label=label,
        concept_dirs=concept_dirs,
        concept_names=concept_names,
        is_topk=False,
    )


def train_topk_sae(
    X_train,
    input_dim,
    n_latents,
    k,
    seed,
    label,
    concept_dirs=None,
    concept_names=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    sae = TopKSAE(input_dim, n_latents, k).to(DEVICE)
    opt = optim.Adam(sae.parameters(), lr=args.sae_lr)

    return train_sae_with_grad_log(
        X_train=X_train,
        sae=sae,
        optimizer=opt,
        l1_coef=0.0,
        label=label,
        concept_dirs=concept_dirs,
        concept_names=concept_names,
        is_topk=True,
    )


def get_codes(sae, X, batch_size=1024):
    sae.eval()
    X = X.to(DEVICE)
    out = []

    with torch.no_grad():
        for s in range(0, X.shape[0], batch_size):
            _, z = sae(X[s:s + batch_size])
            out.append(z.detach().cpu())

    return torch.cat(out).numpy()


# ============================================================
# METRICS
# ============================================================

def finalize_corr_recovery(Y, Z):
    Y = np.asarray(Y, np.float64)
    Z = np.asarray(Z, np.float64)

    Yc = Y - Y.mean(0, keepdims=True)
    Zc = Z - Z.mean(0, keepdims=True)

    yn = np.sqrt((Yc ** 2).sum(0)) + 1e-12
    zn = np.sqrt((Zc ** 2).sum(0)) + 1e-12

    corr = np.abs((Yc.T @ Zc) / (yn[:, None] * zn[None, :]))

    return corr.max(1), corr.argmax(1)


def one_hot(chosen, n):
    Y = np.zeros((len(chosen), n), dtype=np.float32)
    Y[np.arange(len(chosen)), chosen] = 1.0
    return Y


def make_recovery_df(
    recovery,
    best_latent,
    condition,
    direction_group,
    layer,
    seed,
    signal_scale,
    sae_latents,
    l1_coef,
    direction_names,
    snrs,
):
    return pd.DataFrame({
        "direction_group": direction_group,
        "layer": layer,
        "seed": seed,
        "signal_scale": signal_scale,
        "sae_latents": sae_latents,
        "l1_coef": l1_coef,
        "condition": condition,
        "concept_id": np.arange(len(direction_names)),
        "concept": direction_names,
        "snr": snrs,
        "recovery_score": recovery,
        "best_latent": best_latent,
    })


def summarize_condition(df):
    return {
        "direction_group": df["direction_group"].iloc[0],
        "layer": int(df["layer"].iloc[0]),
        "seed": int(df["seed"].iloc[0]),
        "signal_scale": float(df["signal_scale"].iloc[0]),
        "sae_latents": int(df["sae_latents"].iloc[0]),
        "l1_coef": float(df["l1_coef"].iloc[0]),
        "condition": df["condition"].iloc[0],
        "mean_recovery": float(df["recovery_score"].mean()),
        "median_recovery": float(df["recovery_score"].median()),
        "min_recovery": float(df["recovery_score"].min()),
        "max_recovery": float(df["recovery_score"].max()),
        "frac_recovered": float((df["recovery_score"] >= args.recovery_threshold).mean()),
        "mean_snr": float(df["snr"].mean()),
    }


# ============================================================
# GRADIENT LOG PLOT PER RUN
# ============================================================

def plot_gradient_log(grad_logs, direction_group, run_dir, l1_coef):
    if not grad_logs:
        return

    conditions_local = list(grad_logs.keys())
    fig, axes = plt.subplots(2, len(conditions_local), figsize=(6 * len(conditions_local), 10))

    if len(conditions_local) == 1:
        axes = axes.reshape(2, 1)

    for ci, cond in enumerate(conditions_local):
        df = grad_logs[cond]
        if df.empty:
            continue

        steps = sorted(df["step"].unique())

        ax_top = axes[0, ci]
        mean_gp = df.groupby("step")["max_grad_proj"].mean()
        ax_top.plot(mean_gp.index, mean_gp.values, linewidth=2, label="decoder grad proj")

        ax_top.axhline(l1_coef, color="black", ls="--", lw=1.2, label=f"λ={l1_coef}")
        ax_top.set_yscale("log")
        ax_top.set_title(f"{cond}\nDecoder-gradient projection", fontsize=10, fontweight="bold")
        ax_top.set_xlabel("Step")
        ax_top.set_ylabel("Max raw decoder-grad proj")
        ax_top.legend(fontsize=7)
        ax_top.grid(alpha=0.3)

        ax_bot = axes[1, ci]
        mean_cos = df.groupby("step")["max_dec_cos"].mean()
        ax_bot.plot(mean_cos.index, mean_cos.values, linewidth=2, label="decoder allocation")

        ax_bot.set_title(f"{cond}\nDecoder alignment", fontsize=10, fontweight="bold")
        ax_bot.set_xlabel("Step")
        ax_bot.set_ylabel("Max decoder cosine")
        ax_bot.legend(fontsize=7)
        ax_bot.grid(alpha=0.3)

    plt.suptitle(
        f"Decoder-gradient and allocation dynamics\n"
        f"group={direction_group} | λ={l1_coef}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    plot_path = run_dir / "gradient_dominance_gpt2.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved gradient plot: {plot_path}")


# ============================================================
# RUN ONE SETTING
# ============================================================

def run_one_setting(
    direction_group,
    direction_names,
    snrs,
    target_dirs,
    H_bg,
    nuisance_dirs,
    layer,
    seed,
    signal_scale,
    sae_latents,
    l1_coef,
):
    print("\n" + "#" * 80)
    print(
        f"RUN | group={direction_group} | layer={layer} | seed={seed} "
        f"| scale={signal_scale} | K={sae_latents}"
    )
    print("#" * 80)

    run_dir = OUT_DIR / f"{direction_group}_l{layer}_s{seed}_scale{signal_scale}_k{sae_latents}"
    run_dir.mkdir(parents=True, exist_ok=True)

    H_tr, Y_tr, _ = generate_mixed_dataset(
        H_bg,
        target_dirs,
        nuisance_dirs,
        args.n_train,
        signal_scale,
        seed + 1,
    )
    H_te, Y_te, _ = generate_mixed_dataset(
        H_bg,
        target_dirs,
        nuisance_dirs,
        args.n_test,
        signal_scale,
        seed + 2,
    )
    D_tr, ch_tr = generate_delta_dataset(
        target_dirs,
        args.n_train,
        signal_scale,
        seed + 3,
    )
    D_te, ch_te = generate_delta_dataset(
        target_dirs,
        args.n_test,
        signal_scale,
        seed + 4,
    )
    RP_tr, RP_te = make_random_pair_deltas(H_tr, H_te, seed)

    input_dim = H_tr.shape[1]
    n_dirs = len(direction_names)

    # Log target directions for this direction group.
    # These are the directions whose recovery/allocation we evaluate.
    log_dirs = target_dirs
    log_names = direction_names

    std_sae, gl_std = train_standard_sae(
        H_tr,
        input_dim,
        sae_latents,
        l1_coef,
        seed + 10,
        f"std_L1_{direction_group}_l{layer}_s{seed}",
        concept_dirs=log_dirs,
        concept_names=log_names,
    )

    topk_sae, gl_topk = train_topk_sae(
        H_tr,
        input_dim,
        sae_latents,
        args.topk_k,
        seed + 20,
        f"topK{args.topk_k}_{direction_group}_l{layer}_s{seed}",
        concept_dirs=log_dirs,
        concept_names=log_names,
    )

    delta_sae, gl_delta = train_standard_sae(
        D_tr,
        input_dim,
        sae_latents,
        l1_coef,
        seed + 100,
        f"delta_{direction_group}_l{layer}_s{seed}",
        concept_dirs=log_dirs,
        concept_names=log_names,
    )

    rp_sae, gl_rp = train_standard_sae(
        RP_tr,
        input_dim,
        sae_latents,
        l1_coef,
        seed + 200,
        f"randpair_{direction_group}_l{layer}_s{seed}",
        concept_dirs=log_dirs,
        concept_names=log_names,
    )

    grad_logs = {
        "std_L1": gl_std,
        f"topK{args.topk_k}": gl_topk,
        "delta": gl_delta,
        "rand_pair": gl_rp,
    }

    for cond_name, gl_df in grad_logs.items():
        if not gl_df.empty:
            gl_df.to_csv(run_dir / f"grad_log_{cond_name}.csv", index=False)

    plot_gradient_log(grad_logs, direction_group, run_dir, l1_coef)

    Z_std = get_codes(std_sae, H_te)
    Z_topk = get_codes(topk_sae, H_te)
    Z_delta = get_codes(delta_sae, D_te)
    Z_rp = get_codes(rp_sae, RP_te)

    rng = np.random.default_rng(seed)

    std_rec, std_best = finalize_corr_recovery(Y_te.numpy(), Z_std)
    topk_rec, topk_best = finalize_corr_recovery(Y_te.numpy(), Z_topk)
    delta_rec, delta_best = finalize_corr_recovery(one_hot(ch_te, n_dirs), Z_delta)
    rl_rec, rl_best = finalize_corr_recovery(one_hot(rng.permutation(ch_te), n_dirs), Z_delta)
    rp_rec, rp_best = finalize_corr_recovery(
        one_hot(rng.integers(0, n_dirs, len(ch_te)), n_dirs),
        Z_rp,
    )

    kw = dict(
        direction_group=direction_group,
        layer=layer,
        seed=seed,
        signal_scale=signal_scale,
        sae_latents=sae_latents,
        l1_coef=l1_coef,
        direction_names=direction_names,
        snrs=snrs,
    )

    std_df = make_recovery_df(
        std_rec,
        std_best,
        f"standard_L1_K{sae_latents}_mixed",
        **kw,
    )
    topk_df = make_recovery_df(
        topk_rec,
        topk_best,
        f"topK{args.topk_k}_K{sae_latents}_mixed",
        **kw,
    )
    delta_df = make_recovery_df(
        delta_rec,
        delta_best,
        f"delta_K{sae_latents}",
        **kw,
    )
    rl_df = make_recovery_df(
        rl_rec,
        rl_best,
        "random_label_control",
        **kw,
    )
    rp_df = make_recovery_df(
        rp_rec,
        rp_best,
        "random_pair_delta_control",
        **kw,
    )

    per_concept_df = pd.concat(
        [std_df, topk_df, delta_df, rl_df, rp_df],
        ignore_index=True,
    )
    summary_df = pd.DataFrame([
        summarize_condition(d)
        for d in [std_df, topk_df, delta_df, rl_df, rp_df]
    ])

    per_concept_df.to_csv(run_dir / "per_concept_recovery.csv", index=False)
    summary_df.to_csv(run_dir / "summary.csv", index=False)

    print("\nSummary:")
    print(summary_df[["condition", "mean_recovery", "frac_recovered"]])

    del std_sae, topk_sae, delta_sae, rp_sae
    del H_tr, H_te, D_tr, D_te, RP_tr, RP_te
    del Z_std, Z_topk, Z_delta, Z_rp

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary_df, per_concept_df


# ============================================================
# MAIN SWEEP
# ============================================================

all_summary = []
all_per_concept = []

for layer in args.layers:
    H_bg, nuisance_dirs, direction_assets = get_or_create_layer_assets(layer)

    for direction_group, (target_dirs, meta_df) in direction_assets.items():
        direction_names = meta_df["target"].tolist()
        snrs = meta_df["snr"].tolist() if "snr" in meta_df.columns else [0.0] * len(direction_names)

        group_dir = OUT_DIR / f"layer_{layer}" / f"group_{direction_group}"
        group_dir.mkdir(parents=True, exist_ok=True)
        meta_df.to_csv(group_dir / "direction_metadata.csv", index=False)

        sweep_configs = [
            (scale, k, l1, seed)
            for scale in args.rare_scales
            for k in args.sae_latents
            for l1 in args.l1_list
            for seed in args.seeds
        ]

        for scale, k, l1, seed in tqdm(
            sweep_configs,
            desc=f"layer={layer} group={direction_group}",
        ):
            s_df, pc_df = run_one_setting(
                direction_group=direction_group,
                direction_names=direction_names,
                snrs=snrs,
                target_dirs=target_dirs,
                H_bg=H_bg,
                nuisance_dirs=nuisance_dirs,
                layer=layer,
                seed=seed,
                signal_scale=scale,
                sae_latents=k,
                l1_coef=l1,
            )

            all_summary.append(s_df)
            all_per_concept.append(pc_df)

            pd.concat(all_summary, ignore_index=True).to_csv(
                OUT_DIR / "ALL_SUMMARY_PARTIAL.csv",
                index=False,
            )
            pd.concat(all_per_concept, ignore_index=True).to_csv(
                OUT_DIR / "ALL_PER_CONCEPT_PARTIAL.csv",
                index=False,
            )

    del H_bg, nuisance_dirs, direction_assets

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# AGGREGATE SUMMARY
# ============================================================

all_summary_df = pd.concat(all_summary, ignore_index=True)
all_per_concept_df = pd.concat(all_per_concept, ignore_index=True)

all_summary_df.to_csv(OUT_DIR / "ALL_SUMMARY.csv", index=False)
all_per_concept_df.to_csv(OUT_DIR / "ALL_PER_CONCEPT.csv", index=False)

agg = (
    all_summary_df
    .groupby([
        "direction_group",
        "layer",
        "signal_scale",
        "sae_latents",
        "l1_coef",
        "condition",
    ])
    .agg(
        mean_recovery_mean=("mean_recovery", "mean"),
        mean_recovery_std=("mean_recovery", "std"),
        frac_recovered_mean=("frac_recovered", "mean"),
        frac_recovered_std=("frac_recovered", "std"),
        mean_snr=("mean_snr", "mean"),
    )
    .reset_index()
)

agg.to_csv(OUT_DIR / "AGGREGATE_RESULTS.csv", index=False)

print("\nAGGREGATE SUMMARY:")
print(agg.to_string())


# ============================================================
# SUMMARY PLOTS
# ============================================================

if not args.no_plots:
    groups = [g for g in ["weak", "strong", "random"] if g in args.direction_groups]

    # Plot 1: Recovery vs signal scale
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), squeeze=False)

    for ax, group in zip(axes.flatten(), groups):
        df_g = agg[(agg["direction_group"] == group) & (agg["sae_latents"] == 512)]

        for pat in ["standard_L1", "topK", "delta_K", "random_pair"]:
            df_c = df_g[df_g["condition"].str.contains(pat, regex=False)]

            if pat == "delta_K":
                df_c = df_c[df_c["condition"].str.startswith("delta_K")]

            if df_c.empty:
                continue

            df_c = df_c.sort_values("signal_scale")

            ax.plot(
                df_c["signal_scale"],
                df_c["mean_recovery_mean"],
                marker="o",
                lw=2,
                label=df_c["condition"].iloc[0],
            )

            ax.fill_between(
                df_c["signal_scale"],
                df_c["mean_recovery_mean"] - df_c["mean_recovery_std"].fillna(0),
                df_c["mean_recovery_mean"] + df_c["mean_recovery_std"].fillna(0),
                alpha=0.15,
            )

        ax.set_xlabel("Signal scale")
        ax.set_ylabel("Mean recovery")
        ax.set_title(f"{group} | K=512")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    plt.suptitle(
        "Recovery vs signal scale\nGPT2-Small layer 8",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "recovery_vs_scale.png", dpi=200)
    plt.close()

    # Plot 2: Capacity comparison
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), squeeze=False)

    for ax, group in zip(axes.flatten(), groups):
        df_g = agg[
            (agg["direction_group"] == group)
            & (np.isclose(agg["signal_scale"], 0.05))
        ]

        for pat in ["standard_L1", "topK", "delta_K"]:
            df_c = df_g[df_g["condition"].str.contains(pat, regex=False)]

            if pat == "delta_K":
                df_c = df_c[df_c["condition"].str.startswith("delta_K")]

            if df_c.empty:
                continue

            df_c = df_c.sort_values("sae_latents")

            ax.plot(
                df_c["sae_latents"],
                df_c["mean_recovery_mean"],
                marker="o",
                lw=2,
                label=pat,
            )

            ax.fill_between(
                df_c["sae_latents"],
                df_c["mean_recovery_mean"] - df_c["mean_recovery_std"].fillna(0),
                df_c["mean_recovery_mean"] + df_c["mean_recovery_std"].fillna(0),
                alpha=0.15,
            )

        ax.set_xscale("log")
        ax.set_xticks(args.sae_latents)
        ax.set_xticklabels([str(k) for k in args.sae_latents])
        ax.set_xlabel("SAE latents")
        ax.set_ylabel("Mean recovery")
        ax.set_title(f"{group} | scale=0.05")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(
        "Does more capacity help?\nGPT2-Small layer 8",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "recovery_vs_sae_size.png", dpi=200)
    plt.close()

print(f"\nDone. Results in: {OUT_DIR}")
print("\nKey outputs:")
print("  ALL_SUMMARY.csv")
print("  ALL_PER_CONCEPT.csv")
print("  AGGREGATE_RESULTS.csv")
print("  recovery_vs_scale.png")
print("  recovery_vs_sae_size.png")
print("  {run_dir}/grad_log_*.csv")
print("  {run_dir}/gradient_dominance_gpt2.png")