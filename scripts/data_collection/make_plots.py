"""
Generate all figures for the Research Paper Recommender writeup / README.

Every number is HARDCODED from the final runs — these results are locked, and
the training log survives only as a saved snapshot (see PROVENANCE below), so
there is nothing to re-derive and nothing to re-run.

PROVENANCE (important — two training runs exist, don't mix them):
  * The EVALUATED model is user_history_encoder_best.pt, produced by the
    NORMALIZED + HARD-NEGATIVE retrain. Verified: best.pt != last.pt, so early
    stopping fired and `best` is an earlier epoch. Its log is TRAINING_LOG below
    (val loss ~0.32 scale — hard negatives make the triplet task much harder).
  * An earlier abandoned run (random negatives, val ~0.065 scale, 8 epochs, no
    early stop) is NOT the model behind any result here. Ignore it.

Outputs 4 PNGs into figures/:
  1. coldstart_convergence.png  -- THE figure. 4 methods x input levels + popularity floor.
  2. training_curves.png        -- train falls / val RISES: genuine overfitting, early stop.
  3. popularity_bias.png        -- citation-quartile skew, model vs mean-pool vs corpus.
  4. main_eval.png              -- headline benchmark: model vs 3 baselines.

Style: clean-academic, readable on GitHub. Colours are SEMANTIC and consistent
across every figure: blue = mean-pool (the winner), red = attention tower (the
loser), grey = category / baselines. Learn the code once, read every plot fast.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

C_MPOOL = "#2166ac"   # mean-pool — the winner
C_TOWER = "#b2182b"   # attention tower — the loser
C_CAT   = "#999999"   # category — the weak signal
C_POP   = "#666666"   # popularity floor / baselines


# =============================================================================
# 1. COLD-START CONVERGENCE  (the centerpiece)
#    10,000 held-out users, history_len=5, >=3 categories, 1 held-out positive.
#    Seed-paper sweeps 1-4 papers; category sweeps 1-3 (diversity-capped).
# =============================================================================
def fig_coldstart():
    lv_p = [1, 2, 3, 4]
    lv_c = [1, 2, 3]

    data = {
        "Recall@10": {
            "seed_mpool": [0.0173, 0.0184, 0.0198, 0.0218],
            "seed_tower": [0.0084, 0.0095, 0.0125, 0.0122],
            "cat_mpool":  [0.0000, 0.0003, 0.0002],
            "cat_tower":  [0.0001, 0.0001, 0.0002],
            "pop_floor":  0.0125,
        },
        "Recall@100": {
            "seed_mpool": [0.0567, 0.0659, 0.0717, 0.0773],
            "seed_tower": [0.0384, 0.0492, 0.0565, 0.0640],
            "cat_mpool":  [0.0008, 0.0015, 0.0014],
            "cat_tower":  [0.0011, 0.0009, 0.0012],
            "pop_floor":  0.0391,
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (metric, d) in zip(axes, data.items()):
        ax.plot(lv_p, d["seed_mpool"], "o-", color=C_MPOOL, lw=2.2, ms=6,
                label="Seed-paper + mean-pool")
        ax.plot(lv_p, d["seed_tower"], "s-", color=C_TOWER, lw=2.2, ms=6,
                label="Seed-paper + attention tower")
        ax.plot(lv_c, d["cat_mpool"], "^--", color=C_CAT, lw=1.6, ms=5,
                label="Category + mean-pool")
        ax.plot(lv_c, d["cat_tower"], "v--", color=C_CAT, lw=1.6, ms=5, alpha=0.6,
                label="Category + attention tower")
        ax.axhline(d["pop_floor"], color=C_POP, ls=":", lw=2,
                   label=f"Popularity floor ({d['pop_floor']:.4f})")
        ax.set_xlabel("Cold-start signal (number of inputs)")
        ax.set_ylabel(metric)
        ax.set_title(f"Cold-start convergence — {metric}")
        ax.set_xticks(lv_p)
        ax.set_ylim(bottom=0)

    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Cold-start: precise signal (papers) vs coarse signal (fields)\n"
                 "10,000 held-out users · 5-paper histories · 1 held-out positive",
                 fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/coldstart_convergence.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote coldstart_convergence.png")


# =============================================================================
# 2. TRAINING CURVES  (the overfit story — REAL divergence)
#    Hard-negative retrain. Train falls, val RISES. Best val = epoch 1.
#    Early stopping fires at epoch 4 (patience=3 exhausted after epoch 1).
#    Verified independently: best.pt != last.pt, so early stop did trigger.
# =============================================================================
def fig_training_curves():
    epochs = [1, 2, 3, 4]
    train  = [0.04975068615469615, 0.04582982505340049,
              0.044847754225224405, 0.044430208477908455]
    val    = [0.3142716463130948,  0.3205049301438644,
              0.32118169692034687, 0.3213635507073543]
    best_epoch = 1   # lowest val loss — the checkpoint that was evaluated

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # left: both curves on separate y-axes (very different scales)
    ax1.plot(epochs, train, "o-", color=C_MPOOL, lw=2.2, ms=7, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color=C_MPOOL)
    ax1.tick_params(axis="y", labelcolor=C_MPOOL)
    ax1.set_xticks(epochs)

    ax1b = ax1.twinx()
    ax1b.plot(epochs, val, "s-", color=C_TOWER, lw=2.2, ms=7, label="Validation loss")
    ax1b.set_ylabel("Validation loss", color=C_TOWER)
    ax1b.tick_params(axis="y", labelcolor=C_TOWER)
    ax1b.spines["top"].set_visible(False)
    ax1b.grid(False)

    ax1.axvline(best_epoch, color="#444444", ls="--", lw=1.3, alpha=0.7)
    ax1.text(best_epoch + 0.08, max(train) * 0.995,
             f"best val (epoch {best_epoch})\n← checkpoint evaluated",
             fontsize=8.5, va="top", color="#444444")
    ax1.set_title("Train loss falls while validation loss rises")
    ax1.legend(handles=[
        plt.Line2D([], [], color=C_MPOOL, marker="o", lw=2.2, label="Train loss"),
        plt.Line2D([], [], color=C_TOWER, marker="s", lw=2.2, label="Validation loss"),
    ], loc="center right", fontsize=9)

    # right: normalized to epoch 1 — makes the DIVERGENCE unmistakable
    tr_n = [t / train[0] for t in train]
    va_n = [v / val[0] for v in val]
    ax2.plot(epochs, tr_n, "o-", color=C_MPOOL, lw=2.2, ms=7, label="Train loss")
    ax2.plot(epochs, va_n, "s-", color=C_TOWER, lw=2.2, ms=7, label="Validation loss")
    ax2.axhline(1.0, color="#888888", ls=":", lw=1.2)
    ax2.fill_between(epochs, tr_n, va_n, color="#cccccc", alpha=0.35,
                     label="Generalization gap")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss relative to epoch 1")
    ax2.set_title("Divergence: −10.7% train vs +2.3% validation")
    ax2.set_xticks(epochs)
    ax2.legend(loc="center left", fontsize=9)

    fig.suptitle("Overfitting, as designed — the correlated sliding-window data "
                 "produced genuine train/val divergence\n"
                 "Hard-negative retrain · early stopping fired at epoch 4 "
                 "(patience=3) · epoch-1 checkpoint used for all results",
                 fontsize=11.5, y=1.10)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/training_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote training_curves.png")


# =============================================================================
# 3. POPULARITY BIAS  (citation-quartile skew)
#    NOTE: only Q1 and Q4 were recorded in the audit. Q2/Q3 are INFERRED to sum
#    to 100% (split evenly) and the figure says so. Swap in real values if found.
# =============================================================================
def fig_popularity_bias():
    quartiles = ["Q1\n(lowest cited)", "Q2", "Q3", "Q4\n(most cited)"]
    model    = [14.3, 25.45, 25.45, 34.8]   # Q2/Q3 inferred
    meanpool = [17.7, 24.95, 24.95, 32.4]   # Q2/Q3 inferred
    corpus   = [25.0, 25.1,  25.0,  24.9]   # ~flat baseline

    x = np.arange(len(quartiles))
    w = 0.26

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - w, corpus,   w, label="Corpus baseline (flat)", color="#cccccc",
           edgecolor="white")
    ax.bar(x,     meanpool, w, label="Mean-pool", color=C_MPOOL, edgecolor="white")
    ax.bar(x + w, model,    w, label="Model (attention)", color=C_TOWER,
           edgecolor="white")
    ax.axhline(25, color="#444444", ls="--", lw=1.2, alpha=0.7)
    ax.text(3.42, 25.7, "unbiased (25%)", fontsize=8.5, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(quartiles)
    ax.set_ylabel("% of top-10 recommendations")
    ax.set_title("Popularity bias: both recommenders over-surface highly-cited papers\n"
                 "(bias originates in embedding geometry, not architecture)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 42)

    fig.text(0.5, -0.06,
             "Q1/Q4 measured; Q2/Q3 inferred to sum to 100%. "
             "Citation quartile edges: 8 / 35 / 105. 2,000 test users.",
             ha="center", fontsize=8, color="#666666")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/popularity_bias.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote popularity_bias.png")


# =============================================================================
# 4. MAIN EVAL  (the headline benchmark)
# =============================================================================
def fig_main_eval():
    methods = ["Mean-pool\n(no parameters)", "Model\n(attention)", "Popularity", "Random"]
    recall  = [0.0225, 0.0140, 0.0075, 0.0000]
    ndcg    = [0.0113, 0.0074, 0.0038, 0.0000]
    colors  = [C_MPOOL, C_TOWER, C_POP, "#cccccc"]

    x = np.arange(len(methods))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar(x - w/2, recall, w, color=colors, edgecolor="white")
    b2 = ax.bar(x + w/2, ndcg,   w, color=colors, alpha=0.55, edgecolor="white")

    for bars, vals in ((b1, recall), (b2, ndcg)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.0004, f"{v:.4f}",
                    ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title("Main benchmark: the no-parameter mean-pool baseline beats\n"
                 "the trained attention model")
    ax.set_ylim(0, max(recall) * 1.30)
    ax.legend(handles=[Patch(facecolor="#888888", label="Recall@10"),
                       Patch(facecolor="#888888", alpha=0.55, label="NDCG@10")],
              loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/main_eval.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote main_eval.png")


if __name__ == "__main__":
    fig_coldstart()
    fig_training_curves()
    fig_popularity_bias()
    fig_main_eval()
    print(f"\nall figures -> ./{OUT_DIR}/")