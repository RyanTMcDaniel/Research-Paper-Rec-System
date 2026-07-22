"""
STAGE 2: why do all 23 category centroids sit within cosine distance 0.003-0.042?

Four competing explanations, each with a decisive measurement:

  H1  the 768->256 projection destroyed the field structure.
      -> rebuild the SAME centroids in raw 768-d and compare the pairwise
         cosine-distance matrix against 256-d. Same papers, same construction,
         only the dimension differs. If 768-d is meaningfully more spread, the
         projection did it. If both are collapsed, H1 is dead.
      -> plus a forensic sub-test: fit cached_256 ~ affine(pooled_768) and check
         whether the trained paper_encoder.projection from the checkpoint actually
         reproduces the cache. build_embedding_cache.py never loads a checkpoint,
         so this asks whether the corpus matrix was built with a RANDOM projection.

  H2  mean-pooled SciBERT is anisotropic out of the box (the known BERT pathology).
      -> mean pairwise cosine sim over random paper pairs, + covariance eigenspectrum
         (top-k explained variance, dims needed for 90%). Reported for 768 AND 256.

  H3  the centroids are meaningless because the corpus is ~94% CS.
      -> per category: papers in the centroid, how many hold it as PRIMARY field vs
         merely one tag in the multi-label list, and what fraction also carry a CS tag.
      -> control: recompute centroids from primary-field papers only, keep categories
         with enough of them, and re-report the distances among survivors.

  H4  frozen encoder + L2 NN is just a weak metric space.
      -> nearest-neighbor cosine-distance distribution, and the ratio of mean
         intra-category to mean inter-category distance under primary-field labels.
         intra ~= inter means the space does not encode field structure at all.

Reuses centroid_calc.py (get_cats / build_pairs / compute_centroids -- the EXACT
centroid construction) and evaluate.py (paths, normalized geometry, raw paper index).
Torch only. FAISS is never imported.

Writes a full report to the scratch dir. Touches no doc file.
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "retrieval"))

from evaluate import CORPUS, IDS_PATH, MATRIX_PATH  # noqa: E402
from centroid_calc import build_pairs, compute_centroids, get_cats  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_2T   = os.path.join(REPO_ROOT, "two_tower_model_best.pt")
SCRATCH   = "/private/tmp/claude-501/-Users-ryanmcdaniel/a7fd4cc6-2a49-4466-80b7-a29a78771a2e/scratchpad"
SAMPLE768 = os.path.join(SCRATCH, "sample_768.pt")
REPORT    = os.path.join(SCRATCH, "embedding_space_diagnosis.md")

MIN_PRIMARY = 100    # a category needs this many PRIMARY-field papers to survive the H3 control
N_PAIRS     = 200_000
N_NN_QUERY  = 2000
SEED        = 42

OUT = []
def say(s=""):
    print(s)
    OUT.append(s)


# ---------- shared geometry helpers ----------
def cos_dist_matrix(M):
    """Pairwise cosine DISTANCE between rows of M (numpy)."""
    T = F.normalize(torch.from_numpy(np.asarray(M, dtype=np.float32)), p=2, dim=1)
    return (1.0 - (T @ T.T)).numpy()

def offdiag(D):
    n = D.shape[0]
    return D[~np.eye(n, dtype=bool)]

def spread(D, label):
    o = offdiag(D)
    say(f"  {label:<34} min={o.min():.4f}  median={np.median(o):.4f}  max={o.max():.4f}  mean={o.mean():.4f}")
    return o

def primary_field(cats):
    """Primary field = most frequent tag in the multi-label list; ties -> first seen."""
    c = Counter(cats)
    return c.most_common(1)[0][0] if c else None

def anisotropy(X, label, rng):
    """Mean pairwise cosine sim over random pairs + covariance eigenspectrum."""
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    n = Xt.shape[0]
    Xn = F.normalize(Xt, p=2, dim=1)
    i = rng.integers(0, n, N_PAIRS)
    j = rng.integers(0, n, N_PAIRS)
    keep = i != j
    cos = (Xn[i[keep]] * Xn[j[keep]]).sum(1).numpy()

    Xc = Xt - Xt.mean(0, keepdim=True)          # centered -> covariance spectrum
    cov = (Xc.T @ Xc) / (n - 1)
    ev = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0).numpy()
    evr = ev / ev.sum()
    cum = np.cumsum(evr)
    d90 = int(np.searchsorted(cum, 0.90) + 1)

    say(f"  {label}")
    say(f"    mean pairwise cosine sim (random pairs) : {cos.mean():.4f}  (sd {cos.std():.4f})")
    say(f"    top-1 / top-5 / top-10 explained var    : {evr[0]:.4f} / {evr[:5].sum():.4f} / {evr[:10].sum():.4f}")
    say(f"    dims needed for 90% of variance         : {d90} of {Xt.shape[1]}")
    return cos.mean(), evr, d90


def main():
    rng = np.random.default_rng(SEED)

    # ---------- load everything ----------
    if not os.path.exists(SAMPLE768):
        sys.exit(f"missing {SAMPLE768} -- run diag_build_768_sample.py first")
    # weights_only=False: this is our own scratch file of numpy arrays, not a model ckpt
    raw768 = torch.load(SAMPLE768, map_location="cpu", weights_only=False)
    cache768 = {k: np.asarray(v, dtype=np.float32) for k, v in raw768.items()}

    matrix = np.load(MATRIX_PATH).astype("float32")                     # the live 256-d retrieval space
    paper_ids = [str(p) for p in np.load(IDS_PATH, allow_pickle=True)]
    id_to_idx = {p: i for i, p in enumerate(paper_ids)}
    cache256 = {p: matrix[i] for p, i in id_to_idx.items()}

    corpus = pd.read_parquet(CORPUS)
    corpus["cat_list"] = corpus["s2FieldsOfStudy"].apply(get_cats)      # reused parser
    corpus["primary"] = corpus["cat_list"].apply(primary_field)
    corpus["paperId"] = corpus["paperId"].astype(str)

    say("# Embedding-space diagnosis: why are the 23 category centroids all touching?")
    say()
    say(f"768-d sample: {len(cache768):,} papers (re-run through frozen SciBERT, mean-pooled)")
    say(f"256-d space : {matrix.shape[0]:,} papers x {matrix.shape[1]} (the live retrieval matrix)")
    say()

    # =====================================================================
    # H1 -- did the 256-d projection destroy the field structure?
    # =====================================================================
    say("=" * 78)
    say("## H1 -- the 768->256 projection destroyed the field structure")
    say("=" * 78)
    say()

    # Same papers, same construction, both dims. build_pairs applies the same
    # MIN_CAT_SIZE floor and the same multi-label explode as the real pipeline.
    sub = corpus[corpus["paperId"].isin(cache768.keys())]
    pairs = build_pairs(sub, cache768)
    cent768 = compute_centroids(pairs, cache768)
    cent256 = compute_centroids(pairs, cache256)      # identical papers, identical grouping
    assert list(cent768["category"]) == list(cent256["category"])
    cats = list(cent768["category"])

    D768 = cos_dist_matrix(np.stack(cent768["centroid"].to_numpy()))
    D256 = cos_dist_matrix(np.stack(cent256["centroid"].to_numpy()))

    say(f"Centroids rebuilt on the SAME {len(cats)} categories from the SAME sampled papers,")
    say("so the only thing that changes between these two rows is the dimension.")
    say()
    o768 = spread(D768, "raw 768-d (pre-projection)")
    o256 = spread(D256, "256-d (post-projection, sample)")

    # is the sample faithful to the real full-corpus 256-d centroids?
    pairs_full = build_pairs(corpus, cache256)
    cent256_full = compute_centroids(pairs_full, cache256)
    D256_full = cos_dist_matrix(np.stack(cent256_full["centroid"].to_numpy()))
    spread(D256_full, "256-d (post-projection, FULL corpus)")
    say()
    say(f"  ratio of median spread, 768 vs 256 (sample): "
        f"{np.median(o768) / max(np.median(o256), 1e-12):.2f}x")
    say()

    say("### Full pairwise cosine-distance matrix, raw 768-d")
    say()
    hdr = "  " + " " * 26 + "".join(f"{c[:6]:>8}" for c in cats)
    say(hdr)
    for a, c in enumerate(cats):
        say(f"  {c:<26}" + "".join(f"{D768[a, b]:>8.4f}" for b in range(len(cats))))
    say()
    say("### Full pairwise cosine-distance matrix, 256-d (same papers)")
    say()
    say(hdr)
    for a, c in enumerate(cats):
        say(f"  {c:<26}" + "".join(f"{D256[a, b]:>8.4f}" for b in range(len(cats))))
    say()

    # --- forensic: WHICH projection built the corpus matrix? ---
    say("### Forensic: which linear map actually produced the 256-d cache?")
    say()
    ids_both = [p for p in cache768 if p in cache256]
    P = np.stack([cache768[p] for p in ids_both])          # (n, 768)
    C = np.stack([cache256[p] for p in ids_both])          # (n, 256)

    # least-squares affine fit  C ~ P @ W.T + b
    Pa = np.hstack([P, np.ones((len(P), 1), dtype=np.float32)])
    sol, *_ = np.linalg.lstsq(Pa, C, rcond=None)
    resid = C - Pa @ sol
    r2 = 1.0 - (resid ** 2).sum() / ((C - C.mean(0)) ** 2).sum()
    say(f"  cached_256 ~ affine(pooled_768) least-squares R^2 : {r2:.6f}")
    say("    (R^2 ~ 1.0 => the cache IS an affine map of mean-pooled SciBERT, as designed)")

    sd = torch.load(CKPT_2T, map_location="cpu")
    W = sd["paper_encoder.projection.weight"].numpy()      # (256, 768), TRAINED
    b = sd["paper_encoder.projection.bias"].numpy()
    pred = P @ W.T + b
    pn = F.normalize(torch.from_numpy(pred), p=2, dim=1)
    cn = F.normalize(torch.from_numpy(C), p=2, dim=1)
    cos_ckpt = (pn * cn).sum(1).mean().item()
    say(f"  cosine(cache_256, TRAINED_projection(pooled_768)) : {cos_ckpt:.4f}")
    say("    (~1.0 => cache was built with the trained projection;")
    say("     ~0.0 => cache was built with some OTHER matrix, i.e. the random init in")
    say("     build_embedding_cache.py, which constructs PaperEncoder and never loads a ckpt)")
    say()

    # a random projection preserves cosine geometry (Johnson-Lindenstrauss). show it.
    g = np.random.default_rng(0)
    Wr = g.normal(0, 1 / np.sqrt(768), size=(256, 768)).astype(np.float32)
    Dr = cos_dist_matrix(np.stack(cent768["centroid"].to_numpy()) @ Wr.T)
    spread(Dr, "768-d centroids -> RANDOM 256-d proj")
    say("    (if this looks like the real 256-d row, the projection is geometry-preserving,")
    say("     and the collapse was already present in 768-d)")
    say()

    # =====================================================================
    # H2 -- is mean-pooled SciBERT anisotropic out of the box?
    # =====================================================================
    say("=" * 78)
    say("## H2 -- mean-pooled SciBERT is anisotropic out of the box")
    say("=" * 78)
    say()
    say("Centroid-independent. Straight at the paper vectors.")
    say()
    cos768, evr768, d90_768 = anisotropy(P, "raw 768-d mean-pooled SciBERT", rng)
    say()
    ridx = rng.choice(matrix.shape[0], size=min(20000, matrix.shape[0]), replace=False)
    cos256, evr256, d90_256 = anisotropy(matrix[ridx], "256-d projected space", rng)
    say()
    say("  Reference: an isotropic space has mean pairwise cosine ~0.0 and needs ~90% of its")
    say("  dims for 90% of variance. The BERT anisotropy pathology = high mean cosine (a narrow")
    say("  cone) + variance concentrated in a handful of directions.")
    say()

    # =====================================================================
    # H3 -- are the centroids just re-averaging the CS corpus?
    # =====================================================================
    say("=" * 78)
    say("## H3 -- the centroids are meaningless because the corpus is ~94% CS")
    say("=" * 78)
    say()

    # NOTE: build_pairs explodes s2FieldsOfStudy WITHOUT dedup, and that column repeats a
    # category when both the 'external' and the 's2-fos-model' source assign it. So the real
    # pipeline averages such papers TWICE into their own centroid. n_rows below is what the
    # centroid actually saw; n_papers is the unique-paper truth. The gap is a live bug.
    tagged = (corpus[["paperId", "cat_list", "primary"]]
              .explode("cat_list").dropna(subset=["cat_list"])
              .rename(columns={"cat_list": "category"}))
    uniq = tagged.drop_duplicates(["paperId", "category"])
    has_cs = corpus.set_index("paperId")["cat_list"].apply(lambda L: "Computer Science" in L)
    uniq = uniq.assign(has_cs=uniq["paperId"].map(has_cs))

    cs_frac_corpus = has_cs.mean()
    say(f"Corpus papers carrying a CS tag: {cs_frac_corpus:.1%}  ({int(has_cs.sum()):,} / {len(has_cs):,})")
    say(f"Tag rows fed to centroids: {len(tagged):,} across {len(corpus):,} papers "
        f"-> {len(tagged) - len(uniq):,} are DUPLICATE (paper, category) rows.")
    say("  (centroid_calc.build_pairs never dedups, so those papers are averaged in twice)")
    say()
    say("Per category: is its centroid built from ITS OWN papers, or from CS papers wearing its tag?")
    say("  n_rows   = rows the centroid actually averaged (with the duplicate-tag double-count)")
    say("  n_papers = unique papers carrying the tag")
    say("  n_primary= unique papers whose PRIMARY field is this category")
    say()
    h = (f"  {'category':<28}{'n_rows':>9}{'n_papers':>9}{'n_primary':>10}"
         f"{'primary%':>10}{'also_CS%':>10}")
    say(h)
    say("  " + "-" * (len(h) - 2))
    rows = []
    for cat, g_ in uniq.groupby("category"):
        n_pap = len(g_)
        n_rows_ = int((tagged["category"] == cat).sum())
        n_pri = int((g_["primary"] == cat).sum())
        also_cs = float(g_["has_cs"].mean())
        rows.append({"category": cat, "n_rows": n_rows_, "n_papers": n_pap, "n_primary": n_pri,
                     "primary_pct": n_pri / n_pap, "also_cs": also_cs})
    rows = pd.DataFrame(rows).sort_values("n_papers", ascending=False)
    for _, r in rows.iterrows():
        say(f"  {r['category']:<28}{r['n_rows']:>9,}{r['n_papers']:>9,}{r['n_primary']:>10,}"
            f"{r['primary_pct']:>9.1%}{r['also_cs']:>10.1%}")
    say()

    # --- control: centroids from PRIMARY-field papers only ---
    say(f"### Control: centroids rebuilt from PRIMARY-field papers only (>= {MIN_PRIMARY} such papers)")
    say()
    surv = rows[rows["n_primary"] >= MIN_PRIMARY]["category"].tolist()
    dropped = sorted(set(rows["category"]) - set(surv))
    say(f"  survivors ({len(surv)}): {', '.join(sorted(surv))}")
    say(f"  dropped   ({len(dropped)}): {', '.join(dropped) if dropped else '(none)'}")
    say()

    pri = corpus[corpus["primary"].isin(surv)]
    pc, pcats = [], []
    for cat, g_ in pri.groupby("primary"):
        idx = [id_to_idx[p] for p in g_["paperId"] if p in id_to_idx]
        if not idx:
            continue
        pcats.append(cat)
        pc.append(matrix[idx].mean(0))
    Dpri = cos_dist_matrix(np.stack(pc))
    say("  pairwise cosine distance among PRIMARY-only centroids (256-d, full corpus):")
    say()
    hdr2 = "  " + " " * 26 + "".join(f"{c[:6]:>8}" for c in pcats)
    say(hdr2)
    for a, c in enumerate(pcats):
        say(f"  {c:<26}" + "".join(f"{Dpri[a, b]:>8.4f}" for b in range(len(pcats))))
    say()
    o_pri = spread(Dpri, "PRIMARY-only centroids (256-d)")
    say(f"  vs tag-based centroids median {np.median(offdiag(D256_full)):.4f} "
        f"-> {np.median(o_pri) / max(np.median(offdiag(D256_full)), 1e-12):.2f}x")
    say()

    # the headline README claim, re-derived
    say("### The 0.9998 claim: CS centroid vs corpus mean")
    say()
    corpus_mean = matrix.mean(0)
    cs_ids = corpus[corpus["cat_list"].apply(lambda L: "Computer Science" in L)]["paperId"]
    cs_idx = [id_to_idx[p] for p in cs_ids if p in id_to_idx]
    cs_cent = matrix[cs_idx].mean(0)

    def cs(a, b_):
        a = torch.from_numpy(np.asarray(a, dtype=np.float32))
        b_ = torch.from_numpy(np.asarray(b_, dtype=np.float32))
        return F.cosine_similarity(a.unsqueeze(0), b_.unsqueeze(0)).item()

    say(f"  cos(CS centroid [tag-based], corpus mean)      : {cs(cs_cent, corpus_mean):.4f}   <- the README number")
    say(f"  CS-tagged share of corpus                      : {cs_frac_corpus:.1%}")
    say()
    say("  Is 0.9998 a geometric finding or an averaging artifact? Compare against control centroids")
    say("  built from RANDOM papers -- no field semantics at all, matched for size:")
    for frac in (cs_frac_corpus, 0.50, 0.10, 0.01):
        k = max(2, int(frac * matrix.shape[0]))
        ridx2 = rng.choice(matrix.shape[0], size=k, replace=False)
        say(f"    cos(mean of {frac:>6.1%} RANDOM papers, corpus mean) : "
            f"{cs(matrix[ridx2].mean(0), corpus_mean):.4f}  (n={k:,})")
    say()
    say("  and every real category centroid vs the corpus mean:")
    for _, r in rows.iterrows():
        cat = r["category"]
        ids_ = uniq[uniq["category"] == cat]["paperId"]
        idx_ = [id_to_idx[p] for p in ids_ if p in id_to_idx]
        if not idx_:
            continue
        say(f"    cos({cat:<26}, corpus mean) = {cs(matrix[idx_].mean(0), corpus_mean):.4f}"
            f"   (n={len(idx_):,})")
    say()

    # =====================================================================
    # H4 -- is this just a weak metric space?
    # =====================================================================
    say("=" * 78)
    say("## H4 -- frozen encoder + L2 NN is a weak metric space")
    say("=" * 78)
    say()

    Mn = F.normalize(torch.from_numpy(matrix), p=2, dim=1)
    q = rng.choice(matrix.shape[0], size=N_NN_QUERY, replace=False)
    nn1, nn10 = [], []
    for s in range(0, len(q), 256):
        blk = Mn[q[s:s + 256]]
        sim = blk @ Mn.T
        sim[torch.arange(len(blk)), torch.from_numpy(q[s:s + 256])] = -2.0   # drop self
        top = torch.topk(sim, 10, dim=1).values
        nn1.append((1 - top[:, 0]).numpy())
        nn10.append((1 - top).mean(1).numpy())
    nn1 = np.concatenate(nn1); nn10 = np.concatenate(nn10)

    rp = rng.integers(0, matrix.shape[0], N_PAIRS)
    rq = rng.integers(0, matrix.shape[0], N_PAIRS)
    k_ = rp != rq
    rand_d = (1 - (Mn[rp[k_]] * Mn[rq[k_]]).sum(1)).numpy()

    say("  nearest-neighbour cosine distance (256-d retrieval space, n=%d queries):" % N_NN_QUERY)
    for p_ in (1, 25, 50, 75, 99):
        say(f"    NN-1  p{p_:<2} = {np.percentile(nn1, p_):.4f}")
    say(f"    NN-1  mean = {nn1.mean():.4f}   |  mean over top-10 = {nn10.mean():.4f}")
    say(f"    RANDOM pair mean distance = {rand_d.mean():.4f}")
    say(f"    separation (random - NN1)  = {rand_d.mean() - nn1.mean():.4f}")
    say("    -> if a paper's nearest neighbour is barely closer than a random paper,")
    say("       the metric carries almost no usable local structure.")
    say()

    say(f"  intra- vs inter-category distance, PRIMARY-field labels only ({len(surv)} survivors):")
    say()
    intra_all, inter_all = [], []
    say(f"  {'category':<28}{'n':>8}{'intra':>9}{'inter':>9}{'ratio':>8}")
    say("  " + "-" * 60)
    cent_by_cat = {c: pc[i] for i, c in enumerate(pcats)}
    for cat in pcats:
        g_ = pri[pri["primary"] == cat]
        idx_ = [id_to_idx[p] for p in g_["paperId"] if p in id_to_idx]
        take = idx_ if len(idx_) <= 400 else list(rng.choice(idx_, 400, replace=False))
        V = F.normalize(torch.from_numpy(matrix[take]), p=2, dim=1)
        S = V @ V.T
        n_ = len(take)
        iu = torch.triu_indices(n_, n_, offset=1)
        intra = (1 - S[iu[0], iu[1]]).mean().item()

        other = [i for c2 in pcats if c2 != cat
                 for i in ([id_to_idx[p] for p in pri[pri["primary"] == c2]["paperId"] if p in id_to_idx][:120])]
        O = F.normalize(torch.from_numpy(matrix[other]), p=2, dim=1)
        inter = (1 - (V @ O.T)).mean().item()

        intra_all.append(intra); inter_all.append(inter)
        say(f"  {cat:<28}{len(idx_):>8,}{intra:>9.4f}{inter:>9.4f}{inter / max(intra, 1e-9):>8.3f}")
    say("  " + "-" * 60)
    mi, me = float(np.mean(intra_all)), float(np.mean(inter_all))
    say(f"  {'MEAN':<28}{'':>8}{mi:>9.4f}{me:>9.4f}{me / max(mi, 1e-9):>8.3f}")
    say()
    say("  ratio ~1.0 => a paper is no closer to its own field than to a different field:")
    say("  the metric space does not encode field structure. ratio >> 1.0 => it does.")
    say()

    with open(REPORT, "w") as f:
        f.write("\n".join(OUT) + "\n")
    print(f"\n[report written to {REPORT}]")


if __name__ == "__main__":
    main()
