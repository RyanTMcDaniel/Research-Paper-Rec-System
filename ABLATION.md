# Why the Attention Model Loses to Mean-Pool

**Result.** The self-attention user-history encoder scores Recall@10 = 0.0140
against a mean-pool baseline's 0.0225. A no-parameter baseline beats the
learned model.

**Conclusion.** This is not a bug. On 3-to-5 paper histories, a learned
attention + projection encoder is genuinely worse than averaging raw
embeddings. Mean-pooling is the correct inductive bias for short sequences,
where a simple centroid is a strong, stable estimator and the learned machinery
adds variance without adding signal.

Every more flattering explanation was tested and ruled out. What follows is the
trail.

**Update (later session).** The headline 0.0140 was measured with the paper tower
random-init (see [Provenance](#provenance)) — a crippled configuration. Training the
paper tower changes the number: co-training both towers reaches **0.0315**
(Experiment 8), the architecture's real ceiling here. The conclusion below is
unchanged — the model still loses, because mean-pool rises to 0.0410 in that same
trained space — but where this document implies 0.0140 is *the model's* number, read
it as the number for *one untrained tower*. Experiments 7–8 close that hole.

**Jump to an experiment:**

1. [Did training actually work?](#experiment-1-did-training-actually-work) — trained vs. random-init
2. [Did attention collapse into mean-pool?](#experiment-2-did-attention-collapse-into-mean-pool) — attention-weight distribution
3. [Is the projection layer the culprit?](#experiment-3-is-the-projection-layer-the-culprit) — component ablation
4. [Is the eval searching the wrong space?](#experiment-4-is-the-eval-searching-the-wrong-space) — projected-space retrieval
5. [**Length-stratified: the gap widens, not narrows**](#experiment-5-length-stratified-evaluation) — the sharpest result
6. [Does it replicate in cold start?](#experiment-6-does-it-replicate-in-cold-start) — third confirmation
7. [Is the paper tower the culprit? — train it](#experiment-7-is-the-paper-tower-the-culprit--train-it) — the untrained projection
8. [**Co-train both towers**](#experiment-8-co-train-both-towers) — the fix the chain implies, and the gap holds

Also: [the category arm](#the-category-arm-and-why-its-reported-separately) · [overfitting](#overfitting) · [artifact-provenance note](#a-note-on-artifact-provenance)

---

## Provenance

Most numbers in this document come from one checkpoint and one training run —
the main benchmark and Experiments 1–6. Experiments 7–8 add two further training
runs, as flagged in the Update box above: a paper-projection-only run (Exp 7) and
a co-training run (Exp 8), each with its own checkpoint (noted in those
experiments). Three training runs in total.

| Artifact | Path |
|---|---|
| Evaluated checkpoint (Exp 1–6, benchmark) | `scripts/models/user_history_encoder_best.pt` |
| Its training log | `results/training_log_hardneg.csv` |
| Exp 7 paper projection | `scripts/models/paper_projection_trained.pt` |
| Exp 8 co-trained towers | `scripts/models/user_tower_cotrained.pt`, `scripts/models/paper_projection_cotrained.pt` |
| Experiment scripts | `scripts/eval/` |

That checkpoint is what the Experiment 1–6 scripts in `scripts/eval/` load, and
it remains the query encoder for Experiment 7. Experiment 8's `eval_cotrained.py`
instead loads the co-trained user tower `user_tower_cotrained.pt`. It came from the
normalized + chain-guarded-hard-negative run: 4 epochs, early stopping fired.
`results/legacy/` holds two earlier random-negative runs, which are superseded and
produced nothing in this document. If you are reading a training curve from one of
those, you are reading the wrong model.

**What this document originally left unstated: the paper tower was never trained.**
`paper_matrix.npy` — the 253,703-vector index every experiment here searches — is
frozen SciBERT under a *random-init* 768→256 projection. The build pipeline saved the
projected 256-d cache but never loaded trained projection weights, and discarded the
raw 768-d intermediate; `two_tower.py`'s optimizer is `Adam(history_encoder.parameters())`
only. So across Experiments 1–6 the "paper embeddings" are a random linear map of
SciBERT — not raw SciBERT, and not a trained space. This is load-bearing for the
document's own method: you cannot conclude "the design is what's wrong" while one tower
is random-init. A later session re-encoded the full corpus to raw 768-d
(`paper_768_full.npy`, validated by its ≈0.85 mean random-pair cosine — the SciBERT
anisotropy signature) and ran Experiments 7–8 to close exactly this hole.

The checkpoint is single; the evaluation samples are not. Each script draws its
own different-sized random subsample of the test set — `evaluate.py` 2,000 users,
`diag_ablation.py` and `diag_trained_vs_rand.py` 3,000, `cold_start_eval.py`
10,000 — so the same model-vs-mean-pool comparison appears as 0.0140 / 0.0225 in
the main benchmark (the headline above) and 0.0143 / 0.0210 in the ablation
experiments. That is sampling variation, not disagreement: the finding is
invariant to it, mean-pool winning by a wide margin at every sample size. Each
results table below names the script and sample size it came from.

One caveat on that run label. The training code that produced this checkpoint
lived in a Kaggle notebook that no longer exists; `scripts/models/two_tower.py`
is the reconstructed equivalent, and its validation protocol does not exactly
reproduce the logged val series (the shipped chain-guarded loss on the val split
is ~0.10, not the logged ~0.31). So "normalized + chain-guarded" is inferred from
what *is* verifiable against the artifacts — `best.pt` differs from `last.pt` by
direct tensor comparison, the logged curve matches this run's early-stopping
shape, and its ~0.31 loss scale is the signature that separates it from the
legacy runs — rather than read from the source that generated it. This is the
fourth instance of the same artifact-provenance gap this document keeps hitting:
unpersisted cache weights, unseeded RNG, unnamed logs, and now the training
notebook itself.

I am stating this up front because reconstructing it was harder than it should
have been. See [A note on artifact provenance](#a-note-on-artifact-provenance) at
the bottom.

---

## The suspects

Before I could conclude "the architecture is wrong," I had to eliminate four
explanations that would have meant "the code is wrong." In order of how much I
wanted them to be true:

1. **The training run is broken.** The weights never learned anything and I'm
   evaluating noise.
2. **Attention collapsed.** It learned uniform weights, silently became
   mean-pool, and the gap is variance.
3. **The projection layer is mangling the fingerprints.** Mean-pool wins partly
   because it skips the projection entirely, so the learned map is a distortion
   to remove rather than a component to trust.
4. **The evaluation searches the wrong space.** User fingerprints and paper
   vectors live in different geometries, so retrieval is meaningless and the
   number is an artifact.

Each of these is a bug with a fix. "Attention is the wrong inductive bias for
this sequence length" is not. I ran them in that order deliberately: I did not
want to accept the expensive conclusion until the cheap ones were dead.

---

## Experiment 1: did training actually work?

**Hypothesis.** The training run is broken; the weights are effectively random.

**Script.** `scripts/eval/diag_trained_vs_rand.py` evaluates the trained
checkpoint against a randomly-initialized model of identical architecture, on the
same test users, same geometry.

**Result.** 0.0143 vs 0.0001. The random-init figure is roughly one hit across
three seeds and 3,000 users — a denominator too noisy to support a precise
multiplier. The supportable claim is simpler: random-init retrieval is
approximately zero, and the trained model is not.

**Verdict.** Training worked. The model learned something substantial relative to
its own initialization. Eliminated.

---

## Experiment 2: did attention collapse into mean-pool?

**Hypothesis.** Attention learned near-uniform weights, meaning it effectively
*became* mean-pool, and the performance gap is noise between two versions of the
same thing.

**Script.** `scripts/eval/diag_attn_weights.py` extracts the attention weights
over test histories and compares their distribution to uniform.

**Result.** The weights are non-uniform. Max weight ~0.33 against 0.25 uniform on
4-item histories.

**Verdict.** Eliminated, and this is where the investigation stopped being about
finding a bug.

Attention is making real choices. It looks at a user's history, decides some
papers matter more than others, and acts on that. And the model that does this
performs worse than the model that refuses to choose at all. The selections
aren't absent, they aren't collapsed, they aren't noise. They are confident,
non-uniform, and actively harmful. Whatever the model learned to prioritize is
worse than treating every paper equally, which puts the failure in the selection
itself rather than in the machinery producing it.

---

## Experiment 3: is the projection layer the culprit?

**Hypothesis.** The learned 256-to-256 projection is distorting the fingerprints,
and mean-pool wins partly because it never applies that projection at all.

**Script.** `scripts/eval/diag_ablation.py` runs four configs against the same
trained weights, bypassing one component at a time.

| Config | Recall@10 | NDCG@10 |
|---|---|---|
| full (attn → pool → proj → norm) | 0.0143 | 0.0072 |
| no_projection | 0.0003 | 0.0001 |
| no_attention | 0.0003 | 0.0001 |
| mean-pool baseline | 0.0210 | 0.0108 |

<sub>Source: `scripts/eval/diag_ablation.py`, 3,000 test users.</sub>

**Result.** Removing either learned component collapses the model to
approximately random. Neither is a distortion to strip out. Both are
load-bearing.

**Verdict.** Eliminated, and the shape of the result matters more than the
elimination.

The naive read is "the projection is fine, moving on." What the table actually
shows is that attention and the projection are jointly trained and mutually
dependent. The projection learned to map *attention outputs* into the retrieval
space, so handing it a raw mean is out-of-distribution input and it produces
garbage. Symmetrically, the attention output is only meaningful once passed
through the projection it was trained alongside. These aren't independent modules
to mix and match. They're one learned function with an internal interface, and
the ablation cuts that interface.

That makes the result a stronger indictment, not a weaker one. If the projection
had been broken, I'd have a bug and a fix. Instead the learned pipeline is
internally coherent, every piece doing exactly what it was trained to do, every
piece necessary to the other, and the coherent whole still loses to an average.
Within the user tower, there is no component to blame. But this experiment tested
only the user side. It never tested the *paper* tower — whose 768→256 projection, the
Provenance note reveals, was never trained. Experiment 7 trains it and Experiment 8
co-trains both, and the model more than doubles: a component *was* partly to blame. The
honest statement is narrower than the one I first wrote here — the design is suboptimal
at this history length *and* the paper tower was crippled — and Experiment 3 closed the
component question one tower too early. What survives intact is what this experiment
actually established: attention and the user-side projection are one jointly-trained
function, load-bearing on both sides.

---

## Experiment 4: is the eval searching the wrong space?

**Hypothesis.** Space mismatch. User fingerprints and paper vectors live in
different geometries, so nearest-neighbor retrieval between them is meaningless
and the score is an artifact of the evaluation rather than the model.

**Script.** `scripts/eval/eval_projected_space.py`. The user tower was trained to
map history vectors *toward* the paper index — which is itself a random 768→256
projection of SciBERT, not raw SciBERT (see Provenance) — so searching that index
directly should be correct by construction. This tests the alternative: project the
paper index through the user tower's map, then search projected against projected.

**Result.** Projecting the paper index produced 0.0000. Not degraded.
Categorically mismatched. Raw-space search was correct all along.

**Verdict.** No space bug. Eliminated.

This one was quick, and it had to be run regardless, because "you evaluated in
the wrong space" is the first thing anyone asks when a trained model loses to an
average. Now it's answered with a number instead of an argument.

**Correction (Experiments 7–8).** The blanket reading — "projecting the index is
categorically mismatched, raw-space was correct all along" — is wrong, and the 0.0000
says less than it seems. It is the score for projecting through *this* map: an
*untrained* one. Project the corpus through a *trained* projection and search a
co-trained user query against it, and retrieval works — **0.0315** (Experiment 8), the
best model number in this document. So there is no space bug for the shipped model, but
the deeper claim is the opposite of "eliminated": query–index geometry is a live,
first-class variable. Experiment 7 shows that moving one tower's space alone *breaks*
retrieval; Experiment 8 shows co-training both *restores* it. "The eval searches the
wrong space" was the right question — it just has a richer answer than one number.

---

## Experiment 5: length-stratified evaluation

The bug explanations are gone. What survives is a hypothesis about the
architecture itself, and it makes a testable prediction.

**Hypothesis.** If attention underperforms because short sequences don't give it
enough to work with, the gap should *narrow* as histories get longer. More
papers, more structure to attend over, more room for a learned weighting to beat
a flat average.

**Script.** `scripts/eval/eval_by_histlen.py` buckets test users by history
length (3, 4, 5) and computes the model-versus-mean-pool gap in each bucket.

**Result.** The gap widens. Roughly -0.005 at length 3, roughly -0.010 at
length 5. Mean-pool improves monotonically with more papers. The model does not
keep up.

**Verdict.** The surviving hypothesis is confirmed, but in the opposite direction
from the naive prediction, and the reversal is the whole finding.

The intuitive story is that attention is *starved*: give it more sequence and it
starts earning its parameters, and somewhere past length 5 or 10 it overtakes the
average. If that were true the gap would shrink as histories grow. It does the
reverse. Every additional paper in a user's history makes mean-pool better and
makes the model relatively worse. The extra paper is extra signal, and a plain
average absorbs it cleanly while the learned weighting spends it.

That reframes the failure. Attention isn't underfed, it's destructive. It takes
information a simple centroid would have used and misweights or discards it, and
it does so *more* the more information you give it.

It makes sense once you look at what these histories are. They're co-citation
windows: papers cited together, which makes them topically tight by construction.
There is no signal-versus-noise separation for attention to discover, because
every paper in the history is relevant. In that regime the mean is not a lazy
approximation of the right answer, it *is* the right answer, and it's a
low-variance estimator of it. Attention introduces a learned weighting over items
that should be weighted equally, which can only add variance. The more items you
hand it, the more chances it has to get that weighting wrong, and the wider the
gap grows.

Which is exactly what the stratification shows.

---

### The obvious objection

There's a circularity here and it needs stating before someone else states it. These
histories are co-citation windows: papers cited together in the same reference list.
That makes them topically tight by construction. And the explanation I just gave for
why mean-pool wins is that the histories are topically tight, so a centroid is the
right estimator. Which raises the fair question of whether I built a dataset where
mean-pool was guaranteed to win and then went and discovered that mean-pool wins.

Largely, yes. That's a real limit on what this result can claim. Attention exists to
find structure inside a sequence, and specifically to separate the parts that matter
from the parts that don't. My data generator produces sequences with no such
separation to find. Every paper in a co-citation window is on-topic, because that's
what a co-citation window is. So the model was handed a sequence with nothing to
filter and asked to filter it, and it did the only thing available to it: invented a
weighting where none was warranted. That adds variance and nothing else, which is
exactly what the length stratification shows.

Real reading histories are not like this. People read a paper because a colleague
sent it, because they were reviewing it, because they needed one method from it,
because it was 2am. Those are incidental, and down-weighting incidental items is the
job attention was built for. My histories contain none, so attention was never
actually given that job. Testing it properly would need either real interaction data
or synthetic histories with noise papers deliberately injected. I have neither.

So the honest scope: I did not disprove attention. I disproved attention on this
data, and the construction of the data is part of why. What survives without
qualification is everything the eight experiments established about *this system* — that
training worked, that attention didn't collapse, that the projection isn't broken,
that the space isn't mismatched, and that the failure gets worse rather than better
with more input. What doesn't survive is the general sentence. Not "attention loses
to mean-pool on short histories," but "attention loses to mean-pool on short
homogeneous histories, and mine were homogeneous by construction." That's a smaller
claim. It's also the one I can actually defend.

---

## Experiment 6: does it replicate in cold start?

**Script.** `scripts/eval/cold_start_eval.py`. 10,000 users, seed-paper cold
start, sweeping 1 to 4 input papers, against a signal-independent popularity
floor. Fixed population, both methods evaluated on the same users.

**Result.**

| Inputs | Tower R@10 | Mean-pool R@10 | Tower R@100 | Mean-pool R@100 |
|---|---|---|---|---|
| 1 | 0.0084 | 0.0173 | 0.0384 | 0.0567 |
| 2 | 0.0095 | 0.0184 | 0.0492 | 0.0659 |
| 3 | 0.0125 | 0.0198 | 0.0565 | 0.0717 |
| 4 | 0.0122 | 0.0218 | 0.0640 | 0.0773 |
| **popularity floor** | **0.0125** | **0.0125** | **0.0391** | **0.0391** |

<sub>Source: `scripts/eval/cold_start_eval.py`, 10,000 users (fixed cold-start population).</sub>

Mean-pool beats the tower by roughly 2x at every input level, on Recall@10,
NDCG@10, and Recall@100. Both improve monotonically with more seed papers. The
gap stays roughly constant. The tower never catches up.

**The sharper version.** On Recall@10 the trained tower never beats the
popularity floor at any input level: 0.0084, 0.0095, 0.0125, 0.0122 against a
floor of 0.0125 — below at 1, 2, and 4 inputs, a tie at 3. On Recall@100 it
clears the floor from 2 inputs onward but loses at 1 (0.0384 vs 0.0391). A
system reading the user's own chosen papers performs at or below one that
recommends the same most-cited list to everyone. Seed-paper mean-pool is the
only method that clears the floor at every input level, at roughly 1.4 to 2x
over it.

That is the central inductive-bias finding at its most extreme: in the
shortest-sequence regime, the learned encoder isn't merely worse than an average,
it's no better than not looking at the user at all.

**Verdict.** The finding replicates where sequences are maximally short. Third
independent confirmation of one thesis (main eval, length stratification, cold
start).

**Methodology note.** A preliminary run on 2,000 users appeared to show two
structural effects: a dip in the tower's performance at 4 inputs, and convergence
between the curves at 3. Both looked like findings. Both were noise. Re-running at
10,000 users erased them, leaving a cleanly monotonic tower and two curves still
rising through 4 inputs.

I caught that by increasing n, not by staring at the curve and deciding whether I
believed it. That's why the note is here. The numbers elsewhere in this document
survived being re-run at a larger sample, and I don't publish the first structure
I see.

---

## Experiment 7: is the paper tower the culprit? — train it

The four bug explanations died and Experiment 5 pinned the architecture. But the
Provenance note surfaces a fifth suspect the original six never tested: the paper
tower's 768→256 projection was never trained. Every number to this point was measured
against a *random* linear map of SciBERT. Before "the design is what's wrong" can
stand, that has to be ruled out — you cannot indict the architecture while one of its
two towers is random-init.

**Hypothesis.** The model loses because the paper projection is untrained. Train it —
alone, user tower frozen — and the model should climb.

**Script.** `scripts/models/train_paper_projection.py` trains a fresh
`nn.Linear(768→256)` on the raw 768-d corpus (`paper_768_full.npy`) against the
*frozen* trained user tower as anchor, same triplet objective.
`build_matrix_from_projection.py` rebuilds the index; `eval_trainedproj.py` runs a
decoupled eval — query built from the original space, index from the new one.

**Result.** The model got *worse*, and mean-pool got *better*.

| Configuration | model R@10 | mean-pool R@10 | gap |
|---|---|---|---|
| baseline (random paper proj) | 0.0140 | 0.0225 | −0.0085 |
| paper-projection-only, frozen user tower | 0.0090 | 0.0300 | −0.0210 |

<sub>Source: `scripts/eval/eval_trainedproj.py`, 2,000 test users (seed 42).</sub>

**Verdict.** Not the fix — but a sharp diagnostic. The trained paper space is genuinely
better: mean-pool, which lives entirely in the paper space and never routes through the
user tower, rose from 0.0225 to 0.0300. Yet the model *dropped* to 0.0090, widening the
gap to −0.0210. The reason is geometry coupling. The frozen user tower was trained to
emit fingerprints into the *original* random-projection space; re-projecting the papers
moves the index to a new geometry the user tower was never taught to read. Query and
index now live in different spaces, and retrieval degrades.

This does two things. It corrects Experiment 3 — a component *was* partly to blame, and
Exp 3 declared "no component to blame" one tower too early. And it sets up the real
test: you cannot improve one tower against a space the other doesn't share. Fix that,
and see what's left.

---

## Experiment 8: co-train both towers

**Hypothesis.** If Experiment 7 failed because the frozen user tower couldn't read a
re-projected index, then training *both* towers from scratch in one shared geometry
should remove the misalignment — and let the model finally exploit a trained paper
space. Maybe enough to beat mean-pool.

**Script.** `scripts/models/train_cotrained.py`: a fresh user tower and a fresh
`nn.Linear(768→256)`, **525,824 trainable parameters**, one optimizer, trained jointly.
SciBERT is never instantiated — the tower consumes precomputed 768-d vectors, so BERT
is frozen by construction. Negatives are **in-batch semi-hard (a FAISS-free
approximation of full-corpus hard mining)** at batch 512; the random in-batch negatives
the project started with make the task trivially easy and the training loss
eval-worthless. Single geometry throughout: the user tower's history input is
`normalize(proj(history_768))`, the exact transform eval applies, so query and index are
the same space by construction. `build_cotrained_matrix.py` builds the index;
`eval_cotrained.py` runs the full 253,703-paper eval on the same 2,000 users.

**Result.**

| Configuration | model R@10 | model NDCG | mean-pool R@10 | mean-pool NDCG | gap |
|---|---|---|---|---|---|
| user-tower-only (baseline, random paper proj) | 0.0140 | 0.0074 | 0.0225 | 0.0113 | −0.0085 |
| paper-projection-only, frozen user tower (Exp 7) | 0.0090 | 0.0047 | 0.0300 | 0.0152 | −0.0210 |
| **co-trained, both towers (Exp 8)** | **0.0315** | **0.0154** | **0.0410** | **0.0212** | **−0.0095** |

<sub>Source: `scripts/eval/eval_cotrained.py`, 2,000 test users (seed 42), full-index retrieval.</sub>

**Verdict.** Co-training more than doubled the model — Recall@10 0.0140 → 0.0315
(+125%), NDCG 0.0074 → 0.0154 — and cleanly removed the geometry confound from
Experiment 7. And it still lost. The same co-trained space that lifted the model lifted
mean-pool just as much, 0.0225 → 0.0410, so the gap held essentially flat: −0.0085 →
−0.0095. The model closed the *ratio* to mean-pool from 0.62 to 0.77 but never overtook
it.

This is the result that closes the chain. It removes the last two confounds at once.
Query–index misalignment is not the cause: co-training fixed it and the gap didn't move.
The untrained paper tower is not the cause of the *gap*: training it doubled *both*
methods, because a better paper space helps the average exactly as much as it helps the
model. What is left is precisely Experiment 5's finding, now with every representational
and geometric confound eliminated. At histories of ~4 papers (max 5), there is almost no
sequential or relational structure for attention to exploit over a mean; mean-pool is
the correct inductive bias at this length, and the bottleneck is the user-side task, not
the representation or the geometry.

The scope caveat from Experiment 5 stands and matters more here, not less: this is
short, homogeneous, co-citation histories. "Co-training doesn't beat mean-pool at
4-paper histories" is the defensible claim. "Attention never helps recommendation" is
not, and Experiment 8 is not evidence for it.

---

## The category arm, and why it's reported separately

`cold_start_eval.py` also tests a second cold-start signal: the user selects
fields of interest rather than naming papers, and retrieval runs from the
corresponding category centroid.

It fails, and it fails below the popularity floor. Category selection scores
roughly 0.0002 at Recall@10 and 0.0012 at Recall@100, against a floor of 0.0125
and 0.0391.

Read naively, 0.0002 looks like a bug. It isn't. The mechanism is geometric: the
Computer Science centroid sits at **cosine 0.9998 to the mean of the entire
corpus.** 94% of the corpus carries a CS tag (74% has CS as its primary field),
so the CS centroid — the mean of all CS-tagged papers — is averaging nearly the
whole corpus, and it lands on the corpus mean. For retrieval purposes they are
the same vector. A user who selects "Computer
Science" moves the query point essentially nowhere, and nearest-neighbor search
from the center of the space returns whatever happens to be closest to the
center, which is a geometric accident rather than a recommendation.

The honest framing is that category signal is directionally valid but far too
coarse for specific-paper retrieval: roughly 3x better than random at top-100
(0.0012 against a random baseline of 100/253,703 ≈ 0.0004) and roughly 5x at
top-10 (0.0002 against ≈ 0.00004). It carries some information. Not enough to
beat a baseline that ignores the user entirely.

This is the same geometry as the popularity-bias audit, surfacing a second time.
The center of the embedding space is crowded and over-retrieved, which is why
popular papers dominate recommendations and why a query at the corpus mean
returns nothing useful. One property of the space, two symptoms, found in two
separate investigations.

The comparison between the two arms is the actual result, and it's the tradeoff
the cold-start experiment was designed to measure: precise signal (naming papers)
massively outperforms coarse signal (naming fields), and the cost is paid in user
effort. Naming four papers is real work. Clicking a field is free. The free option
produced results indistinguishable from a signal-independent baseline. Generalized:
a cold-start signal is only worth collecting if it is discriminative *within the
population you actually serve.* A label most of your corpus shares carries
near-zero information no matter how meaningful it sounds, and you cannot see that
without measuring the geometry.

---

## Overfitting

The training data was built with deliberately correlated, overlapping
sliding-window examples. That was a design choice, not a data accident. Sliding a
3-to-5 window along a co-citation chain produces examples sharing most of their
content with their neighbors, which is a textbook setup for memorization. I wanted
a real opportunity to diagnose overfitting on data whose correlation structure I
had built myself and therefore understood.

**Split discipline.** Temporal split assigned at the co-citation-chain level, not
the example level. Verified against the shipped artifact by
`scripts/tests/check_split_integrity.py`: 106,497 chains, 0 span splits. Without this, overlapping
windows from a single chain could straddle train and test, and any train/val gap I
measured would be leakage rather than overfitting.

**Observed** (`results/training_log_hardneg.csv`):

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.0498 | 0.3143 |
| 2 | 0.0458 | 0.3205 |
| 3 | 0.0448 | 0.3212 |
| 4 | 0.0444 | 0.3214 |

<sub>Source: `results/training_log_hardneg.csv` (per-epoch training log, full train/val splits — not a subsample).</sub>

Train loss falls steadily. Validation is flat to slightly rising after epoch 1.
Mild divergence: the model stops generalizing further but doesn't visibly
degrade. The engineered correlation produced the effect it was designed to
produce, in a gentle form.

Early stopping fired at 4 epochs. The shipped `best.pt` is not the final epoch,
verified by direct weight comparison against `last.pt` rather than by trusting the
filename.

**On the loss scale.** I originally read the 5x gap between this run's val loss
(~0.32) and the legacy runs' (~0.065) as evidence that the chain-guarded sampler
had made the triplet objective harder. The logs contradict that story. The
train-loss trajectories of all three runs are identical to the third decimal
(epoch 1: 0.0498 / 0.0497 / 0.0498; epoch 4: 0.0444 / 0.0444 / 0.0443) — a
sampler that changed the objective would have moved the loss actually being
optimized, and it didn't. The entire 5x difference lives in validation, which
points at the validation-negative protocol, not the objective.

Probing the shipped checkpoint against the val split localizes it further. The
shipped `evaluate_cached` (chain-guarded negatives) yields 0.103 — not the
logged 0.314 — and on the chain-ordered, unshuffled val loader the guard drops
60% of rows because whole batches share one chain. Unguarded roll negatives on
the same weights yield 0.380. The logged 0.314 is consistent with a mostly
unguarded validation protocol saturated with same-chain false negatives —
triplets whose "negative" comes from the anchor's own chain, pinning the loss
near the margin — and inconsistent with the shipped guarded code. Normalizing
the geometry moves the number by less than 0.002, so geometry is not the
driver either.

What I cannot do is finish the attribution: the exact code that produced
either logged validation series is not in this repository (the legacy runs'
training code and checkpoints are gone), so the residual gap to 0.065 —
plausibly a shuffled-vs-ordered val loader turning roll negatives from easy
cross-chain samples into same-chain false negatives — remains a hypothesis,
stated as one.

None of this touches what the section actually relies on: the within-run
comparison. Protocol held fixed, train falls while validation rises after
epoch 1. That is the overfitting signal, and it survives.

**Mitigation.** Early stopping. No regularization sweep.

**Why not more.** Because the ablation says the attention tower shouldn't be
there. A dropout and weight-decay sweep to close a mild train/val gap would mean
carefully tuning a component my own evidence says to remove: optimization in
service of a conclusion I had already disproven.

The overfitting result is real and it's mild, and it's *subsumed* by the larger
finding. The model isn't merely memorizing a bit, it's using an inductive bias
that loses to an average regardless of how well it's regularized. Regularizing
harder would have made the wrong architecture slightly less wrong, and I'd have
spent the runway polishing something I was about to conclude shouldn't exist.
Early stopping is the mitigation proportionate to the size of the problem.
Following the evidence beat reaching for the textbook fix.

---

## A note on artifact provenance

Reconstructing which training log belonged to which checkpoint was harder than it
should have been, and the reason is worth stating plainly rather than hiding
behind a tidy results table.

This project accumulated three training logs across two configurations, none of
them named to distinguish the runs, and the projection weights used to build the
253K-vector paper cache were never persisted at all. When it came time to write
this document I could not tell from the filenames which curve belonged to the
model I had evaluated.

I resolved it by forensics rather than by reading a filename:

- The hard-negative run has a val loss around 0.32 against roughly 0.065 for the
  random-negative runs. Same data, same architecture, 5x separation — a
  fingerprint that identifies which log is which. (The separation turned out to
  reflect differing validation-negative protocols rather than a changed training
  objective — see "On the loss scale" — but as an identifying signature it works
  regardless of its cause.)
- The hard-negative run stopped at 4 epochs with validation rising after epoch 1,
  so early stopping must have fired and `best.pt` must differ from `last.pt`. The
  random-negative runs ran a full 8 epochs with validation still falling at the
  end, so their `best` and `last` would be the same weights.
- I loaded both checkpoints and compared the tensors directly. They differ. Only
  the hard-negative run is consistent with that.

That triangulation is sound and I believe the result. But needing to do it at all
is the finding. Three separate instances of one class of gap in a single project:
unpersisted encoder weights, unseeded RNG, unnamed run logs. That is not three
mistakes, it's one missing habit. Anything that produces a downstream-anchored
artifact should persist the thing that produced it, and name the artifact after
the config that produced it. `results/` and its README exist because of this, and
they exist several months later than they should have.

---

## What this cost, and what it bought

I spent the back half of this project's runway on diagnosis rather than on making
the number go up, and I'd make the same call again.

What I'd have otherwise: a model with a few more points of Recall@10 from a
regularization sweep and a longer history window, still probably losing to
mean-pool, and no explanation for any of it. A number I couldn't defend in a room.

What I have instead: a negative result with a mechanism. Attention loses to
averaging on short, topically homogeneous histories. Six alternative explanations are
ruled out by experiment rather than by assertion — the four original bug suspects, plus
the untrained paper tower (training it doubled the model without closing the gap) and
query–index geometry misalignment (co-training removed it and the gap held). The effect strengthens with
sequence length, in the direction that identifies the cause. It reproduces
independently in cold start, where the model drops below a baseline that ignores
the user entirely. And the same embedding geometry explains both the popularity
bias and the category-signal failure, which means these aren't four findings.
They're one system understood four ways.

The model is worse than an average. I know why, and I can show the work.

That's the trade. I think it's the right one.
