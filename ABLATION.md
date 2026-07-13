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

---

## Provenance

Every number in this document comes from one checkpoint and one training run.

| Artifact | Path |
|---|---|
| Evaluated checkpoint | `scripts/models/user_history_encoder_best.pt` |
| Its training log | `results/training_log_hardneg.csv` |
| Experiment scripts | `scripts/eval/` |

That checkpoint is what every script in `scripts/eval/` loads. It came from the
normalized + field-aware-hard-negative run: 4 epochs, early stopping fired.
`results/legacy/` holds two earlier random-negative runs, which are superseded and
produced nothing in this document. If you are reading a training curve from one of
those, you are reading the wrong model.

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

**Result.** 0.0143 vs 0.0001. A factor of 143.

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
There is no component to blame. The design is what's wrong.

---

## Experiment 4: is the eval searching the wrong space?

**Hypothesis.** Space mismatch. User fingerprints and paper vectors live in
different geometries, so nearest-neighbor retrieval between them is meaningless
and the score is an artifact of the evaluation rather than the model.

**Script.** `scripts/eval/eval_projected_space.py`. The projection was trained to
map user-history vectors *toward* raw paper embeddings, so searching the raw
paper index should be correct by construction. This tests the alternative:
project the paper index through the same map, then search projected against
projected.

**Result.** Projecting the paper index produced 0.0000. Not degraded.
Categorically mismatched. Raw-space search was correct all along.

**Verdict.** No space bug. Eliminated.

This one was quick, and it had to be run regardless, because "you evaluated in
the wrong space" is the first thing anyone asks when a trained model loses to an
average. Now it's answered with a number instead of an argument.

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

Mean-pool beats the tower by roughly 2x at every input level, on Recall@10,
NDCG@10, and Recall@100. Both improve monotonically with more seed papers. The
gap stays roughly constant. The tower never catches up.

**The sharper version.** The trained tower loses to the popularity floor at 1 and
2 inputs. It performs worse than recommending the same most-cited papers to every
user, regardless of who they are. Seed-paper mean-pool is the only method that
clears the floor at all input levels, at roughly 1.4 to 1.7 times over it.

That is the central inductive-bias finding at its most extreme: in the
shortest-sequence regime, the learned encoder isn't merely worse than an average,
it's worse than not looking at the user at all.

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

## The category arm, and why it's reported separately

`cold_start_eval.py` also tests a second cold-start signal: the user selects
fields of interest rather than naming papers, and retrieval runs from the
corresponding category centroid.

It fails, and it fails below the popularity floor. Category selection scores
roughly 0.0002 at Recall@10 and 0.0012 at Recall@100, against a floor of 0.0125
and 0.0391.

Read naively, 0.0002 looks like a bug. It isn't. The mechanism is geometric: the
Computer Science centroid sits at **cosine 0.9998 to the mean of the entire
corpus.** Because the corpus is roughly 85% CS, the CS centroid and the corpus
mean are the same vector for retrieval purposes. A user who selects "Computer
Science" moves the query point essentially nowhere, and nearest-neighbor search
from the center of the space returns whatever happens to be closest to the
center, which is a geometric accident rather than a recommendation.

The honest framing is that category signal is directionally valid but far too
coarse for specific-paper retrieval: roughly 6x better than random at top-100,
near-zero at top-10. It carries some information. Not enough to beat a baseline
that ignores the user entirely.

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
the example level. Verified: 0 chains span splits. Without this, overlapping
windows from a single chain could straddle train and test, and any train/val gap I
measured would be leakage rather than overfitting.

**Observed** (`results/training_log_hardneg.csv`):

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.0498 | 0.3143 |
| 2 | 0.0458 | 0.3205 |
| 3 | 0.0448 | 0.3212 |
| 4 | 0.0444 | 0.3214 |

Train loss falls steadily. Validation is flat to slightly rising after epoch 1.
Mild divergence: the model stops generalizing further but doesn't visibly
degrade. The engineered correlation produced the effect it was designed to
produce, in a gentle form.

Early stopping fired at 4 epochs. The shipped `best.pt` is not the final epoch,
verified by direct weight comparison against `last.pt` rather than by trusting the
filename.

**On the loss scale.** The absolute train/val gap is large because of the hard
negatives, not the overfitting. Discriminating a CS paper from another CS paper is
a far harder triplet task than discriminating it from a random Medicine paper, so
the loss scale is not comparable to a random-negative run. The two superseded runs
in `results/legacy/` are exactly that comparison: same data, same architecture,
random negatives, and they plateau around val 0.065 — roughly 5x lower. That
difference is the clearest evidence available that the hard-negative sampler
changed the objective it was meant to change.

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
  random-negative runs. Same data, same architecture, 5x separation. The objective
  changed, so the loss scale changed, and that signature identifies the run.
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
averaging on short, topically homogeneous histories. The four bug explanations are
ruled out by experiment rather than by assertion. The effect strengthens with
sequence length, in the direction that identifies the cause. It reproduces
independently in cold start, where the model drops below a baseline that ignores
the user entirely. And the same embedding geometry explains both the popularity
bias and the category-signal failure, which means these aren't four findings.
They're one system understood four ways.

The model is worse than an average. I know why, and I can show the work.

That's the trade. I think it's the right one.
