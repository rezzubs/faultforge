# `syndromes` design

What this crate does, the decisions behind it, and the order to build it in.

## Purpose

Take a gate-level netlist of a floating point multiply-add and a source of
realistic inputs. Estimate how a single stuck-at fault somewhere in the
netlist corrupts the output. The result is a histogram of *syndromes*, where
a syndrome is `correct_output XOR faulty_output` as a 32-bit mask.

The consumer is a fault injection experiment on a simulated systolic array.
Simulating the netlist inside every processing element is too slow, so the
experiment instead draws a mask from this histogram and applies it to the
multiply-add result. The histogram is the fault model. This crate builds it.

## What it is not

- Not exhaustive over inputs. `2^96` input combinations is out of reach, and
  some planned input sources cannot enumerate anyway.
- Not one histogram per gate. The fault model is deliberately "some unknown
  gate is stuck", averaged over all gates. Per-gate histograms would multiply
  the cost by the number of gates (thousands) and would no longer be that
  model. A fixed-gate mode exists only for diagnostics.
- Not incremental. A run produces one file from scratch. Appending to an
  existing histogram would need the generation parameters fingerprinted and
  compared, which is not worth building without a need. The seed and the
  evaluation count are recorded, so any run is reproducible.
- Not tied to particular netlists. Port names and constant input assignments
  come from the command line. The crate knows nothing about the netlists it
  is given beyond "these buses are inputs, this bus is the output".

## Core loop

One evaluation:

1. Ask the input source for an `(activation, weight, partial_sum)` triple.
2. Remove any fault, write the inputs, settle, read the correct output.
3. Pick a fault case uniformly: one gate output and one polarity (stuck at 0
   or 1). Apply it, settle, read the faulty output.
4. Add `correct XOR faulty` to the histogram.

Correct before faulty is deliberate. Writing new inputs re-propagates the
whole netlist; switching the fault re-propagates only the cone downstream of
one gate. This order makes the second settle the cheap one.

The correct output comes from the netlist with no fault, never from the CPU.
The netlist's rounding may legitimately differ from the host's, and that
difference is not a syndrome.

### Unknown signals

The simulator has a three-valued signal: 0, 1, unknown. The netlists here are
combinational with every input driven, so an unknown output bit means the
netlist or its port configuration is broken, not that a fault did something
interesting. Generation stops with an error naming the input triple and the
fault case that produced it.

Past the netlist boundary the crate uses its own two-valued bit type, so no
code has to handle a variant that cannot occur. A syndrome is a plain `u32`.

## Seams

Two things are injected. Both exist because the technique is not final and
the alternatives are already known.

**The computer.** Takes a triple, returns output bits. Knows how many fault
cases it has and how to apply case `n`. Two implementations:

- Separate multiplier and adder netlists. The multiplier's output feeds the
  adder. A fault lives in exactly one of the two.
- One fused multiply-add netlist.

A netlist is described by a path, its input bus names, its output bus name
and a list of constant assignments (a rounding mode input, for example). Bus
widths are checked at load time.

**The input source.** Yields triples. Planned implementations:

- A profiling artifact: inputs recorded from processing elements while a
  real model ran through the simulated array, grouped by position and by the
  stage of the computation the element was in. The source is restricted to
  one stage and one group (the whole array, one column, one element) and
  samples with replacement.
- The same artifact, but drawing activation, weight and partial sum
  independently from their marginals. Exists to test whether the joint
  distribution matters or only the individual value distributions do.
- Independent draws from a normal distribution, with a scale per input.
  Recorded values cluster around zero, so this is a candidate parametric
  stand-in for the artifact. Whether it is a good one is the test.
- A constant triple. One stage of the computation always sees `(0, 0, 0)`.
- Uniformly random bit patterns. Not physically meaningful, useful for
  testing the crate.

A source is a pure function of an RNG handed to it (see Parallelism): it
holds shared read-only data at most, never an RNG or other mutable state.

Fault selection is not a seam. It is an enum: uniform over all cases, or one
fixed case. Nothing else is foreseen.

## When to stop

Experience from an earlier prototype: the head of the histogram, holding
essentially all the mass, stabilises early, but new masks keep appearing
indefinitely and many of them are seen exactly once. "No new masks recently"
never happens, so it cannot be the criterion. Two statistics that work:

- **Missing mass**, `masks seen exactly once / evaluations` (the Good-Turing
  estimate): the probability that the next sample is a mask never seen
  before. This is what matters to the consumer. If it is `1e-5`, a campaign
  of a million injections expects about ten draws from masks the file does
  not contain. The tail keeps growing, but its total weight is bounded and
  this tracks it.
- **Self-split distance**. The true distribution is unknown, so the closest
  thing to "how far off are we" is "how much do two independent estimates
  disagree". Evaluations alternate between two halves, A and B, each a full
  independent run at half the budget; the saved histogram is their sum. The
  statistic is the total variation distance between A and B as fractions:
  `sum |p_A(mask) - p_B(mask)| / 2`, which is the largest disagreement
  between the halves about the probability of any set of masks. It is
  computed over the head only, masks whose combined count is at least a
  small threshold, because a mask seen once lands in one half and not the
  other and every such mask adds about `1/N`; over all masks the distance
  would just be the missing mass again. Restricted to the head it measures
  the thing the first statistic does not: whether the bulk is stable.

Headless runs stop when both are below thresholds given on the command line.
The UI shows both live so the thresholds can be calibrated by eye before
being trusted headless.

Masks seen once stay in the output. The consumer samples them at `1/N`, which
is the right weight.

## Output

JSON. One file holds one histogram: the `(mask, count)` list, the seed, the
evaluation count, and the two statistics above at stop time. Masks and counts are plain integers. Counts
rather than fractions, so histograms can be added together later without
loss.

Chosen over a binary or array format because at tens of thousands of entries
size is irrelevant, and being able to open the file and read it is worth
more. The consumer loads it from Python with the standard library.

Fuller provenance (netlist hashes, port names, a description of the input
source) is deferred. It is not needed for the questions this crate exists to
answer right now.

## Parallelism

Evaluations are independent, so generation runs on as many workers as there
are cores. The design goal is that the result depends on the seed alone, not
on the worker count, the batch size or scheduling.

Jobs are numbered. Job `k` is `(input triple, fault case)`, both drawn from
an RNG seeded from `(seed, k)`. Any worker can produce any job by itself, so
there is no shared input stream: workers claim contiguous batches of job
indices from an atomic counter, and each owns a clone of the netlist
simulation and collects its batch as a list of syndromes. Seeding a small
RNG per job costs nanoseconds against a netlist evaluation.

The input source seam follows from this: a source is "given an RNG, produce
a triple", holding no RNG of its own. An adaptive source that chooses inputs
based on results so far would not fit, and would need a producer thread
instead; the worker side would be unchanged.

Workers hand finished batches to an aggregator, which records them into the
master histogram strictly in job order, buffering any batch that arrives
early. The stopping rule is asked at a fixed interval of jobs, independent
of the batch size. It and the UI snapshot only ever see the committed
prefix "jobs `0..M`", so the decision to stop at `M` depends on the job
sequence, not on timing.
Work past `M` is discarded, at most a batch per worker, and so is any
evaluation error in it: a failure is reported only when its job would have
been committed. The self-split halves are job parity.

Result: given the seed, both the histogram and the evaluation count at stop
are fully determined.

## User interface

Optional, behind a flag. The UI pulls a snapshot (sorted entries plus
statistics) from the aggregator at a fixed rate and never touches the live
histogram. Headless mode is the same workers with no UI attached.

The earlier prototype ran the simulation inside the frame loop and rebuilt,
cloned, sorted and labelled every bar from the live map on every frame, which
is why it fell over at large entry counts. The number of distinct syndromes
is not limited; only the number of bars drawn is.

Two views: syndromes sorted by frequency, and per-bit flip frequency. Both
can show counts or fractions.

A second histogram can be loaded from a file for comparison. Bars are drawn
rank by rank next to the primary; when the mask at a rank is the same in
both, the bar is highlighted. The header shows the total variation distance
between the two. Compare mode defaults to fractions, since evaluation counts
differ. The same view works on two files with no simulation running.

Colours are constants in one module: primary, primary for the zero mask,
secondary, secondary for the zero mask, and a highlight variant of each.

## Layout

A library and a binary in one crate. The library holds the netlist wrapper,
the two seams, the histogram with its statistics, the worker and aggregator,
and the file format. The binary holds the command line and the UI.

## Implementation order

Each step is reviewable on its own and leaves the crate building.

1. Library core: bit type, netlist wrapper (load, assign constants,
   evaluate), computer seam with both implementations, fault selection,
   histogram with the two statistics. Tested against small hand-written
   netlists.
2. Input sources: random bits, constant and normal first, then the
   profiling artifact reader and the independent-marginals variant.
3. Workers and aggregator, headless command line, stopping rule, JSON
   output.
4. UI on saved files only: one or two histograms, both views, compare mode.
5. UI attached to live generation.

## Open questions

- Stopping thresholds, and the head cut-off for the self-split distance. To
  be read off the UI on the real netlists before headless runs are trusted.
- Whether the independent-marginals and normal sources change the
  histogram. That is the experiment they exist for.
