# Encoded Memory

A [FaultForge](https://github.com/rezzubs/faultforge) experiment: fault
injection into ECC-protected model parameters. Installing it provides both
the `EncodedFaultInjection` library API and an `encoded-memory` CLI command
for recording results.

`EncodedFaultInjection` runs a model whose parameters are stored through an
[`Encoder`](../../docs/library.md#encoder--encoding), injects bit flips (or
stuck-at faults) into that encoded memory, and scores the result according to
a [`Metric`](../../src/faultforge/metric.py): `Accuracy`, `AccuracyDegradation`, `Sdc` (Silent Data
Corruption - any output logit changed vs. a golden model), `Top1Sdc`
(Critical SDC - the top-1 prediction changed vs. a golden model) or
`Perplexity` (for language models). `AccuracyDegradation`, `Sdc` and
`Top1Sdc` additionally require a golden (unencoded, or golden-encoded)
reference run.

It's the reference [`Experiment`](../../docs/library.md#experiment)
implementation to follow when adding a new one.

This document covers the encoding techniques it can use, the
`encoded-memory` CLI, and using it directly as a library.

## Installation

This experiment isn't published to PyPI but doesn't depend on any native code
other than what `faultforge` provides. `faultforge` itself is a pinned, regular
dependency so it installs straight from its prebuilt PyPI wheel. Point pip at
this subdirectory of whichever repository revision you want:

```sh
# latest release (the latest branch tracks the most recent tagged release)
pip install 'encoded-memory @ git+https://github.com/rezzubs/faultforge.git@latest#subdirectory=experiments/encoded_memory'

# a specific released version
pip install 'encoded-memory @ git+https://github.com/rezzubs/faultforge.git@vX.Y.Z#subdirectory=experiments/encoded_memory'

# a specific commit
pip install 'encoded-memory @ git+https://github.com/rezzubs/faultforge.git@<commit-sha>#subdirectory=experiments/encoded_memory'

# development (main is kept in sync with the newest code going forward)
pip install 'encoded-memory @ git+https://github.com/rezzubs/faultforge.git#subdirectory=experiments/encoded_memory'
```

The CLI can also be run directly with [`uv`](https://docs.astral.sh/uv/),
without an explicit install step - `uvx` fetches it into a throwaway
environment on first use and reuses that environment on later calls:

```sh
uvx --from 'git+https://github.com/rezzubs/faultforge.git#subdirectory=experiments/encoded_memory' encoded-memory <command> --help
```

Swap in `@latest`, `@vX.Y.Z`, or `@<commit-sha>` the same way as the `pip
install` variants above to pin a revision.

Requires Python 3.14 or newer.

## Encoding techniques

An encoded-memory experiment protects parameters with an
[`Encoder`](../../docs/library.md#encoder--encoding). FaultForge has three
built-in techniques, plus `IdentityEncoder` as an unprotected baseline. All
of them share the `Encoder`/`Encoding` interface, so they can also be
composed with `EncoderSequence` (e.g. an MSET or CEP pass followed by a
SECDED pass over the result).

### SECDED

Single Error Correction, Double Error Detection - Hamming codes.

- CLI: `--secded N`
- Library: `faultforge.encoding.SecdedEncoder(bits_per_chunk=N)`

Parameter memory is chunked into groups of `bits_per_chunk` data bits, each
protected by its own Hamming code (any positive integer works, but multiples
of 8 perform better). This mirrors the memory line width in real ECC memory;
64 and 128 are the most common real-world values. Single-bit errors within a
chunk are corrected during decoding; double-bit errors are detected but the
detection result isn't currently used (a DED-triggering buffer is decoded
as-is rather than flagged).

### MSET

Most Significant Exponent bit Triplication.

- CLI: `--mset`
- Library: `faultforge.encoding.MsetEncoder()`

Copies the second-highest bit of each parameter (the top exponent bit) into
the two lowest bits. A majority vote across the three copies determines the
decoded value. This targets the observation that the highest exponent bits
of floating-point DNN parameters are disproportionately vulnerable - errors
there tend to do far more damage to accuracy than errors in the lower,
near-mantissa bits, which MSET leaves unprotected and reuses for the vote.

### CEP

Chunked Embedded Parity.

- CLI: `--cep` (with `--cep-scheme`)
- Library: `faultforge.encoding.CepEncoder(scheme=...)`

Divides each parameter's higher, more consequential bits into evenly-sized
chunks and embeds a parity bit for each chunk in the lower bits. If a
chunk's parity doesn't match during decoding, every bit in that chunk is set
to zero (the parity bits themselves are always zeroed on decode). Compared
to MSET, CEP can mitigate multiple faults within a single parameter, at the
cost of zeroing a whole chunk on any detected mismatch.

`CepScheme` controls the data-to-parity-bit ratio, all evenly distributing
across 16- and 32-bit buffers:

| Scheme  | Data bits per parity bit |
| ------- | ------------------------- |
| `D3P1`  | 3 (default)                |
| `D7P1`  | 7                          |
| `D15P1` | 15                         |

`D3P1` typically gives the best accuracy: more, smaller chunks per parameter
means a single fault zeroes less data and more faults can be tolerated
overall. The wider schemes are only worth considering if the lower bits
`D3P1` would use for parity are too significant to give up.

## CLI usage

Installing `encoded-memory` (see [Installation](#installation) above)
provides an `encoded-memory` command:

```sh
encoded-memory <command> --help
```

### `record`

Runs an encoded-memory experiment and records the results. Flags are grouped
into help panels:

- **Model Setup**: `--model`, `--batch-size` (required, since a good value
  depends on the model and the hardware), `--dataset`
  (`cifar10`/`cifar100`/`imagenet`/`wikitext`), `--imagenet-root` (required
  for `imagenet`), `--model-revision`, `--sequence-length` (required for
  `wikitext`) and `--stride` (`wikitext` only), `--preload-batches`,
  `--batch-limit`, `--f16` (run in float16 instead of float32, halving the
  encoded bit width per parameter), `--reliability-metric` (defaults to
  `perplexity` for `wikitext` and `accuracy` otherwise).
- **Fault Injection**: `--bit-error-rate`/`--ber` and `--faults` (mutually
  exclusive - a rate or an exact count), `--compare-bitwise` (record
  per-run XOR bitmasks against the golden model),
  `--fault-summary` (print a per-run histogram of bits flipped and, with
  `--compare-bitwise`, affected/masked counts).
- **Encoding Settings**: `--secded N`, `--mset`, `--cep` (with
  `--cep-scheme`), `--golden-is-encoded` (compare against the non-faulty
  *encoded* model instead of the unencoded one).
- **Recording Settings**: `--output`, `--autosave` (seconds between
  autosaves), `--compress` (zstd, for new output files), `--overwrite`
  (discard a mismatched existing `--output` instead of aborting).
- **Stop Conditions**: `--runs`/`--max-runs`/`--min-runs` and the margin
  limits (see below) for controlling how long to run.
- **Misc Settings**: `--device`.

```sh
encoded-memory record \
  --dataset cifar10 --model resnet20 --batch-size 256 \
  --secded 64 \
  --bit-error-rate 1e-3 \
  --max-relative-margin 1.0 \
  --output result.json.zst --compress \
  --fault-summary
```

With `wikitext`, `--model` is any causal language model on the HuggingFace
Hub. `--model-revision` (default `main`) is resolved to a commit hash, which
is what gets recorded.

```sh
encoded-memory record \
  --dataset wikitext --model openai-community/gpt2 \
  --batch-size 8 --sequence-length 1024 \
  --secded 64 \
  --bit-error-rate 1e-6 \
  --runs 100 \
  --output gpt2.json.zst --compress
```

#### Margin limits

`--max-relative-margin N` and `--max-absolute-margin N` run until an
estimate's 95% margin of error is at most N, as a percentage of the estimated
value or in the estimate's own units. With both, whichever is reached first
stops. `--min-runs` delays the check, `--max-runs` caps the run count.
`--estimate` picks what the limits apply to:

- `mean` (default) - the mean score. Stops with an error at the first
  non-finite score, since the mean of scores that can diverge (like an
  infinite perplexity) isn't meaningful.
- `failure-rate` - the percentage of runs whose score is worse than a failure
  threshold, in percentage points for `--max-absolute-margin`. Non-finite
  scores always count as failures, so it works for scores that can diverge.
  Which direction is worse comes from the metric. Takes exactly one of:
  - `--relative-failure-threshold N` - worse than N times the golden model's
    score, e.g. `2.0` for perplexity. The golden score is computed once with
    an extra inference pass and saved to `--output`. It must be finite and
    positive, so this doesn't work for metrics whose golden score is 0
    (`accuracy-degradation`, `sdc`, `top1-sdc`).
  - `--absolute-failure-threshold N` - worse than N in the metric's units,
    e.g. `5.0` for 5% SDC.

A relative limit is never reached while the estimate is 0, e.g. a failure
rate without failures. Combine it with `--max-absolute-margin` or
`--max-runs` in that case. The absolute margin of a failure rate always
reaches its limit eventually.

```sh
encoded-memory record \
  --dataset wikitext --model openai-community/gpt2 \
  --batch-size 8 --sequence-length 1024 \
  --secded 64 \
  --bit-error-rate 1e-6 \
  --estimate failure-rate --relative-failure-threshold 2.0 \
  --max-absolute-margin 2.0 \
  --output gpt2.json.zst --compress
```

### `check`

Reports whether saved result files need more runs, without loading the model
or dataset. It takes the stop conditions of `record` (except `--runs`) and
applies them to every file. At least one of the margin limits or `--max-runs`
is required. Useful before submitting jobs to a queue:

```sh
encoded-memory check results/*.json.zst \
  --estimate failure-rate --relative-failure-threshold 2.0 \
  --max-absolute-margin 2.0 --max-runs 1000
```

A directory checks every `.json`, `.json.zst` and `.json.zstd` file below it,
in sorted order. Symlinked directories aren't followed.

It prints a JSON list to stdout, with one object per path:

```json
[
  {
    "path": "results/secded.json.zst",
    "status": "stable",
    "reason": "Reached absolute margin of error 2.00% (1.85%)",
    "run_count": 412,
    "estimate": {"value": 3.4, "lower": 1.9, "upper": 5.25}
  }
]
```

The `status` is one of:

- `stable` - the margin limits are reached.
- `max-runs` - `--max-runs` is reached.
- `incomplete` - neither is reached yet. If the margins need a golden score
  that the file doesn't have yet, the `reason` says so.
- `missing` - the file doesn't exist. Only for paths given directly, since a
  directory only lists files that exist.
- `error` - the file couldn't be checked. The object has an `error` message
  instead of the other fields.

The exit code is 0 if no file needs more runs, 1 if any is `incomplete` or
`missing` and 3 if any is an `error`. The worst one wins. 2 means invalid
arguments.

From Python, `check_saved` does the same for one file:

```python
from encoded_memory import StabilityLimits, check_saved
from faultforge.experiment import RelativeThreshold

limits = StabilityLimits(
    failure_threshold=RelativeThreshold(2.0), max_absolute_margin=2.0
)
result = check_saved("results/secded.json.zst", stability=limits, max_runs=1000)
if result.should_run():
    ...
```

### `discard-bitmasks`

Shrinks a saved result file by dropping its recorded bitmasks in place,
without needing to reload the model or dataset that produced it:

```sh
encoded-memory discard-bitmasks result.json.zst
```

## Library usage

`EncodedFaultInjection` can be used directly, without the CLI:

```python
from encoded_memory import EncodedFaultInjection
from faultforge.encoding import SecdedEncoder
from faultforge.experiment import MaxRuns
from faultforge.loading import Cifar, CifarDataset, CifarModel
from faultforge.metric import Sdc

bundle = Cifar(model=CifarModel.ResNet20, dataset=CifarDataset.Cifar10)
encoder = SecdedEncoder(bits_per_chunk=64)

experiment = EncodedFaultInjection(
    bundle,
    encoder,
    Sdc(),
    batch_size=256,
    faults=1e-3,  # a bit error rate; an int is treated as an exact fault count
    compare_bitwise=True,
    fault_summary=True,
)
experiment.run_loop(stop_conditions=[MaxRuns(total=50)])
experiment.save_atomic("result.json")
```

`Sdc`, `Top1Sdc` and `AccuracyDegradation` each additionally run
a golden (by default unencoded; pass `golden_is_encoded=True` to compare
against the non-faulty *encoded* model instead) reference pass to score
against.

A saved result can be inspected without reconstructing the model or
dataset that produced it, via `SavedResult`:

```python
from encoded_memory import SavedResult

saved = SavedResult.load("result.json")
saved.scores()  # every recorded run's score, in run order
saved.metric_display_name  # the name of the metric it was recorded with
saved.bit_error_rate()  # faults / total_bits
```

When `compare_bitwise=True`, results are `DetailedResults` (per-run bitmasks
included) rather than `SimpleResults`; `DetailedResults.discard_bitmasks()` (or
the module-level `discard_bitmasks_in_file(path)`, which rewrites a saved
file in place) drops them once they're no longer needed, shrinking the
result down to a `SimpleResults`.
