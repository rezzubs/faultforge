# Encoded Memory

`EncodedFaultInjection` (`faultforge.experiments.encoded_memory`) runs a
model whose parameters are stored through an
[`Encoder`](../library.md#encoder--encoding), injects bit flips (or stuck-at
faults) into that encoded memory, and scores the result according to a
`ReliabilityMetric`: `Accuracy`, `AccuracyDegradation`, `Sdc` (Silent Data
Corruption - any output logit changed vs. a golden model), or `Top1Sdc`
(Critical SDC - the top-1 prediction changed vs. a golden model). The last
three metrics additionally require a golden (unencoded, or golden-encoded)
reference run.

It's the one built-in [`Experiment`](../library.md#experiment) today, and
it's meant to be read as the reference implementation to follow when adding
a new one.

This document covers the encoding techniques it can use, the
`faultforge encoded-memory` CLI, and using it directly as a library.

## Encoding techniques

An encoded-memory experiment protects parameters with an
[`Encoder`](../library.md#encoder--encoding). FaultForge has three built-in
techniques, plus `IdentityEncoder` as an unprotected baseline. All of them
share the `Encoder`/`Encoding` interface, so they can also be composed with
`EncoderSequence` (e.g. an MSET or CEP pass followed by a SECDED pass over
the result).

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

Installing `faultforge-cli` provides a `faultforge` command; all
encoded-memory functionality lives under its `encoded-memory` subcommand:

```sh
faultforge encoded-memory <command> --help
```

### `record`

Runs an encoded-memory experiment and records the results. Flags are grouped
into help panels:

- **Model Setup**: `--model`, `--dataset` (`cifar10`/`cifar100`/`imagenet`),
  `--imagenet-root` (required for `imagenet`), `--batch-size`,
  `--preload-batches`, `--batch-limit`, `--f16` (run in float16 instead of
  float32, halving the encoded bit width per parameter), `--reliability-metric`.
- **Fault Injection**: `--bit-error-rate`/`--ber` and `--faults` (mutually
  exclusive - a rate or an exact count), `--compare-bitwise` (record
  per-run XOR bitmasks against the golden model; required by `heatmap`),
  `--fault-summary` (print a per-run histogram of bits flipped and, with
  `--compare-bitwise`, affected/masked counts).
- **Encoding Settings**: `--secded N`, `--mset`, `--cep` (with
  `--cep-scheme`), `--golden-is-encoded` (compare against the non-faulty
  *encoded* model instead of the unencoded one).
- **Recording Settings**: `--output`, `--autosave` (seconds between
  autosaves), `--compress` (zstd, for new output files), `--overwrite`
  (discard a mismatched existing `--output` instead of aborting), `--runs`/
  `--max-runs`/`--min-runs`/`--stability-threshold` for controlling how long
  to run.
- **Misc Settings**: `--device`.

```sh
faultforge encoded-memory record \
  --dataset cifar10 --model resnet20 \
  --secded 64 \
  --bit-error-rate 1e-3 \
  --stability-threshold 1.0 \
  --output result.json.zst --compress \
  --fault-summary
```

### `compare`

Plots reliability score vs. bit error rate, one line per configuration:

```sh
faultforge encoded-memory compare result-a.json result-b.json=Unprotected --row-by model
```

Accepts result files and/or directories (searched recursively), and
automatically clusters results into "configurations" - everything in their
`Fingerprint` matching except the bit error rate - so several recorded bit
error rates of the same model/encoder/dtype become one line regardless of
which file they came from. `--row-by`/`--col-by` facet the plot grid by a
fingerprint-derived key (`Model`, `Dtype`, `Dataset`, `Metric`); `--percentile`
plots a percentile instead of the mean; `--log-x` and `--output` control the
axis scale and where the figure is saved (shown interactively otherwise).

### `heatmap`

Plots per-run score density: one panel against total residual (post-decode)
faults, one against the faulty bit position. Requires results recorded with
`--compare-bitwise`. Options include `--bins`, `--min-score`/`--max-score`
and `--max-total-faults` (drop outlier runs), `--skip-multi-bit-faults`, and
`--output`.

### `discard-bitmasks`

Shrinks a saved result file by dropping its recorded bitmasks in place,
without needing to reload the model or dataset that produced it:

```sh
faultforge encoded-memory discard-bitmasks result.json.zst
```

## Library usage

`EncodedFaultInjection` can be used directly, without the CLI:

```python
from faultforge.encoding import SecdedEncoder
from faultforge.experiment import MaxRuns
from faultforge.experiments.encoded_memory import (
    EncodedFaultInjection,
    ReliabilityMetric,
)
from faultforge.loading import Cifar, CifarDataset, CifarModel

bundle = Cifar(model=CifarModel.ResNet20, dataset=CifarDataset.Cifar10)
encoder = SecdedEncoder(bits_per_chunk=64)

experiment = EncodedFaultInjection(
    bundle,
    encoder,
    ReliabilityMetric.Sdc,
    faults=1e-3,       # a bit error rate; an int is treated as an exact fault count
    compare_bitwise=True,
    fault_summary=True,
)
experiment.run_loop(stop_conditions=[MaxRuns(total=50)])
experiment.save_atomic("result.json")
```

`ReliabilityMetric.Sdc`/`Top1Sdc`/`AccuracyDegradation` each additionally run
a golden (by default unencoded; pass `golden_is_encoded=True` to compare
against the non-faulty *encoded* model instead) reference pass to score
against.

A saved result can be inspected without reconstructing the model or
dataset that produced it, via `SavedResult`:

```python
from faultforge.experiments.encoded_memory import SavedResult

saved = SavedResult.load("result.json")
saved.scores()               # every recorded run's score, in run order
saved.reliability_metric()   # the ReliabilityMetric it was recorded with
saved.bit_error_rate()       # faults / total_bits
```

When `compare_bitwise=True`, results are `DetailedResult` (per-run bitmasks
included) rather than `SimpleResult`; `DetailedResult.discard_bitmasks()` (or
the module-level `discard_bitmasks_in_file(path)`, which rewrites a saved
file in place) drops them once they're no longer needed, shrinking the
result down to a `SimpleResult`.
