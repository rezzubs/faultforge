# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-11-19

0.4 adds language model support (WikiText, perplexity), configurable stop
conditions built on estimators, and makes SECDED fault injection scale to
LLM-sized models. Several of these changes break custom metrics, encoders and
stop conditions, and some saved `encoded-memory` results need a manual fix, see
below.

### Added

- `Perplexity` metric. `Perplexity.unigram_score` gives the perplexity of a
  unigram model fit on the scored targets, a reference point for a model that
  has stopped using its context. ([#40])
- `WikiTextBundle`, a `ModelBundle` for causal language models on WikiText,
  with `WikiTextConfig`, `WikiTextSplit` and `resolve_huggingface_revision`.
  ([a1d453e])
- `Metric.direction` and `ScoreDirection`, which tell whether higher or lower
  scores are better.
- Estimators: an `Estimator` summarizes the scores of many runs as an
  `Estimate`, a value with a 95% confidence interval.
  - `Mean` - Student's t interval. Rejects non-finite scores.
  - `FailureRate` - the share of runs worse than a `RelativeThreshold` (a
    factor of the golden score) or an `AbsoluteThreshold`, with a Wilson score
    interval. Non-finite scores count as failures.
  - `NonFiniteRate` - the share of runs with a non-finite score.
- `Experiment.golden_score` and `Experiment.estimate`.
- `RecordedRuns`, the scored part of an `Experiment`. Stop conditions can be
  checked against runs loaded from a file, not only a live experiment. ([#42])
- `StatusReporter`: stop conditions implementing it add to the status line.
- `Encoding.decode_into` and `Encoding.empty_decoded` decode into existing
  tensors on any device. ([#45])
- `Encoding.decode_changes_into` decodes into tensors that already hold the
  fault-free decode. SECDED only writes the bytes that faults changed. ([#45])
- `encoded-memory`:
  - `--dataset wikitext` and `--metric perplexity`. ([427df1e])
  - A `check` command that reports as JSON whether saved result files (or
    directories of them) need more runs, without loading the model or dataset.
    The exit code tells whether anything needs to run. `check_saved` does the
    same for one file from Python. ([#42])
  - `--estimate` (`mean`, `failure-rate`, `non-finite-rate`) chooses what the
    margin limits apply to, with `--max-relative-margin`,
    `--max-absolute-margin`, `--relative-failure-threshold` and
    `--absolute-failure-threshold`. Margin limits now work with perplexity.

### Changed

- **Breaking:** Custom metrics must implement `Metric.direction`.
- **Breaking:** `ExperimentDisplay.score_name`, `score_unit` and
  `format_score` are merged into `score_display`, which returns a
  `ScoreDisplay`. Every `Metric` is a `ScoreDisplay`, so a metric can define
  how its scores are formatted.
- **Breaking:** `Stability` takes an `Estimator` and limits its margin in
  absolute terms, relative to the value, or both. A `RelativeThreshold` that
  can't be used, e.g. with a golden score of 0 as with SDC, raises before the
  first run.
- **Breaking:** `ExperimentDisplay.format` takes `condition_status` instead of
  `mean`, `margin_of_error` and `stop_conditions`. The default status line no
  longer shows a mean.
- **Breaking:** `StopCondition` and `StatusReporter.status` take a
  `RecordedRuns` instead of an `Experiment`. ([#42])
- **Breaking:** `Encoder.encode` must not modify its inputs or keep references
  to them. `EncodedModule` no longer copies the parameters before encoding, so
  a custom encoder that modifies its inputs corrupts the module. ([#45])
- **Breaking:** `Encoding.decode_into` and `Encoding.empty_decoded` are
  abstract. `decode()` is no longer abstract and always returns new tensors.
  `IdentityEncoding.decode()` used to return its own tensors. ([#45])
- **Breaking:** `SecdedEncoder.bits_per_chunk` (and `encoded-memory --secded`)
  must be a positive multiple of 8. Fingerprints are unchanged, so existing
  results with valid sizes still resume. ([#45])
- **Breaking:** The parameters of a module returned by `EncodedModule.decode()`
  must not be modified. Later decodes only write what faults changed on top of
  them. ([#45])
- **Breaking:** `SecdedEncoder` raises `NotImplementedError` on big-endian
  hosts. Fault targets address the encoded bytes in little-endian order. ([#45])
- **Breaking:** The `AccuracyDegradation` fingerprint kind is now spelled
  `accuracy_degradation`. Saved results with `accuracy_degredation` need to be
  fixed manually.
- **Breaking:** `encoded-memory`:
  - `--stability-threshold` is replaced by the margin options above.
  - `--batch-size` is required.
  - The `test_image_limit` key in saved results is now `sample_limit`. Results
    recorded with 0.3 need to be updated manually.
- SECDED fault injection needs far less memory and time per run, enough to
  scale to LLM-sized models. A run only copies and decodes the parts of the
  encoding that its faults touch, so the gain shrinks as the bit error rate
  grows. ([#45])
- `EncodedModule` with SECDED, MSET or CEP uses less memory. It keeps a single
  decoded copy of the parameters, where 0.3 kept extra full copies. ([#45])

### Fixed

- SECDED corrects a single flip of the overall parity bit instead of reporting
  a double error. This changes double error detection results, not decoded
  data. ([#45])
- SECDED encoding no longer depends on the previous contents of the destination
  buffer. A set bit 0 could produce an encoding that decoded as a double error.
  ([#45])
- `MsetEncoder` and `CepEncoder` overwrote CPU input tensors with their encoded
  values. ([#45])
- `IdentityEncoder` stored its input tensors, so faults applied to the encoding
  changed the caller's tensors. ([#45])
- `Experiment.run_loop` saves results when it raises, instead of discarding
  every run since the last autosave.
- `Experiment` saves changes made outside of `run`. ([#42])
- `encoded-memory` saves and loads non-finite results. ([262d2f0])
- `encoded-memory` saves a golden score computed without new runs. ([#42])

### Removed

- **Breaking:** `Experiment.mean_score`, `Experiment.margin_of_error` and
  `faultforge.experiment.relative_margin_of_error`. Use `Experiment.estimate`
  with `Mean`.
- **Breaking:** `TensorEncoding.trigger_recompute`. No encoding caches decoded
  tensors anymore. ([#45])
- **Breaking:** The `encoded-memory` `compare` and `heatmap` plotting commands.
- The PyPy classifier. Only CPython is supported.

[#40]: https://github.com/rezzubs/faultforge/pull/40
[#42]: https://github.com/rezzubs/faultforge/pull/42
[#45]: https://github.com/rezzubs/faultforge/pull/45
[262d2f0]: https://github.com/rezzubs/faultforge/commit/262d2f0c07c0d8f660a8bf2b25df32bf3b466a23
[427df1e]: https://github.com/rezzubs/faultforge/commit/427df1e8135520bb8d8d9c92ab3043cad5b00ac1
[a1d453e]: https://github.com/rezzubs/faultforge/commit/a1d453e0eeaad1fa61d29130b49f4ffc3fba5df4

## [0.3.0] - 2026-09-29

### Added

- The `BatchedDataset` API now supports shuffling. ([#32])
- The `Progress` API now properly handles nested stages. ([#33])
- A reusable reliability metrics system under `faultforge.metric`. ([#34])

### Changed

- Experiments live in the repository under `experiments/` as their own packages.
  ([#30])
- `faultforge`'s Python source now lives directly at the
  repository root (`src/`, `tests/`) rather than nested under
  `packages/faultforge`/`faultforge/`. ([#30])
- **Breaking:** `faultforge encoded-memory <command>` is now `encoded-memory
  <command>`, shipped as part of the `experiments/encoded_memory` package.
  `faultforge-cli` is removed. ([#30])
- **Breaking:** `BatchedDataset` is now an iterable rather than an iterator. Not
  possible to forget to call `reset` anymore. ([#35])
- **Breaking:** Model/dataset dtype is bundle configuration, not a loading
  parameter. ([#36])

### Removed

- **`faultforge-cli`** is no longer published to PyPI or maintained as a
  separate package. ([#30])
- **Breaking:** `faultforge.loading.DEFAULT_DTYPE` ([#36])

[#30]: https://github.com/rezzubs/faultforge/pull/30
[#32]: https://github.com/rezzubs/faultforge/pull/32
[#33]: https://github.com/rezzubs/faultforge/pull/33
[#34]: https://github.com/rezzubs/faultforge/pull/34
[#35]: https://github.com/rezzubs/faultforge/pull/35
[#36]: https://github.com/rezzubs/faultforge/pull/36

## [0.2.1] - 2026-07-08

`0.2.0`'s release pipeline failed to publish `faultforge` and never
completed, so it's a dead/uninstallable version - `faultforge-cli==0.2.0`'s
`faultforge` dependency was also left unconstrained instead of pinned to a
matching version. `0.2.1` is the first fully published, correctly paired
release.

## [0.2.0] - 2026-07-08

0.2 is a more or less full rewrite of both the Rust and Python layers. If you
used 0.1's `System`-based API (`faultforge.system.System`,
`faultforge.cifar.system`, `faultforge.imagenet.system`, `EncodedSystem`), none
of that exists anymore. The Python package now lives in a `packages/faultforge`
+ `packages/faultforge_cli` uv workspace, with public top-level modules
(`faultforge.encoding`, `faultforge.experiment`, `faultforge.dataset`,
`faultforge.loading`, `faultforge.progress`, `faultforge.fingerprint`). The Rust
side moved from one crate to a `picker`/`memory`/`bindings` workspace.

### Added

- **`Fingerprint`**: a structural, serializable identity for an experiment's
  configuration (model, dataset, encoder, fault count, etc). Saved results now
  carry a fingerprint, and resuming/verifying a saved file against your current
  setup raises a precise `FingerprintError` (e.g.
  `encoder.head[0].bits_per_chunk: changed parameter`) instead of silently
  mixing incompatible data - 0.1 had no protection against this at all.
  Fingerprint scalars are normalized (e.g. `encoded_memory`'s fault count is
  always recorded as a resolved integer, whether you originally passed
  `--faults` or `--bit-error-rate`), so switching between equivalent ways of
  specifying the same config across a resume doesn't spuriously trip
  verification.
- **Stop conditions** (`StopCondition`, `MaxRuns`, `AdditionalRuns`,
  `Stability`): experiments can now stop themselves once a statistical stability
  threshold (95% CI margin of error, as a percentage of the mean) is reached,
  instead of always running a fixed, manually chosen number of iterations.
  Ctrl+C is itself a stop condition, so interrupting an experiment finishes the
  in-flight run cleanly instead of corrupting state.
- **Atomic, resumable saving** (`SaveConfig`, `Experiment.save_atomic`):
  periodic autosave during a long run, written via a temp file + rename so a
  crash mid-write can't corrupt the output file.
- **zstd-compressed save files**: `save`/`save_atomic`/`load_from` and the CLI's
  `--output` can transparently read/write zstd-compressed result files (detected
  by sniffing the file's magic bytes, not its extension); the new `--compress`
  flag only affects newly created files - an existing output file keeps whatever
  format it already had.
- **Saved-result inspection and visualization** for `encoded_memory`: a saved
  result file is now a public `SavedResult` type with `.load()`/`.scores()`/
  `.reliability_metric()`/`.bit_error_rate()`, so it can be inspected without
  reconstructing the model/dataset that produced it. The CLI's new `compare` and
  `heatmap` commands build on this to plot accuracy/reliability across a batch
  of saved runs, grouped by matching `Fingerprint`s - a from-scratch redesign of
  0.1's scatter/mean/configurations commands, which were dropped early in the
  Fingerprint rewrite and are now properly reinstated (plotting stays a
  `faultforge_cli`-only dependency; the `faultforge` library itself doesn't
  depend on matplotlib).
- **`StuckAt` faults**: alongside bit flips (`BitFlip`), faults can now be
  "stuck at 0/1" - a fault model 0.1 didn't support at all.
- **Resumable `Picker`**: the Fisher-Yates fault-location sampler can now
  reconstruct its state from a set of already-returned indices
  (`Picker.from_returned`) and is exposed directly to Python, so a long
  fault-injection campaign can be suspended and resumed without replaying
  already-picked locations.
- **Fault injection API reshaped around explicit, batchable `Fault` values**:
  0.1 exposed one function, `tensor_list_fault_injection(ts, faults_count)`,
  which always flipped `faults_count` random bits chosen internally by Rust - no
  control over which bits, and no non-flip fault types. 0.2 splits this into
  `Picker` (choose locations explicitly, see above) plus
  `tensor_list_fault`/`tensor_list_faults(ts, [(fault, target_bit), ...])`,
  applying one or many explicit `Fault`s (`BitFlip` or `StuckAt`) at
  caller-chosen bit positions in a single call (`Encoding.apply_faults`/
  `EncodedModule.apply_faults` do the same for encoded memory). A batched call
  still pays the tensor round-trip cost once for the whole set, same as 0.1's
  count-based calls did; what's newly improved is non-uniform/encoded sequences
  (MSET/CEP), which used to locate each fault's owning chunk with an O(n) linear
  scan and now do it in O(log n).
- **Per-run bitwise comparison and fault-injection summaries** in the
  `encoded_memory` experiment: `--compare-bitwise` records per-run XOR bitmasks
  against the golden model (with `discard-bitmasks`/ `discard_bitmasks_in_file`
  to shrink a saved result afterwards once the raw bitmasks aren't needed), and
  `--fault-summary` reports a histogram of bits flipped, parameters affected,
  and the fraction of faults masked by the encoding.
- **`Progress` reporting** (`faultforge.progress`): long operations (model
  download, dataset precompute, encoding, fault injection) now emit periodic
  "N/total (%) - elapsed" log messages instead of running silently.
- **`CachedDataset`/`precompute()`**: keeps all batches of a dataset in memory
  once, instead of re-loading/transforming them on every fault-injection run.
- A dedicated CLI package, `faultforge_cli` (`faultforge-cli encoded-memory
  record ...`), covering model/dataset setup, fault injection, encoding scheme
  selection, and recording/stop-condition flags - see `--help` (now also
  available as `-h`).

### Changed

- **`System` → `Experiment` + `ModelBundle` + `EncodedModule`.** 0.1's `System`
  was a single class mixing model loading, dataset loading, accuracy
  computation, and fault injection, with one subclass per domain (`CifarSystem`,
  `ImageNetSystem`, `EncodedSystem`). 0.2 splits these concerns: `ModelBundle`
  (`Cifar`, `ImageNet`) only knows how to load a model + its eval dataset;
  `EncodedModule` wraps any `torch.nn.Module` so its parameters live in
  simulated encoded memory; `Experiment` is a pure behavior ABC
  (`run`/`scores`/`serialize`/`deserialize`) with all the
  run-loop/stopping/saving machinery described above generic and reusable across
  experiment types. `faultforge.experiments.encoded_memory` is the one built-in
  experiment and a reference implementation for new ones.
- **Encoding schemes renamed and reorganized**, same underlying ideas: 0.1's
  "embedded parity" is now **CEP** (Chunked Embedded Parity), 0.1's "MSB"
  (odd-duplicate-bits + majority vote) is now **MSET** (Most Significant
  Exponent bit Triplication, generalized over chunk count), SECDED is
  unchanged in concept. All three plus `sequence` (compose encoders) and
  `identity` (unprotected baseline) now share common `Encoder`/`Encoding`/
  `TensorEncoder` ABCs with their own fingerprints.
- **Rust workspace restructured**: one `faultforge` crate +
  `faultforge-bindings` → three crates, `picker` (fault-location sampling),
  `memory` (bit buffers + ECC encodings), `bindings` (PyO3 glue).
- **Repo layout**: single Python package → `uv` workspace
  (`packages/faultforge`, `packages/faultforge_cli`); all dependencies
  (`matplotlib`, `pillow`, `timm`, `torch`, `torchvision`, `scipy`, ...) are now
  required by default rather than gated behind `all`/`cifar`/`imagenet`/`cli`
  extras.
- Dev tooling switched from `basedpyright` to `ty`.
- **`strip_faults` CLI command** - Now available as `discard-bitmasks`.

### Removed

- **Bit-pattern encoding** (0.1's `BitPattern`/`encoding_bit_pattern` bindings)
  - no equivalent in the new `memory` crate. This was mostly an experimental
  technique which wasn't as useful as initially hoped, Included in v0.1 for
  completeness sake but now removed to get rid of the maintenacne burden.
- **Standalone bitwise-comparison Rust bindings**
  (`compare_array_list_bitwise_*`) - comparison logic now lives on the
  Python side.

[Unreleased]: https://github.com/rezzubs/faultforge/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/rezzubs/faultforge/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/rezzubs/faultforge/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/rezzubs/faultforge/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/rezzubs/faultforge/compare/0.1.0...v0.2.0
[0.1.0]: https://github.com/rezzubs/faultforge/releases/tag/0.1.0
