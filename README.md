# FaultForge

Fault mitigation simulations for PyTorch models.

## Highlights

- Apply [various encodings](#encoding-formats) on PyTorch models.
- A CLI for running fault injection experiments with the provided techniques
- Designed to be [extended](#library-architecture). Supports:
  - Custom encodings
  - Custom DNN models
  - Custom test benches.
- Many common DNN models supported out of the box for ImageNet and CIFAR10/100.

## Installation

The only system dependency is a rust compiler (only for installation).
You can install rust using your system package manager or with [rustup](https://rustup.rs/).

To install the faultforge package into your virtual environment, run:

```sh
pip install 'git+https://github.com/rezzubs/faultforge.git[all]'
```

The main branch is kept in sync with the latest version. To install a specific version (tag) run:

```sh
pip install 'git+https://github.com/rezzubs/faultforge.git@0.1.0[all]'
```

The tag can also be replaced with a commit hash

### Optional features

Some features are optional and dependencies for them are not installed by default.
These can be enabled by specifying the relevant groups within during pip install (`[group1,group2]`)

| group name | description |
| - | - |
| all | Install all optional dependencies |
| imagenet | Add support for ImageNet models |
| cli | Enables the usage of the CLI |
| cifar | Add support for CIFAR models |

By default, only the dependencies required by the core library installed.

## CLI Usage

The CLI is documented in help pages. For a comprehensive overview of the options see:

```sh
faultforge --help
faultforge record --help
...
```

### General Flow

User choices:
1. Choose DNN model
2. Choose zero or more encodings.

Application:
1. Load dataset and model parameters.
2. Encode parameters.
3. Inject faults into encoded data.
4. Decode the faulty parameters.
5. Evaluate the decoded model to get an accuracy metric.
6. Compare decoded faulty parameters with the originals (Unless the `--skip-comparison` flag is given)
7. Restore the model to a non-faulty state and repeat from step 3 onwards until an end condition is met.

If no encoding is chosen then steps 2 and 4 are skipped.

In order to record new experiment data the `record` subcommand should be used.
A [model](#supported-models) should be selected with the `--cifar-model` or `--imagenet-model` options, the dataset is implied.
See the [Encoding Formats](#encoding-formats) section for details about encoding.

> [!TIP]
> Use the `--runs` option to control the number of runs and the `--summary` flag to print a short summary after each run.

### CIFAR

The CIFAR datasets and models will be downloaded automatically.

### ImageNet

> [!IMPORTANT]
> While the models for ImageNet will be loaded automatically the dataset needs to be prepared manually.


The dataset can be given with the `--imagenet-path` option.
The value of that option is expected to be a directory that contains:
- `images`: A directory containing jpg images.
- `name_to_id.json`: A mapping from image file names to the corresponding label IDs.

Example `name_to_id.json`:
```json
{
  "ILSVRC2012_val_00049927.JPEG": 887,
  "ILSVRC2012_val_00028335.JPEG": 685,
  "ILSVRC2012_val_00041118.JPEG": 202
}
```

> [!TIP]
> The `--imagenet-limit` option can be used to specify the maximum number of images to load.


## Encoding Formats

### SECDED

Single Error Correction Double Error Detection.

Relevant CLI options:
- `--secded`
- `--secded-chunk-size`

An encoding format based on [Hamming Codes](https://en.wikipedia.org/wiki/Hamming_code).
The memory for the model parameters is chunked based on the `--secded-chunk-size` parameter.
The chunk size determines how many data bits are protected by a single hamming code, this maps to the memory line width in hardware.
Any positive integer can be used for the chunk size but multiples of 8 perform better in terms of CPU time.
64 and 128 are the most common values in common [ECC memory](https://en.wikipedia.org/wiki/ECC_memory).

The double error detection results are ignored right now.

<details>
<summary>Bit pattern based SECDED</summary>
Optionally a `--secded-bit-pattern` option can be specified to first partition the memory into protected and non-protected regions.
The *protected* regions will be encoded using the method described above and the *unprotected* regions will be placed as is after the protected data.
</details>

### MSET

Most Significant Exponent bit Triplication.

Relevant CLI options:
- `--mset`

This method takes the second highest bit of each DNN parameter (bit 30 or 14 depending on the data type) and copies it into two lower bits.
A majority voting scheme will be used during decoding to determine the final value of the exponent bit.

This method works with the assumption that the highest exponent bits of DNN parameters are significantly more vulnerable compared to other bits.
Having errors in those bits tends to lead to worse accuracy than errors in other bits.
Additionally, the lower bits of floating point numbers have a miniscule effect on the final value.
MSET uses these assumptions to provide strong protection to only the most vulnerable bit without needing additional memory (like with ECC).
From our experimentation it seems like MSET can achieve equivalent or better resilience than SECDED.

### CEP

Chunked Embedded Parity

Relevant CLI options:
- `--embedded-parity`
- `--embedded-parity-scheme`

This technique was built on the conclusions of MSET experiments.
Additionally, the values of DNN parameters are usually fairly small, if a bit flips from 0 to 1 it will increase the value.
From what we observed, these kinds of flips are usually more significant when it comes to the accuracy.
This is especially true for the high exponent bits which are almost always 0 anyway.

CEP divides the "more useful" data bits into even sized chunks and sets a parity bit for each chunk in the less significant bits.
During decoding if a chunk's parity doesn't match the precomputed value then all bits in the chunk will be set to zero.
The parity bits themselves will be set to zero during decoding regardless of any faults.
Compared to MSET, CEP makes it possible to mitigate many faults in a single parameter.
This property gives it the highest accuracy, especially for high bit error rates even though setting a whole chunk to zero causes more actual flips compared to no protection in the majority of cases.

There are three schemes that distribute evenly in both 16 and 32 bit buffers.
- 1 parity bit for 3 data bits
- 1 parity bit for 7 data bits
- 1 parity bit for 15 data bits

From our experiments, the first scheme gives a significantly better accuracy.
It makes sense because it can tolerate more faults per parameter and single faults don't zero large chunks of the original data.
The other schemes should only be considered if the lower bits that would otherwise be used for parity are deemed too important to overwrite. 


## Supported Models

TODO

## Library Architecture

TODO
