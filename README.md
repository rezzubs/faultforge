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

Implementation: `faultforge.encoding.secded`

An encoding format based on [Hamming Codes](https://en.wikipedia.org/wiki/Hamming_code).
The memory for the model parameters is chunked based on the `--secded-chunk-size` parameter.
The chunk size determines how many data bits are protected by a single hamming code, this maps to the memory line width in hardware.
Any positive integer can be used for the chunk size but multiples of 8 perform better in terms of CPU time.
64 and 128 are the most common values in common [ECC memory](https://en.wikipedia.org/wiki/ECC_memory).

The double error detection results are ignored right now.

<details>
  
<summary>Bit pattern based SECDED</summary>
  
Implementation: `faultforge.encoding.bit_pattern`

Optionally a `--secded-bit-pattern` option can be specified to first partition the memory into protected and non-protected regions.
The *protected* regions will be encoded using the method described above and the *unprotected* regions will be placed as is after the protected data.
</details>

### MSET

Most Significant Exponent bit Triplication.

Relevant CLI options:
- `--mset`

Implementation: `faultforge.encoding.mset`

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

Implementation: `faultforge.encoding.embedded_parity`

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

These are the models with built-in support.
If usage with another model is desired see the [System](#system) section in Library Architecture, consider contributing the additions with a pull request.

### ImageNet

A combination of torchvision and hugging face models is used

- deit_tiny_patch16_224
- deit_base_patch16_224
- swin_tiny_patch4_window7_224
- vit_base_patch16_224
- vit_tiny_patch16_224
- inception_v3
- mobilenet_v2
- resnet152

### CIFAR

All the models from https://github.com/chenyaofo/pytorch-cifar-models.
If a new model is added there and is missing here please open an issue.

## Library Architecture

### System

The fundamental building block of the library is the `System` base class in `faultforge.system`.
A `System` represents a neural network model and its associated data.
A `System` is also a complete test bench of sorts, being able to run itself to get an accuracy metric.

> [!NOTE]
> Anything that has the following properties can be used as a system:
> - Have core **data** that can be represented as or converted to a list of PyTorch tensors.
> - Be able to generate an accuracy metric from an instance of that **data**.
> - Perform fault injection on that data

An example of a **system** is a class which stores a pytorch DNN model and a dataset.
The **data** in this case is the root **module** (`nn.Module`) of the model.
The **module** is the appropriate value for **data** because it is the part that should be affected by fault injection.
Given a (possibly faulty) instance of the root module, the **system** can evaluate it using the stored dataset to give an accuracy metric.
This is how `faultforge.cifar.system.System` and `faultforge.imagenet.system.System` are implemented.

> [!TIP]
> If system uses tensors for storing the parameters inside **data** (like is the case for `nn.Module`), then the default implementation of fault injection is likely sufficient.
> 

For details about implementing a new **system** see the docstrings of the base class and it's methods in `faultforge.system`.

#### `EncodedSystem`

`EncodedSystem`, defined in `faultforge.encoding.system` is an subclass `System` which wraps an existing **system** and encodes its **data** using one of many encoders (see the [Encoding API](#encoding-api) section).
It can be used in all the places where other **systems** are used. The **data** of `EncodedSystem` is the encoded **data** of the child system.
Fault injection will be performed on the encoded **data** and methods like `system_data_tensors` and `system_accuracy` will first decode the **data** and delegate to the corresponding methods of the child **system**.

### Encoding API

A new encoding is added by creating two complimentary classes - an **encoder** and an **encoding**.
These classes need to subclass the corresponding base classes defined in `faultforge.encoding`.
- The **encoder** is responsible for encoding a list of tensors and returns an instance of the matching **encoding**.
- The **encoding** needs to be able to decode into the same shape as the original tensors. The **encoding** is also responsible for fault injection into its *encoded* data.

See the [Encoding Formats](#encoding-formats) section for available **encoders**.

#### `TensorEncoding` and `Sequence`

In a lot of cases the **encoding** might also use tensors as the storage type.
In this case it's recommended to subclass `TensorEncoding` and `TensorEncoder` instead (defined in `faultforge.encoding.sequence`).
Doing so enables the sequential application of such **encoders** using `SequenceEncoder` (the final link of a sequence can still be a regular encoder).

### Experiment

Fault injection experiments for a **system**.

The `Experiment` class (from `faultforge.experiment`) defines the fault configuration for a **system** and performs runs of fault injection on it.
At the end of each run the faulty parameters will be compared with the originals to determine how many faults were masked and where faults got through the encoding.

A run results in an `Entry` within the `Experiment` which stores the run accuracy and masks for faults.
A mask is a bitwise xor between the faulty parameters and the originals.
For example `0b0101` would mean that bits 0 and 2 were flipped.

The experiment contains a metadata dictionary to determine its uniqueness.
A **system**'s metadata will be appended.

> [!TIP]
> At the end of a run a short summary like this can be printed:
> ```
> Flipped 204458/204458400 bits - BER: 1.00e-03
> Accuracy: 0.10%
> 9717 parameters were affected
> 12614 bits were measured faulty (93.83% masked)
> 6879 parameters had 1 faulty bit
> 2790 parameters had 2 faulty bits
> 37 parameters had 3 faulty bits
> 11 parameters had 4 faulty bits
> ```
> This is for a run with bit error rate `1e-3` and SECDED encoding with 64 bit chunks.

### Utilities

The library exposes some tensor related utility functions such as fault injection in `faultforge.tensor_ops`.

### Rust

A lot of the heavy computation parts of the library are written in rust. Designed for flexibility but keeping performance in mind, using multithreading where appropriate. The rust API is not meant to be consumed directly and is not versioned. The internal python bindings are defined in [crates/faultforge-bindings](crates/faultforge-bindings) whereas the core implementation can be found in [crates/faultforge](crates/faultforge).
