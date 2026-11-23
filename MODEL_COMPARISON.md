# Model comparison and integration feasibility

Milestone 1: source inspection on 2026-10-31. Proposed integrations below are not installed, trained, or benchmarked. Hardware feasibility on 8 GB VRAM is unmeasured; no minimum-memory claims are made.

## Existing adapters

| Model | Modality / approach | Role and limitations | Execution / status |
| --- | --- | --- | --- |
| Bootstrap | Tabular resampling with perturbations | Cheap baseline; strong memorization risk; no learned complex distribution | CPU; exercised by tests |
| Gaussian Copula | Mixed tabular statistical model | Low-cost baseline; limited complex dependencies | CPU; direct adapter and API smoke checks passed |
| CTGAN | Mixed tabular GAN through SDV | Neural baseline; conditional requests through SDV; training stability/categorical cost must be measured | Explicit CPU/CUDA/auto; tiny real CPU fit/sample test passed in Milestone 2 |
| TVAE | Mixed tabular VAE through SDV | Neural baseline; conditional requests use SDV rejection sampling; rare conditions can fail explicitly | Explicit CPU/CUDA/auto; tiny real CPU fit/sample test passed in Milestone 2 |

SDV is a required application dependency today. The shared lifecycle now configures device/batch/epochs explicitly and never substitutes models. Its installed SDV version is 1.38.3; the requirements file does not pin it. Before redistributing adapters/checkpoints, record the precise installed SDV, CTGAN, RDT, and transitive licenses and notices. This audit does not establish blanket licensing permission for those dependencies.

## Proposed models

| Model | Modality / approach | Intended role | Main integration constraint |
| --- | --- | --- | --- |
| TabSyn | Mixed tabular VAE plus latent diffusion | Tested advanced tabular candidate | Two-stage training, separate preprocessing/checkpoints and pinned worker; tiny mixed-type fit/reload/sample passed; no conditional mode |
| TabDDPM | Numerical/categorical tabular diffusion | Second diffusion comparator | Legacy environment and prepared-array/config interface; CPU/GPU compatibility needs a real test |
| Diffusion-TS | Multivariate sequence diffusion | Tested advanced time-series candidate | Separate entity-aware windows and temporal metadata; fixed configured sequence shapes; tiny CPU and RTX 4060 worker smoke passed |
| TimeGAN | Recurrent adversarial sequence generation | `timegan_torch` comparative baseline | Original TensorFlow 1 implementation is incompatible; local PyTorch fixed-window reimplementation passed a tiny three-phase check and is not checkpoint-compatible |

## Official source findings

**TabSyn:** Apache-2.0. The official README targets Python 3.10 and illustrates PyTorch 2.0.1/CUDA 11.7. It trains a VAE before diffusion and samples through its CLI. Metadata identifies numerical, categorical, and target indices. Treat those as upstream settings, not a tested platform configuration. [Official README](https://github.com/amazon-science/tabsyn), [requirements](https://github.com/amazon-science/tabsyn/blob/main/requirements.txt), [license](https://github.com/amazon-science/tabsyn/blob/main/LICENSE).

Its requirements pin `sdmetrics==0.11.1` and `peft==0.9.0` and include unrelated baseline/evaluation packages. Installing the whole file into the application environment risks dependency changes. Prefer a dedicated Python 3.10 worker environment with a pinned source commit and a minimal, tested dependency set. Do not assume that the upstream target-bearing preprocessing directly supports unlabeled uploads.

**TabDDPM:** MIT. Upstream setup specifies Python 3.9.7 and PyTorch 1.10.1/CUDA 11.1; requirements include NumPy 1.21.4, pandas 1.3.4, sklearn 1.0.2, and SciPy 1.7.2. Its pipeline takes a TOML config and separate train/sample flags. These pins conflict with the application's installed stack. [Official README](https://github.com/yandex-research/tab-ddpm), [requirements](https://github.com/yandex-research/tab-ddpm/blob/main/requirements.txt), [license](https://github.com/yandex-research/tab-ddpm/blob/main/LICENSE).

Use an isolated worker or a documented compatibility patch. The upstream legacy CUDA setup must not be assumed to work on the RTX 4060. Test numerical-only, categorical-only, mixed, conditional, and unlabeled cases individually; retain the actual upstream transformation and inverse mapping.

**Diffusion-TS:** MIT. The README advertises Python 3.8 and a CUDA-capable GPU; requirements pin PyTorch 2.0.1 and older scientific packages. Training uses YAML configuration and checkpoint milestones; sampling emits NumPy arrays. Its transformer decoder separates trend and seasonality. [Official README](https://github.com/Y-debug-sys/Diffusion-TS), [requirements](https://github.com/Y-debug-sys/Diffusion-TS/blob/main/requirements.txt), [license](https://github.com/Y-debug-sys/Diffusion-TS/blob/main/LICENSE).

The future adapter must inspect tensor axes and device assumptions in the pinned dataset/model/solver implementation. Normalize the platform contract to `[sequence, timestep, variable]` only with an explicit tested mapping. Preserve inverse scaling, timestamps, entity IDs, and context separately. Arbitrary-length generation, categorical sequence generation, and CPU operation remain unverified. Begin with short, low-dimensional windows and one model at a time; measure rather than predict peak VRAM.

**TimeGAN:** Original requirements pin TensorFlow 1.15.0 and protobuf 3.20.3. Its loader returns sequences shaped as time by feature, applies min-max normalization, creates windows, then shuffles windows. A platform adapter must split first and fit scaling on training data only. [Official repository](https://github.com/jsyoon0823/TimeGAN), [requirements](https://github.com/jsyoon0823/TimeGAN/blob/master/requirements.txt), [loader](https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py).

The root license was not verified in this inspection (the attempted root LICENSE URL returned 404). Resolve reuse terms before vendoring or distributing it. Do not infer permission from public availability. Smallest viable technical route: assess a faithful maintained port or isolate a compatible legacy runtime; record and test any deviations. RTX 4060 GPU support is not established for the legacy stack.

## Integration acceptance criteria

For each candidate, pin its source revision; verify license and dependency resolution in isolation; test preprocessing/inverse transforms; train a tiny fixture; save/reload a checkpoint; sample requested shapes/counts; record parameters, device, timings, failures, and measured memory. Then run the same held-out evaluation protocol as the baselines. Mark it unavailable if any required step fails. Never substitute another model under its name.

No advanced-model downloads, environment installations, or training were performed during Milestone 1. Full model-code/API inspection belongs to the corresponding integration milestone.
