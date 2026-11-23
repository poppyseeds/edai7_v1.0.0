# Milestones 3–9 execution journal

This journal records actual integration checks. Tiny fixtures validate software paths; they do not establish research performance or RTX 4060 memory requirements.

## Milestone 3 — advanced tabular adapter

- Added a pinned TabSyn source provisioner and a separate-process adapter using the actual upstream VAE, EDM model/loss, and sampler. No full upstream requirements were installed in the application environment.
- Added training-only quantile/categorical encoding, numerical missingness indicators, reversible schema/order/dtype handling, explicit configuration, JSON preprocessing state and tensor-only model checkpoints.
- A fresh sampling process reloads checkpoint weights with `weights_only=True`. Numerical-only/unlabeled tables require no invented target. Conditional generation is explicitly unsupported by this adapter.
- Actual verification: `pytest tests/test_tabsyn.py -ra` **2 passed in 26.52s**, including real tiny VAE/diffusion training, checkpoint reload and mixed-type sampling. Baseline regression results follow in subsequent journal entries.
- Pin: `amazon-science/tabsyn@cb5ac0f74ec36ee88e7a974a393dfbef50d42da7`, Apache-2.0. The local codec, fixed epochs, smaller configurable dimensions, and absent upstream scheduler are disclosed adaptations, not paper reproduction. Python 3.12/current CPU PyTorch passed the fixture despite the upstream Python 3.10 recommendation.
- TabDDPM remains unavailable: its official legacy dependency set requires a separate compatibility effort (Python 3.9.7/PyTorch 1.10.1 and scientific-library pins). No compatible adapter or successful run is claimed.
- Changed: generator config/registry, `tabsyn_generator.py`, `app/research/{sources,codec}.py`, source pins, source setup/worker scripts, optional integration tests, `.gitignore`.
- Baseline regression: **79 passed, 15 warnings in 69.89s** after the TabSyn integration.

## Milestone 4 — sequence subsystem and Diffusion-TS

- Added an explicit timestamp/features/entity/context schema, chronological splitting before windowing, training-only scaling/imputation, regularity diagnostics, fixed-length multivariate tensors and synthetic sequence IDs/timestamps. Irregular data is rejected with no implicit resampling; heterogeneous entity intervals require separate runs. Context is retained as metadata and is not falsely claimed as generated/conditioned.
- Added optional CSV/Parquet ingestion and a sequence generator contract separate from tabular rows. The actual upstream Diffusion_TS network, Fourier loss and sampler run in a separate worker, with checkpoint storage and reload. Fixed-epoch Adam without upstream EMA/scheduler is an explicitly recorded adaptation.
- Pin: `Y-debug-sys/Diffusion-TS@566307e6cf2d8095e58de4c6e3a6ae965b69b5b5`, MIT. Only `einops==0.6.0` was added to a separate research environment. Application dependencies were not upgraded.
- Verification: **5 passed in 18.74s** in `tests/test_timeseries.py`, including actual CPU training/reload/sampling. An additional RTX 4060 smoke check is recorded separately after execution.
- The optional worker environment inherits the machine's separate Python installation (PyTorch 2.5.1+cu124); the application environment still uses CPU-only PyTorch 2.14.0. This is process/dependency separation, not a fully locked container.
- Changed: `app/timeseries/`, research ingestion/source pins, worker extension, research requirements, sequence preprocessing/integration tests.
- RTX 4060 smoke: actual CUDA train/reload/sample passed for an explicitly tiny 8-step/two-variable fixture. Measured fit wall time **6.167 s**, peak PyTorch GPU allocation **18,301,952 bytes**, checkpoint **37,474 bytes**. This is not total VRAM and does not predict full-size training memory. Local evidence: `generated/milestone4_gpu_smoke.json`.

## Milestone 5 — TimeGAN comparative baseline

- Inspected `jsyoon0823/TimeGAN@8f6181cb9b9d2fa0c930cd902411d9ac8a308e07`, including embedding/supervised/joint training and adversarial/moment objectives. Its lowercase `license` file declares Apache-2.0; this resolves the earlier audit's unverified uppercase LICENSE lookup.
- Original TensorFlow 1.15 requirements remain incompatible with the application runtime. Added a separately named `timegan_torch` fixed-window PyTorch GRU reimplementation, with all three phases, checkpoint reload, minibatch inference, and explicit differences in initialization/optimizer/window handling. No TensorFlow checkpoint compatibility or paper reproduction is claimed.
- Uses exactly the same prepared sequence tensor and inverse/export metadata as Diffusion-TS. Original upstream code is not substituted silently.
- Actual tiny three-phase CPU train/reload/sample test outcome is recorded after execution; no model-quality claim follows from a one-epoch run.
- Changed: `app/timeseries/timegan.py`, sequence adapter subclass, worker dispatch/training/sampling, real integration test.
- Verification: actual tiny CPU three-phase fit, checkpoint reload and sampling **1 passed in 15.46s**. The reference implementation remains optional/uninstalled; the tested baseline is named `timegan_torch`.

## Milestone 6 — evaluation and experiment tracking

- Added a durable SQLite experiment store with queued/running/completed/failed records, incremental stage events, configuration/dataset identity, explicit interruption recovery, exportable reports and retained failures. A single research executor serializes training jobs.
- Added labeled tabular TRTR/TSTR/augmentation utility on identical real test rows, applicable group metrics, and separate sequence evaluation: marginals, mean/variance, ACF, cross-variable/lag dependence, trend, spectra and duplicate sequences. Undefined metrics remain null.
- Added an explicit forecasting target/horizon task using chronological held-out windows and a fixed Ridge baseline, with real/synthetic/augmented training comparisons in original target units.
- Added train/holdout nearest-distance controls and an explicitly empirical distance membership attack. Exact copy detection rejects the sequence-resampling baseline. These measurements do not establish differential privacy or certify anonymity.
- Recorded real fit/sample wall times, throughput, checkpoint bytes, worker environment/device and measured GPU allocations. CPU peak memory remains null rather than invented. Full MLflow installation is deferred behind the lightweight store.
- Changed: research config/store/engine/evaluation/ingestion, sequence-resampling baseline, worker provenance, dedicated metric/storage/experiment tests. Verification outcome follows below after execution.
- Verification: **94 passed, 15 warnings in 81.73s** for the complete Python suite after Milestones 6–9 changes.

## Milestone 7 — adaptive selection and resource-aware planning

- Added transparent model inventory/planning (`app/research/planner.py`) with modality, source availability, worker Python/device visibility and explicit experimental opt-in.
- Added bounded candidate selection (`app/research/selection.py`) supporting explicit, rule-based and adaptive modes. Candidates are compared on validation-only utility/quality/cost; failed candidates remain in the report and final held-out tests run only after selection.
- Added temporal quality gates and cost metadata. No unsupported claim of optimality is made; adaptive selection is a bounded Pareto/rule procedure.

## Milestone 8 — dashboard and experiment visibility

- Added `/research/models` and `/experiments` API read endpoints so clients can display model availability and durable experiment state without scraping files.
- Added `/time-series/profile` for actual schema/window validation and extended the normal `/pipeline-runs` contract to dispatch both tabular and temporal jobs. Temporal jobs are polled through the same route and remain durable in the SQLite experiment store.
- Added a React **Time Series** workflow that loads the bundled sensor sample or an uploaded CSV; configures timestamp, entity, numerical features, context, sequence length, stride and missingness policy; launches Diffusion-TS, `timegan_torch` or the diagnostic window baseline; shows progress and measured results; and downloads generated sequences.
- Made `/research/models` fast by default using dependency/source checks. The slower Torch/CUDA import probe now runs only with `probe_hardware=true`.
- Expanded `scripts/setup_research.py` so a fresh machine can create the isolated environment, install pinned research requirements, clone/verify pinned model sources and run readiness checks.
- Existing React dashboard build passed with TypeScript checking (`npm.cmd run build`). The bundle-size advisory remains non-blocking and is documented as a follow-up optimization.
- Dashboard copy distinguishes validation-time optimization from final-test benchmarking, marks audit-only artifacts, and keeps Gemini explanations explicitly opt-in.
- Final verification on 2026-11-23: **101 Python tests passed**, the production React build passed, and direct unified-API smoke runs completed for both Diffusion-TS and `timegan_torch`.

## Milestone 9 — text generation and cross-modal consistency

- Added `TemplateTextGenerator`, a deterministic local structured-to-text baseline with explicit schema fields and no external credentials.
- Added `evaluate_consistency` for rule-based fact coverage/violations with per-record evidence and a declared exact-match method. It is intentionally not marketed as semantic understanding or an embedding-based score.
- Verification: local generation and missing-fact detection are covered by `tests/test_text.py`.
