# Implementation status — Milestones 3–9

Updated 2026-11-23. Milestones 3–9 extend the Milestone 2 baseline with isolated advanced-model adapters, temporal data contracts, durable research experiments, resource-aware selection, dashboard visibility, and a local text-consistency baseline. See [MILESTONES.md](MILESTONES.md) for the execution journal and measured checks.

## Current verification

| Check | Result |
| --- | --- |
| `.venv/Scripts/python.exe -m pytest -q` | **101 passed**; only known deprecation/data-library warnings |
| `frontend/npm.cmd run build` | TypeScript and Vite build passed; existing large-bundle advisory remains |
| Advanced adapters | Actual tiny TabSyn, Diffusion-TS and `timegan_torch` fit/reload/sample checks passed |
| Unified temporal API | `/time-series/profile` and temporal `/pipeline-runs` profile, queue, poll, persist and export sequences |
| Temporal React workflow | Sample loading/upload, schema controls, validation, Diffusion-TS/TimeGAN launch, progress, metrics and CSV download build successfully |
| Model inventory | Fast dependency inventory is the default; the slower Torch/CUDA probe is explicit with `probe_hardware=true` |
| Fresh-machine provisioning | `scripts/setup_research.py` creates/checks the pinned isolated research environment and source revisions |
| Research tracking | Experiment store, candidate selection, temporal metrics, privacy and forecasting tests passed |
| Text baseline | Local template generation and rule-based cross-modal consistency tests passed |

## Implemented milestones

- **3:** isolated pinned TabSyn worker and reversible mixed-type codec; no TabDDPM compatibility claim.
- **4:** explicit temporal schema/windowing, leakage-safe chronological preprocessing, Diffusion-TS worker and optional Parquet ingestion.
- **5:** separately named PyTorch TimeGAN-style fixed-window baseline; no TensorFlow checkpoint-compatibility claim.
- **6:** durable SQLite experiments, retained failures/interruption recovery, temporal metrics, TSTR/forecasting/privacy/fairness applicability and resource metadata.
- **7:** bounded explicit/rule/adaptive candidate selection using validation-only scores and transparent resource/model inventory.
- **8:** research model/experiment API visibility plus a complete React temporal workflow: sample/upload, timestamp/entity/feature/context schema, sequence controls, Diffusion-TS/TimeGAN execution, progress, metrics and export through the unified pipeline route.
- **9:** deterministic local structured-to-text generation and auditable exact-rule consistency scoring; no hosted LLM required.

## Explicit boundaries

These integrations are research baselines, not claims of paper reproduction or production privacy. Advanced workers use tiny configurable runs and separately provisioned environments; full-size quality, multi-seed statistical conclusions, and locked deployment images remain future work. The text module is lexical rule matching, not semantic/embedding evaluation. Gemini remains opt-in and external.

---

# Historical Milestone 2 status

The detailed Milestone 2 report below is preserved for traceability.

## Milestone 2: implemented

- Added typed tabular lifecycle contracts (`fit`, `sample`, metadata, input validation, capabilities) and retained legacy `generate(...)` calls. Shared SDV lifecycle backs the existing Gaussian Copula/CTGAN/TVAE classes; bootstrap is an explicit baseline.
- Expanded registry metadata, validated adapter identity/lifecycle, removed four-name schema restrictions, and discovered eligible alternatives in the optimizer. Unknown/unavailable/disabled models fail explicitly.
- Removed runtime bootstrap substitution. Failures preserve the requested model's identity and produce a failed execution manifest, not a fake result.
- Added CPU/CUDA/auto configuration, explicit CUDA checks, supported-model option validation, CTGAN batch-size validation, pre-fit seeds with caller RNG restoration, and serialized agent model operations within one process.
- Labeled runs now plan/fit on training rows, rank candidates on separate validation rows, freeze candidate selection, and evaluate only the selected candidate on final-test rows. Split row positions are persisted. Rare-class split failures are explicit. Unlabeled similarity remains a training-distribution diagnostic.
- Persisted dataset hashes, requested configuration, settings excluding the API key, software/runtime information, terminal status and failures in atomic local execution manifests. Added `GET /runs/{run_id}/manifest`; failed background jobs include their run ID. Full tracking/queue recovery is still planned.
- Added per-iteration generator metadata, benchmark-role labels, `final_benchmark`, training-only `planning_analysis`, split manifests, and a quality-gate artifact eligibility flag. Skipped runs now have deterministic final reports.
- Bounded upload reads, rejected unsafe filenames, stored uploads/manual outputs under unique server names, scrubbed the example credential, and aligned the example minority ratio with the code.
- Made explanations opt-in in API/pipeline/React, disclosed transfer in React, separated validation-history/final-test displays, and marked rejected downloads as audit artifacts. React no longer automatically forces the full-table preview's generator into the training-only run planner.

## Milestone 2 verification

Milestone 2 is complete. Final checks executed on 2026-10-31:

| Check | Result |
| --- | --- |
| `.venv/Scripts/python.exe -m pytest -ra` | **77 passed, 15 warnings in 44.76 seconds** |
| React `npm.cmd run build` | TypeScript and Vite build passed; existing large-bundle warning remains |
| `python -m pip check` | No broken requirements found |
| Python compilation | Changed generator/runtime modules and Streamlit dashboard compile |
| New integration evidence | Real one-epoch CPU CTGAN/TVAE fits and samples; repeated Gaussian Copula sampling; new registry adapter through full pipeline; background API run, final-test role, disk fetch/download and manifest retrieval |
| Safety/regression evidence | No model substitution; validation-only ranking; train-only planner; disjoint persisted row splits; failed manifest without API key; explicit CUDA/config/input failures; filename isolation; bounded reads; explicit environment ratio override |

Tiny neural tests establish software integration, not research model quality. CUDA training, browser interaction, checkpoint reload, and advanced generators remain unverified. Warnings concern TestClient/httpx deprecation, date parsing, and SDV metadata-save recommendations. The inherited Gaussian Copula test no longer catches failure and substitutes bootstrap, so its passing status now verifies the real adapter.

## Milestone 2 changed files

| Area | Files |
| --- | --- |
| Contracts/adapters/runtime | `app/generators/base.py`, `sdv_generator.py`, `runtime.py`, `bootstrap_generator.py`, `copula_generator.py`, `ctgan_generator.py`, `tvae_generator.py`, `registry.py` |
| Agents and orchestration | `app/agents/generator.py`, `planner.py`, `optimizer.py`, `benchmark.py`; `app/pipeline/orchestrator.py`, `reporting.py` |
| API/config/schemas | `app/api/main.py`, `app/config.py`, `app/schemas/schemas.py`, `.env.example` |
| Dashboard | `frontend/src/App.tsx`, `pages.tsx`, `types/api.ts`; `frontend/dashboard.py` |
| Tests | `tests/test_generator_lifecycle.py`, `test_research_splits.py`, `test_api.py`, `test_generators.py` |
| Documentation | `README.md` and the five milestone documents |

## Remaining boundaries

No neural/diffusion checkpoint save/reload or experiment database was added. No CUDA package installation or GPU memory measurement was attempted. Model operations serialize inside one process; job threads/dictionaries still lack a durable queue or crash recovery. Unlabeled benchmarks are not downstream prediction tests. TSTR, temporal contracts/metrics, adaptive CV/Pareto search, full environment locking, and resource instrumentation remain later milestones. If the historical example credential was real, its owner must revoke it; the repository change cannot revoke it or erase history.

The next milestone is an isolated, pinned advanced-tabular integration, beginning with TabSyn feasibility and an actual tiny train/reload/sample test. It must stay unavailable until that test passes.

---

# Historical Milestone 1 baseline

The following audit describes the pre-Milestone-2 state and is preserved for traceability. Its gaps and plan table are historical; the current status is above.

# Implementation status and Milestone 1 report

Audit date: 2026-10-31. Baseline commit: `6c31cdf` (`final bluetokai sesh`). The working tree was clean before inspection. Milestone 1 adds documentation only; existing application code and dependency versions are unchanged.

## Verification actually executed

| Check | Result | Scope / limits |
| --- | --- | --- |
| `.venv/Scripts/python.exe -m pytest -ra` | **53 passed, 11 warnings in 33.51 seconds** | Existing suite; not proof of all adapters or GPU training |
| `.venv/Scripts/python.exe -m pip check` | No broken requirements found | Installed metadata compatibility only |
| `npm.cmd run build` in `frontend` | Passed TypeScript and Vite build | Warns about a 641.76 kB minified JS chunk; no browser/UI smoke test |
| Direct Gaussian Copula generation | Passed for labeled and unlabeled 80-row fixtures, 12 generated rows each | Called adapter directly with no bootstrap fallback |
| In-process API labeled E2E | `/plan`, `/run`, result fetch, CSV download passed | One iteration, seed 42, LLM off; 8 synthetic rows; no accepted improvement |
| In-process API unlabeled E2E | `/plan`, background `/pipeline-runs`, status polling, result fetch, CSV download passed | One iteration, seed 42, LLM off; 24 synthetic rows; existing unlabeled thresholds passed |
| Disk retrieval | Both result JSON and CSV retrieved after clearing in-memory `RUNS` | Does not verify process restart recovery for job state |
| Hardware preflight | Driver sees RTX 4060 Laptop GPU, 8188 MiB | Installed PyTorch is CPU-only; CUDA unavailable |

Audit smoke outputs are local ignored artifacts in `generated/milestone1_audit/`, including `smoke_summary.json`, input fixtures, per-run result JSON, and CSVs. They are software checks, not statistically meaningful research comparisons. API tests used FastAPI TestClient; no live browser/server deployment was tested. No external LLM request was made in these checks.

Warnings: Starlette/httpx TestClient deprecation; date-format inference; SDV metadata-save recommendations. The direct smoke also exposed SDV's `SingleTableMetadata` deprecation. An initial bare `npm` command was blocked by PowerShell script policy; `npm.cmd` succeeded.

## Implemented and exercised

- CSV analysis, explicit/auto target handling, labeled/unlabeled branches, adaptive sample planning and bounded optimization.
- Bootstrap and actual Gaussian Copula generation; schema/count checks and conditional generation path.
- Statistical fidelity, distribution details, dependency/MMD/discriminator diagnostics, constraints, diversity, and empirical privacy indicators.
- Original-versus-augmented downstream benchmarking, unlabeled KNN similarity, deterministic final reports.
- FastAPI plan/synchronous run/background run/status/result/download paths on small fixtures.
- React TypeScript compilation and production asset generation.

Some generator tests catch failures and use bootstrap, so a passing test named for Gaussian Copula alone would not prove its adapter worked. The separate direct-adapter smoke above removes that ambiguity for Gaussian Copula only.

## Implemented but not fully tested or connected

- CTGAN and TVAE adapters exist, but no real neural training/checkpoint/GPU validation was performed in this audit.
- React runtime, Streamlit interaction, Docker deployment, fairness behavior, and configured Gemini calls were inspected but not end-to-end tested here.
- Registry discovery exists, but schemas and fixed planner/optimizer lists restrict extension to four names.
- Candidate proposals/cache, model-suite CV, and Pareto selector have helper tests but are not wired into the main pipeline.
- Background jobs use daemon threads and process-local dictionaries; no durable job lifecycle, serialization of GPU work, cancellation, or crash recovery.
- A rejected candidate can remain downloadable. `improved=false` must not be interpreted as privacy approval or successful research validation.

## Missing, planned, or blocked

| Capability | Status | Smallest next step |
| --- | --- | --- |
| Generator fit/sample lifecycle and checkpoints | Planned | Extend contracts with backward-compatible wrappers; separate table/sequence inputs |
| TabSyn and TabDDPM | Registry placeholders only; environment compatibility untested | Isolated pinned worker, real preprocessing/training/checkpoint/sample test |
| Diffusion-TS | Planned; CUDA unavailable in current Python environment | Temporal schema/window tests, isolated CUDA worker and tiny model check |
| TimeGAN | Planned; legacy TensorFlow environment and reuse terms unresolved | Resolve license and select/test isolated implementation or faithful port |
| Timestamp/entity-aware CSV/Parquet ingestion | Planned | Explicit schema, regularity detection, split-before-window preprocessing |
| Temporal metrics/forecasting utility | Planned | Signal-based fixtures, boundary-safe metrics, fixed forecasting baseline |
| TSTR and untouched final-test protocol | Planned | Training-only transforms, validation selection, final test once |
| Experiment tracking/multi-seed/ablations | Planned | Versioned manifests and durable local run records, then optional MLflow |
| Efficiency/memory instrumentation | Planned | Separate fit/sample timers, actual hardware, optional memory probes |
| Membership inference and calibrated privacy comparisons | Planned | Threat model and train/holdout control experiments |
| Cross-run dashboard, modality/model configuration, report/config export | Planned | Extend existing React/API after backend contracts stabilize |
| Text and cross-modal consistency | Deferred | Investigate after reproducible table/sequence support |

## Architectural limitations to resolve first

1. **Research leakage:** generators fit on training rows, but profiling/planning see the full dataset and repeated candidate ranking uses original test scores. A shared test set does not make model selection leakage-safe. The existing CV helper is not used by orchestration.
2. **Model attribution:** adapter failures inside generation may return bootstrap while plans, filenames, provenance, and reports retain the requested generator. Record requested/actual model and failure; advanced models must fail explicitly.
3. **Reproducibility:** SDV's random seed is set after fitting; no guarantee of deterministic fitting, stored split IDs, preprocessing artifacts, complete config/environment, source revisions, or trained checkpoints.
4. **Partial registry architecture:** hardcoded generator literals/cycles and missing modality/device/schema contracts prevent plug-in extension. TVAE conditional metadata and invocation behavior disagree.
5. **Selection/reporting:** extra diagnostics are computed but not used by the active selector; unavailable PCA/correlation checks can yield perfect-looking neutral values. Generation-skipped results return before final-report construction. Rejected candidates remain accessible for audit without a dedicated artifact eligibility contract.
6. **Jobs and failure records:** no bounded worker queue, durable job state, process-safe shared storage, or durable exception manifest. Concurrent threads could compete for GPU memory once CUDA is enabled.
7. **Input storage:** API upload paths use the client filename directly and can overwrite same-name files; validate basenames/path containment and assign dataset IDs. Byte limits are checked after reading the full upload. Scope is a local prototype, not hardened hosted ingestion.
8. **Configuration/dependencies:** Python dependencies have open lower bounds and frontend uses several `latest` declarations (with an npm lockfile). Documentation and code disagree on defaults. SDV private seed compatibility hooks and deprecated metadata increase upgrade risk.
9. **External data/configuration:** React enables explanations by default; configured Gemini can receive sensitive summaries. The tracked example environment includes a credential-like value and Compose consumes it. Scrub the template and rotate/revoke that value if valid; no value is reproduced in these documents.

## Prioritized implementation plan

| Milestone | Concrete scope | Exit evidence |
| --- | --- | --- |
| 1 — audit | Inspect current paths, tests/hardware, document actual architecture and gaps | Complete: this report and four companion documents; baseline suite/build/smokes executed |
| 2 — contracts and baseline integrity | Preserve public routes/adapters; add fit/sample/metadata/capabilities and model config; remove false attribution; validate registry names; train/validation/test discipline, seed/device checks; address exposed template value and safe ingestion paths | Existing regressions pass; failure attribution, unavailable model, train-only preprocessing and extension tests; unchanged existing workflows |
| 3 — advanced tabular | Prioritize TabSyn feasibility in an isolated environment; inspect pinned implementation/license; typed preprocessing, checkpoints, real sampling; TabDDPM next if viable | Tiny real training/reload/sample/inverse-transform integration passes; measured timings/memory or explicit blocker |
| 4 — temporal subsystem | Configurable timestamp/features/entity schema; irregularity policy; CSV then optional Parquet; split-before-window; Diffusion-TS adapter | Boundary/leakage/shape/round-trip tests and tiny real training/checkpoint/sample check |
| 5 — TimeGAN | Resolve reuse terms, isolate legacy implementation or justify faithful port | Same sequence representation and evaluation fixtures; actual training/sample validation |
| 6 — evaluation and tracking | Temporal metrics, TSTR/forecasting tasks, privacy/fairness applicability, local experiment store, optional MLflow | Known-signal checks, identical split comparisons, durable failures/config/artifact manifests |
| 7 — adaptive search and efficiency | Connect candidate/CV/selection helpers correctly; resource-aware sequential search, multi-seed matrices and ablations | Reproducible equal-budget comparisons; real timing/memory data; final test isolated |
| 8 — dashboard | Preserve React; add modality/schema/parameters, meaningful job progress, comparison charts, temporal plots and exports | Browser E2E on both modalities; failed/unavailable runs clearly shown |
| 9 — text investigation | Optional pretrained text interface and rule/embedding consistency checks | Small locally testable experiment; explicit disclosure/configuration for external services |

Tracking/config foundations needed to test earlier adapters should be introduced early; full experiment comparison remains Milestone 6. Do not hold working baseline components hostage to all advanced dependencies. Keep each integration optional and report its true validation status.

## Changed files

- [ARCHITECTURE.md](ARCHITECTURE.md): actual components, execution/data flow, metrics, jobs/storage, limitations.
- [MODEL_COMPARISON.md](MODEL_COMPARISON.md): current/proposed adapters, official source findings, licensing/dependency/hardware constraints.
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): questions, datasets, controls, leakage-safe evaluation, ablations/statistics and tracking contract.
- [SETUP.md](SETUP.md): verified local commands, observed environment, CUDA finding, optional environment strategy.
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md): this report, measured verification, gaps and milestone order.
