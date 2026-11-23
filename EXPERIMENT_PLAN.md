# Reproducible experiment plan

This is a proposed protocol, not a report of research results. Milestone 1 smoke checks are software verification only.

## Questions and comparisons

| Question | Controlled comparison |
| --- | --- |
| Do diffusion models improve tabular synthesis? | Bootstrap, Gaussian Copula, CTGAN, TVAE versus tested TabSyn and TabDDPM adapters |
| Does adaptive selection improve on current rules? | Existing fixed planner/cycle versus validation-driven candidate selection with identical budgets |
| How do sequence models compare? | TimeGAN versus Diffusion-TS on identical entity/window splits; add a simple window-resampling baseline |
| What are the tradeoffs? | Fidelity/utility versus training time, sampling throughput, measured memory, and empirical privacy risk |

## Datasets

Start with local generated mixed/numerical tabular fixtures and multivariate sine/trend fixtures for correctness. Existing customer samples are demonstrations, not sufficient evidence for publication.

Candidate research datasets: Adult and a credit/default dataset for mixed tabular classification; a suitably licensed numerical regression dataset; ETTh1 and Appliances Energy for multivariate sequences. Before use, record source/version/license, dataset hash, target, protected attributes if valid, units, timestamp meaning, and entity structure. These datasets are candidates pending acquisition and license review; do not download them automatically. Add multiple entities and irregular-sampling fixtures regardless of real dataset choice.

## Splits and preprocessing

1. Persist train/validation/test row or entity identifiers once per dataset/split seed. A starting tabular split is 60/20/20 with stratification when feasible. Validate rare classes explicitly instead of silently changing the protocol.
2. Fit all imputers, scalers, vocabularies, generator metadata, and inferred constraints on training data only. Exclude direct identifiers or encode their treatment in the schema. No fake targets for unlabeled tables.
3. Select hyperparameters and generators on real validation data or nested training-only CV. Keep the final real test set inaccessible until selection finishes. Milestone 2 implements validation-only selection and a final-test comparison after selection. TSTR, nested CV and full experiment matrices remain planned.
4. For temporal tasks, split raw observations chronologically within the declared entity policy before windowing. Prevent windows from crossing entity/split boundaries. Use a boundary gap where forecast horizons or overlapping windows require it. Hold out entities separately when testing generalization to unseen entities.
5. Record timestamp column/timezone, feature order, sequence/entity IDs, interval tolerance, sequence length, stride, forecast horizon, and missingness policy. Detect irregular sampling; reject it or apply an explicitly configured policy. Never silently interpolate or resample.
6. Keep categorical context outside numerical sequence tensors unless an adapter explicitly supports it. Persist inverse transformations and generated-sequence indexing metadata.

## Metrics and interpretation

| Dimension | Planned measurements | Interpretation limits |
| --- | --- | --- |
| Tabular fidelity | Per-feature KS/Wasserstein or TV/JS, dependency differences, MMD, discriminator AUC | Type-appropriate, sample-size-dependent diagnostics; no single score establishes fidelity |
| Sequence fidelity | Per-variable marginals, mean/variance errors, ACF over predefined lags, cross-correlation, trend/spectral summaries, duplicate-window rate | Respect sequence boundaries; lag/horizon dependent; spectra need adequate regular samples; do not compute global statistics across unrelated entities |
| Supervised utility | TRTR, TSTR, and real-plus-synthetic training evaluated on the same untouched real test set | TSTR is not implemented yet; synthetic labels must be generated as part of genuine labeled records |
| Forecasting utility | Real-only, synthetic-only, and augmented forecasting with a fixed model/horizon; MAE/RMSE and suitable scaled errors | Chronological evaluation, training-only normalization; no claims of utility for unlabeled sequences without a defined task |
| Privacy risk | Exact/near duplicate rates, train-versus-holdout nearest-neighbor distances, optional attack-based membership inference | Empirical risk; calibrate distance and attack baselines; no differential privacy guarantee |
| Fairness | Group sizes, task metrics and gaps, uncertainty where valid protected attributes/targets exist | Report unavailable when prerequisites fail; small groups may be too uncertain for conclusions |
| Efficiency | Separate fit/sample wall time, throughput, checkpoint bytes, hardware/software, peak GPU allocation and process memory where available | Distinguish allocator measurements from total VRAM; unavailable measurements stay null; synchronize CUDA for timing |

Measure fidelity against both real training and held-out distributions where meaningful. Avoid interpreting deliberately minority-conditioned samples as a failure to reproduce the full target distribution; report conditional and full-distribution experiments separately. Class-balancing generation and unconditional synthesis answer different questions.

Existing five-part composite scores and thresholds remain prototype defaults. Report individual raw metrics and not-applicable reasons; do not treat missing metrics as perfect evidence or tune weights using test outcomes.

## Controls and run tracking

Use at least five prespecified generator seeds when feasible, paired across methods. Separate split, training, sampling, and downstream-model seeds. Compare fixed sample/sequence counts and task definitions; report both comparable resource budgets and model-specific training schedules. Small smoke schedules must not stand in for converged model comparisons.

Each run needs a versioned configuration containing dataset fingerprint, split manifest, preprocessing, modality/schema, source/model version, resolved parameters, all seeds, hardware, environment versions, timestamps, fit/sample durations, memory measurements, metrics, artifact paths/checksums, and structured failure details. Save configuration before fitting and terminal state after success or failure. A lightweight JSON manifest plus SQLite/local tracking backend is the first target; add MLflow through an optional backend only after that contract is tested.

Run experiments sequentially on the 8 GB GPU. Record OOM rather than silently lowering batch size or changing precision. Explicitly requested alternative configurations are separate runs. A crash/interruption must remain visible as incomplete/failed after restart; do not infer completion from an existing CSV.

## Ablations and statistics

Prespecify ablations for generator-only versus sample-count search, hard-gate contributions, training-only CV versus single validation ranking, conditional versus unconditional synthesis, sequence length/stride, and diffusion sampling steps. Keep data splits and evaluation settings fixed. Run invalid/privacy-gate ablations in evaluation-only research mode and retain their rejection labels.

Report per-seed values, mean/dispersion, confidence intervals, failures and completion counts, plus effect sizes. Use paired comparisons on matched dataset/seed outcomes; select a justified paired test after checking assumptions and account for multiple comparisons, for example with Holm correction. Do not treat overlapping temporal windows as independent statistical replicates. Never claim significance from a mean difference alone; explain handling of failed runs rather than dropping them silently.

## Execution order and exit criteria

1. Repair model attribution, seed handling, and held-out selection; freeze baseline environments.
2. Establish modality-specific contracts, persisted configuration, and accurate failure records.
3. Validate one advanced tabular adapter with checkpoint reload, then baseline comparisons.
4. Validate temporal schema/windowing and evaluators on known signals before Diffusion-TS.
5. Integrate a license-compatible TimeGAN baseline and run paired temporal comparisons.
6. Add experiment matrices, adaptive selection, efficiency analysis, and comparison UI.
7. Investigate text/cross-modal consistency after the first two modalities have reproducible results; require explicit external-service configuration.

Each milestone runs relevant tests, fixes regressions, records actual results, and updates implementation status. Publication readiness is an experimental outcome, not an implementation label.
