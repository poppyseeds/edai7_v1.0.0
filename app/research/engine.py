"""Sequential local research experiments; final-test data is never used for selection."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import platform
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.agents.analyzer import DatasetAnalyzer
from app.agents.benchmark import BenchmarkAgent
from app.agents.validator import ValidationAgent
from app.generators.registry import create_generator
from app.models.benchmark_model import DownstreamModel
from app.evaluation.fairness import group_performance
from app.research.config import ExperimentConfig
from app.research.evaluation import sequence_metrics, distance_privacy, forecasting_utility
from app.research.store import ExperimentStore, now
from app.research.selection import select_experiment_candidate
from app.research.planner import plan_experiment
from app.timeseries.data import prepare_sequences, sequences_to_frame
from app.timeseries.generators import SEQUENCE_GENERATORS

EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-experiment")


def _write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8")


def submit_experiment(frame, config: ExperimentConfig, store: ExperimentStore, dataset_identifier="upload"):
    identifier = store.create(config.model_dump(), sha256(frame.to_csv(index=False).encode()).hexdigest(), dataset_identifier)
    frame.to_csv(store.directory(identifier) / "input.csv", index=False)
    EXECUTOR.submit(execute_experiment, identifier, store)
    return store.get(identifier)


def execute_experiment(identifier, store: ExperimentStore):
    directory = store.directory(identifier)
    record = store.get(identifier)
    config = ExperimentConfig(**record["config"])
    started = time.perf_counter()
    store.update(identifier, status="running", started_at=now())
    try:
        frame = pd.read_csv(directory / "input.csv")
        store.event(identifier, "preprocessing", "Preparing disjoint real data splits and training-only transformations.")
        if config.modality == "tabular":
            result = tabular_experiment(frame, config, directory, store, identifier)
        else:
            result = temporal_experiment(frame, config, directory, store, identifier)
        result["efficiency"]["experiment_wall_seconds"] = time.perf_counter()-started
        result["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch")}}
        store.update(identifier, status="completed", stage="completed", metrics=result,
                     wall_seconds=time.perf_counter()-started)
        _write(directory / "report.json", store.get(identifier))
        (directory / "report.md").write_text(
            f"# Synthetic-data experiment {identifier}\n\nExecution completed for {result['generator']}.\n\n"
            f"Quality gates passed: {result['artifact_eligible']}. This is not a privacy guarantee.\n\n"
            "Metrics and full limitations are provided in report.json.\n", encoding="utf-8")
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        store.update(identifier, status="failed", stage="failed", failure=failure, wall_seconds=time.perf_counter()-started)
        _write(directory / "report.json", store.get(identifier))
    return store.get(identifier)


def _split_table(frame, config):
    if config.target:
        if config.target not in frame:
            raise ValueError("Requested target column is missing.")
        task = DatasetAnalyzer()._infer_task_type(frame[config.target])
        splits = BenchmarkAgent().research_split(frame, config.target, task, 0.2, 0.25, config.split_seed)
    else:
        development, test = train_test_split(np.arange(len(frame)), test_size=0.2, random_state=config.split_seed)
        train, valid = train_test_split(development, test_size=0.25, random_state=config.split_seed+1)
        splits = {"train": train.tolist(), "validation": valid.tolist(), "test": test.tolist()}
        task = None
    return {key: frame.iloc[rows].reset_index(drop=True) for key, rows in splits.items()}, splits, task


def _efficiency(metadata, fit_time, sample_time, count):
    return {"fit_wall_seconds": fit_time, "sampling_wall_seconds": sample_time,
        "samples_per_second_including_worker_startup": count/sample_time if sample_time > 0 else None,
        "model_size_bytes": metadata.get("model_size_bytes"),
        "peak_gpu_allocated_bytes": metadata.get("peak_gpu_allocated_bytes"),
        "peak_cpu_memory_bytes": None,
        "limitations": "Wall measurements include worker startup/serialization where used. GPU allocation is not total VRAM. CPU peak is unavailable unless instrumented; missing values are null."}


def tabular_experiment(frame, config, directory, store, identifier):
    frames, splits, task = _split_table(frame, config)
    _write(directory / "splits.json", splits)
    train, valid, test = (frames[key] for key in ("train", "validation", "test"))
    if len(train) < 10:
        raise ValueError("At least ten training rows are required.")
    def evaluate(name):
        model = create_generator(name, include_experimental=config.allow_experimental)
        cfg = config.generator_config.model_copy(update={"random_state": config.seed})
        store.event(identifier, "fitting", f"Fitting {name} on real training rows only.")
        start = time.perf_counter(); model.fit(train, cfg); fit_time = time.perf_counter()-start
        store.event(identifier, "sampling", f"Sampling {name} for validation-only comparison.")
        start = time.perf_counter(); generated = model.sample(config.num_samples); sample_time = time.perf_counter()-start
        quality = ValidationAgent().validate(train, generated, random_state=config.seed)
        from app.pipeline.orchestrator import _hard_gate_failures
        gates = _hard_gate_failures(quality)
        score = quality.overall_score
        if task:
            metrics = BenchmarkAgent().evaluate_split(pd.concat([train, generated], ignore_index=True),
                valid, config.target, task, config.seed)
            score = metrics.primary_value
        return {"name": name, "synthetic": generated, "metadata": model.get_metadata(),
                "fit_time": fit_time, "sample_time": sample_time, "score": score, "failed_gates": gates}
    names = _candidate_names(frame, config)
    candidate, selection = select_experiment_candidate(config, names, evaluate)
    synthetic, metadata = candidate["synthetic"], candidate["metadata"]
    fit_time, sample_time = candidate["fit_time"], candidate["sample_time"]
    synthetic.to_csv(directory / "synthetic.csv", index=False)
    store.event(identifier, "evaluating", "Computing tabular metrics and final held-out utility.")
    validation = ValidationAgent().validate(train, synthetic, random_state=config.seed)
    privacy = distance_privacy(train, test, synthetic, config.seed)
    utility, fairness = None, {"status": "unavailable", "reason": "Needs a protected attribute and classification target."}
    if task:
        agent = BenchmarkAgent()
        trtr = agent.evaluate_split(train, test, config.target, task, config.seed)
        tstr = agent.evaluate_split(synthetic, test, config.target, task, config.seed)
        augmented = agent.evaluate_split(pd.concat([train, synthetic], ignore_index=True), test, config.target, task, config.seed)
        utility = {"real_only": trtr.model_dump(), "synthetic_only": tstr.model_dump(),
                   "augmented": augmented.model_dump(), "evaluation_role": "final_test", "task": task}
        if config.sensitive_column and config.sensitive_column in test and task == "classification":
            downstream = DownstreamModel(task, random_state=config.seed)
            downstream.fit(synthetic.drop(columns=config.target), synthetic[config.target].astype(str))
            predicted = downstream.predict(test.drop(columns=config.target))
            fairness = group_performance(test[config.target].astype(str), pd.Series(predicted),
                                         test[config.sensitive_column], task).model_dump()
    from app.pipeline.orchestrator import _hard_gate_failures
    eligible = not _hard_gate_failures(validation) and privacy["exact_training_match_rate"] <= 0.05
    _write(directory / "generator.json", metadata)
    return {"modality": "tabular", "generator": candidate["name"], "selection": selection, "artifact_eligible": eligible,
        "quality": validation.model_dump(), "privacy": privacy, "utility": utility,
        "fairness": fairness, "generator_metadata": metadata,
        "efficiency": _efficiency(metadata, fit_time, sample_time, len(synthetic)),
        "preview": json.loads(synthetic.head(20).to_json(orient="records")),
        "split_counts": {key: len(rows) for key, rows in splits.items()}}


def temporal_experiment(frame, config, directory, store, identifier):
    prepared = prepare_sequences(frame, config.sequence_schema)
    _write(directory / "preprocessing.json", prepared.metadata)
    _write(directory / "splits.json", prepared.provenance)
    def evaluate(name):
        factory = SEQUENCE_GENERATORS.get(name)
        if factory is None:
            raise ValueError(f"Unknown time-series generator {name}.")
        model = factory()
        cfg = config.sequence_config.model_copy(update={"random_state": config.seed})
        store.event(identifier, "fitting", f"Fitting {name} on chronological training windows.")
        start = time.perf_counter(); model.fit(prepared.windows["train"], cfg, prepared.metadata); fit_time = time.perf_counter()-start
        store.event(identifier, "sampling", f"Sampling {name} for validation-only comparison.")
        start = time.perf_counter(); generated = model.sample(config.num_samples); sample_time = time.perf_counter()-start
        quality = sequence_metrics(prepared.windows["validation"], generated)
        privacy = distance_privacy(prepared.windows["train"], prepared.windows["validation"], generated, config.seed)
        gates = _temporal_gates(quality, privacy, config)
        errors = [item["ks_statistic"] for item in quality["per_feature"]]
        errors += [item["acf_error"] for item in quality["per_feature"] if item["acf_error"] is not None]
        score = 1/(1+float(np.mean(errors)))
        if config.forecast_target:
            index = config.sequence_schema.features.index(config.forecast_target)
            utility = forecasting_utility(prepared.windows["train"], prepared.windows["validation"], generated,
                index, config.forecast_horizon, prepared.metadata["scaler"]["span"][index])
            score = 1/(1+utility["metrics"]["synthetic_only"]["mae"])
        return {"name": name, "synthetic": generated, "metadata": model.get_metadata(), "fit_time": fit_time,
                "sample_time": sample_time, "score": score, "failed_gates": gates}
    candidate, selection = select_experiment_candidate(config, _candidate_names(frame, config), evaluate)
    generated, metadata = candidate["synthetic"], candidate["metadata"]
    fit_time, sample_time = candidate["fit_time"], candidate["sample_time"]
    np.save(directory / "synthetic.npy", generated)
    sequences_to_frame(generated, prepared.metadata).to_csv(directory / "synthetic.csv", index=False)
    store.event(identifier, "evaluating", "Evaluating temporal dependence, privacy controls and requested forecasting task.")
    quality = sequence_metrics(prepared.windows["test"], generated)
    privacy = distance_privacy(prepared.windows["train"], prepared.windows["test"], generated, config.seed)
    utility = None
    if config.forecast_target:
        target = config.sequence_schema.features.index(config.forecast_target)
        utility = forecasting_utility(prepared.windows["train"], prepared.windows["test"], generated,
            target, config.forecast_horizon, prepared.metadata["scaler"]["span"][target])
    _write(directory / "generator.json", metadata)
    eligible = not _temporal_gates(quality, privacy, config) and not candidate["failed_gates"]
    real = sequences_to_frame(prepared.windows["test"][:1], prepared.metadata)
    synthetic = sequences_to_frame(generated[:1], prepared.metadata)
    return {"modality": "time_series", "generator": candidate["name"], "selection": selection, "artifact_eligible": eligible,
        "quality": quality, "privacy": privacy, "utility": utility,
        "fairness": {"status": "unavailable", "reason": "No sequence classification labels/protected-group prediction task configured."},
        "generator_metadata": metadata, "preprocessing": prepared.metadata,
        "split_counts": {key: len(values) for key, values in prepared.windows.items()},
        "efficiency": _efficiency(metadata, fit_time, sample_time, len(generated)),
        "real_sequence_preview": json.loads(real.to_json(orient="records", date_format="iso")),
        "synthetic_sequence_preview": json.loads(synthetic.to_json(orient="records", date_format="iso"))}


def _candidate_names(frame, config):
    if config.selection == "adaptive":
        return config.candidates
    if config.selection == "rule":
        recommendation = plan_experiment(frame, config)
        if recommendation["recommended"] is None:
            raise ValueError(recommendation["reason"])
        return [recommendation["recommended"]]
    return [config.generator]


def _temporal_gates(quality, privacy, config):
    failures = []
    if privacy["exact_training_match_rate"] > 0.05:
        failures.append("exact_training_match_rate")
    if privacy["near_training_match_rate"] > 0.1:
        failures.append("near_training_match_rate")
    if np.mean([item["ks_statistic"] for item in quality["per_feature"]]) > config.temporal_max_ks:
        failures.append("marginal_distribution")
    acf = [item["acf_error"] for item in quality["per_feature"] if item["acf_error"] is not None]
    if not acf or np.mean(acf) > config.temporal_max_acf_error:
        failures.append("acf_unavailable_or_above_threshold")
    return failures
