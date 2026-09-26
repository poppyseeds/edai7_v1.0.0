"""Autonomous analyze -> plan -> generate -> validate -> benchmark -> retry loop."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.agents.analyzer import DatasetAnalyzer
from app.agents.benchmark import BenchmarkAgent
from app.agents.generator import GeneratorAgent
from app.agents.llm_reasoner import LLMReasoner
from app.agents.optimizer import OptimizationAgent
from app.agents.planner import GenerationPlanner
from app.agents.validator import ValidationAgent
from app.config import get_settings
from app.evaluation.fairness import group_performance
from app.evaluation.unsupervised_utility import evaluate_unsupervised_utility
from app.evaluation.watermark import provenance_fingerprint
from app.models.benchmark_model import DownstreamModel
from app.schemas.schemas import (
    AgentLog,
    GenerationPlan,
    IterationRecord,
    PipelineResult,
    utc_now,
)
from app.utils.exceptions import PipelineError
from app.utils.logging_config import get_logger, setup_logging

logger = get_logger("Pipeline")


@dataclass
class PipelineConfig:
    max_iterations: int | None = None
    random_state: int | None = None
    synthetic_ratio: float | None = None
    ctgan_epochs: int | None = None
    tvae_epochs: int | None = None
    min_improvement: float | None = None
    min_absolute_improvement: float | None = None
    test_size: float | None = None
    enable_llm: bool = True
    auto_detect_target: bool = False
    preferred_generator: str | None = None
    sensitive_column: str | None = None
    output_dir: str | Path | None = None
    dataset_filename: str | None = None


@dataclass
class PipelineOutput:
    result: PipelineResult
    synthetic_df: pd.DataFrame | None
    original_df: pd.DataFrame
    train_df: pd.DataFrame | None
    test_df: pd.DataFrame | None


def run_pipeline(
    df: pd.DataFrame,
    target_column: str | None = None,
    config: PipelineConfig | None = None,
) -> PipelineOutput:
    setup_logging(get_settings().log_level)
    cfg = config or PipelineConfig()
    settings = get_settings()
    max_iterations = cfg.max_iterations or settings.max_iterations
    random_state = cfg.random_state if cfg.random_state is not None else settings.random_state
    synthetic_ratio = (
        cfg.synthetic_ratio if cfg.synthetic_ratio is not None else settings.default_synthetic_ratio
    )
    ctgan_epochs = cfg.ctgan_epochs if cfg.ctgan_epochs is not None else settings.ctgan_epochs
    tvae_epochs = cfg.tvae_epochs if cfg.tvae_epochs is not None else settings.tvae_epochs
    output_dir = Path(cfg.output_dir or settings.generated_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:12]
    logs: list[AgentLog] = []

    def log(agent: str, message: str) -> None:
        logs.append(AgentLog(agent=agent, message=message, timestamp=utc_now()))
        get_logger(agent).info(message)

    analyzer = DatasetAnalyzer()
    planner = GenerationPlanner()
    generator_agent = GeneratorAgent()
    validator = ValidationAgent()
    benchmark_agent = BenchmarkAgent()
    optimizer = OptimizationAgent()
    reasoner = LLMReasoner() if cfg.enable_llm else None

    auto_detect_target = cfg.auto_detect_target or target_column == "__auto__"
    explicit_target = None if target_column == "__auto__" else target_column
    analysis = analyzer.analyze(
        df,
        target_column=explicit_target,
        auto_detect_target=auto_detect_target,
    )
    log("Analyzer", "Starting dataset analysis")
    log("Analyzer", f"Shape={analysis.rows}x{analysis.columns}; issues={analysis.issues}")

    target = analysis.target_column
    evaluation_mode = "labeled" if target else "unlabeled"
    if target:
        log("Analyzer", f"Target detected: {target}")
    else:
        log("Analyzer", "No target column selected")
    if target and target not in df.columns:
        raise PipelineError(f"Target '{target}' is not in the dataset.")

    llm_summaries: dict[str, str] = {}
    if reasoner:
        llm_summaries["analysis"] = reasoner.explain(
            analysis.model_dump(), "analysis"
        )

    train: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    baseline_metrics = None
    if analysis.task_type:
        train, test = benchmark_agent.split(
            df,
            target,
            analysis.task_type,
            test_size=cfg.test_size,
            random_state=random_state,
        )
        baseline_metrics = benchmark_agent.evaluate_split(
            train, test, target, analysis.task_type, random_state=random_state
        )
        log(
            "Benchmark",
            f"Baseline {baseline_metrics.primary_metric}: {baseline_metrics.primary_value:.4f}",
        )

    plan = planner.plan(
        analysis,
        preferred_generator=cfg.preferred_generator,
        synthetic_ratio=synthetic_ratio,
        random_state=random_state,
        ctgan_epochs=ctgan_epochs,
        tvae_epochs=tvae_epochs,
    )
    log("Planner", f"Dataset classified as {evaluation_mode}")
    log("Planner", f"Selected {plan.generator}: {plan.reason}")
    log(
        "SamplePlanner",
        f"Original rows={analysis.rows}; planned synthetic rows={plan.num_samples}; "
        f"mode={plan.generation_mode}; needed={plan.generation_needed}",
    )
    if reasoner:
        llm_summaries["plan"] = reasoner.explain(plan.model_dump(), "plan")

    iterations: list[IterationRecord] = []
    decisions = []
    best_synth: pd.DataFrame | None = None
    best_path: str | None = None
    best_iter: int | None = None
    best_score = float("-inf")
    improved = False
    fit_df = train if train is not None else df
    cumulative_synthetic_rows = 0

    if not plan.generation_needed:
        log("SamplePlanner", plan.sample_count_reason or "Synthetic generation not justified.")
        result = PipelineResult(
            run_id=run_id,
            timestamp=utc_now(),
            evaluation_mode=evaluation_mode,  # type: ignore[arg-type]
            target_column=target,
            dataset_filename=cfg.dataset_filename,
            dataset_analysis=analysis,
            baseline=baseline_metrics,
            iterations=[],
            optimization_history=[],
            agent_logs=logs,
            llm_summaries=llm_summaries,
            final_evaluation={
                "evaluation_mode": evaluation_mode,
                "generation_needed": False,
                "reason": plan.sample_count_reason or plan.reason,
            },
            final_decision="skipped_generation_not_needed",
            improved=False,
            final_dataset_path=None,
            best_iteration=None,
        )
        (output_dir / f"{run_id}_result.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        return PipelineOutput(
            result=result,
            synthetic_df=None,
            original_df=df,
            train_df=train,
            test_df=test,
        )

    for i in range(1, max_iterations + 1):
        max_total_synthetic = int(analysis.rows * settings.max_total_synthetic_ratio)
        max_per_iteration = max(
            1, int(analysis.rows * settings.max_generation_ratio_per_iteration)
        )
        remaining_total = max_total_synthetic - cumulative_synthetic_rows
        if remaining_total <= 0:
            log("Optimizer", "Maximum total synthetic row budget reached")
            break
        allowed_samples = min(max_per_iteration, remaining_total)
        if plan.num_samples > allowed_samples:
            log(
                "SamplePlanner",
                f"Capping iteration request from {plan.num_samples} to {allowed_samples} "
                "due to per-iteration or total synthetic-row budget.",
            )
            plan = plan.model_copy(
                update={
                    "num_samples": allowed_samples,
                    "samples_to_generate": allowed_samples,
                    "sample_count_details": {
                        **plan.sample_count_details,
                        "per_iteration_cap_applied": plan.num_samples > max_per_iteration,
                        "total_budget_cap_applied": True,
                    },
                }
            )

        log("Generator", f"Training {plan.generator}")
        log(
            "SamplePlanner",
            f"Iteration {i}: requesting {plan.num_samples} rows "
            f"(cumulative before={cumulative_synthetic_rows})",
        )
        synth_path = output_dir / f"{run_id}_iter{i}_{plan.generator}.csv"
        synthetic = generator_agent.generate(fit_df, plan, output_path=synth_path)
        log("Generator", f"Generated {len(synthetic)} rows")
        if len(synthetic) != plan.num_samples:
            log(
                "Generator",
                f"Warning: requested {plan.num_samples} rows but generated {len(synthetic)} rows",
            )
        cumulative_synthetic_rows += len(synthetic)

        validation = validator.validate(fit_df, synthetic, random_state=random_state)
        log("Validator", f"Fidelity score: {validation.fidelity_score:.3f}")
        if reasoner:
            llm_summaries[f"validation_{i}"] = reasoner.explain(
                validation.model_dump(), "validation"
            )

        bench = None
        utility = None
        iter_score = validation.overall_score
        if train is not None and test is not None and analysis.task_type:
            bench = benchmark_agent.compare(
                train=train,
                test=test,
                synthetic=synthetic,
                target_column=target,
                task_type=analysis.task_type,
                random_state=random_state,
                min_improvement=cfg.min_improvement,
                min_absolute_improvement=cfg.min_absolute_improvement,
            )
            log(
                "Benchmark",
                f"Augmented {bench.augmented.primary_metric}: {bench.augmented.primary_value:.4f}",
            )
            delta = bench.improvement.get("primary_value")
            delta_text = f"{delta:+.4f}" if delta is not None else "n/a"
            log(
                "Optimizer",
                f"Iteration {i}: Generator={plan.generator}; Synthetic rows={len(synthetic)}; "
                f"Baseline {bench.baseline.primary_metric}={bench.baseline.primary_value:.4f}; "
                f"Augmented {bench.augmented.primary_metric}={bench.augmented.primary_value:.4f}; "
                f"Delta={delta_text}",
            )
            iter_score = bench.augmented.primary_value
            if reasoner:
                llm_summaries[f"iteration_{i}"] = reasoner.explain(
                    {"improved": bench.improved, "benchmark": bench.model_dump()},
                    "iteration",
                )
        else:
            utility = evaluate_unsupervised_utility(
                fit_df, synthetic, random_state=random_state
            )
            knn_text = (
                f"; KNN similarity={utility.knn_similarity_score:.3f}"
                if utility.knn_similarity_score is not None
                else "; KNN comparison unavailable for this small sample"
            )
            log("Utility", f"Unlabeled similarity benchmark: {utility.overall_score:.3f}{knn_text}")
            iter_score = utility.overall_score
            if reasoner:
                llm_summaries[f"iteration_{i}"] = reasoner.explain(
                    {"utility": utility.model_dump(), "validation": validation.model_dump()},
                    "iteration",
                )

        record = IterationRecord(
            iteration=i,
            plan=plan,
            validation=validation,
            benchmark=bench,
            unsupervised_utility=utility,
            samples_requested=plan.num_samples,
            samples_generated=len(synthetic),
            cumulative_synthetic_rows=cumulative_synthetic_rows,
            decision="pending",
            synthetic_rows=len(synthetic),
            synthetic_path=str(synth_path),
        )
        iterations.append(record)
        if iter_score > best_score:
            best_score = iter_score
            best_synth = synthetic
            best_path = str(synth_path)
            best_iter = i

        tried = [rec.plan.generator for rec in iterations]
        if evaluation_mode == "labeled":
            decision = optimizer.decide(
                iteration=i,
                max_iterations=max_iterations,
                current_plan=plan,
                benchmark=bench,
                validation=validation,
                tried_generators=tried,
            )
        else:
            decision = optimizer.decide_unlabeled(
                iteration=i,
                max_iterations=max_iterations,
                current_plan=plan,
                validation=validation,
                utility=utility,
                tried_generators=tried,
            )
        decisions.append(decision)
        record.decision = "accepted" if decision.accept else ("retry" if decision.continue_loop else "stopped")
        if not decision.continue_loop:
            log("Optimizer", decision.reason)
            break

        log("Optimizer", decision.reason)
        log(
            "Optimizer",
            f"Decision=RETRY; Next candidate={decision.next_generator} + {decision.num_samples}",
        )
        epochs = decision.epochs
        if decision.next_generator == "ctgan":
            epochs = epochs or ctgan_epochs
        elif decision.next_generator == "tvae":
            epochs = epochs or tvae_epochs
        candidate_history = list(plan.sample_count_details.get("candidate_history", []))
        candidate_history.append(
            {"generator": plan.generator, "num_samples": plan.num_samples}
        )
        plan = GenerationPlan(
            generator=decision.next_generator,  # type: ignore[arg-type]
            num_samples=decision.num_samples or plan.num_samples,
            reason=decision.reason,
            generation_needed=True,
            generation_mode=plan.generation_mode,
            target_column=plan.target_column,
            target_class=plan.target_class,
            target_strategy=plan.target_strategy,
            original_rows=plan.original_rows,
            current_target_count=plan.current_target_count,
            desired_target_count=plan.desired_target_count,
            samples_to_generate=decision.num_samples or plan.num_samples,
            augmentation_ratio=(
                (decision.num_samples or plan.num_samples) / max(plan.original_rows or analysis.rows, 1)
            ),
            max_allowed_samples=plan.max_allowed_samples,
            sample_count_reason=decision.reason,
            sample_count_details={
                **plan.sample_count_details,
                "adjusted_by_optimizer": True,
                "previous_iteration": i,
                "candidate_history": candidate_history,
            },
            epochs=epochs,
            random_state=random_state,
            preserve_columns=plan.preserve_columns,
        )

    fairness = None
    if (
        cfg.sensitive_column
        and best_synth is not None
        and train is not None
        and test is not None
        and analysis.task_type == "classification"
        and cfg.sensitive_column in test.columns
    ):
        model = DownstreamModel("classification", random_state=random_state)
        aug = pd.concat([train, best_synth[train.columns]], ignore_index=True)
        model.fit(aug.drop(columns=[target]), aug[target].astype(str))
        preds = model.predict(test.drop(columns=[target]))
        fairness = group_performance(
            y_true=test[target].astype(str),
            y_pred=pd.Series(preds, index=test.index),
            groups=test[cfg.sensitive_column],
            task_type="classification",
        )
    elif cfg.sensitive_column and evaluation_mode == "unlabeled":
        log(
            "Fairness",
            "Skipped fairness evaluation: no target column available for supervised group metrics.",
        )

    provenance = None
    if best_synth is not None:
        provenance = provenance_fingerprint(
            best_synth,
            generator=iterations[best_iter - 1].plan.generator if best_iter else "unknown",
            random_seed=random_state,
        )
        meta_path = output_dir / f"{run_id}_provenance.json"
        meta_path.write_text(provenance.model_dump_json(indent=2), encoding="utf-8")

    best_record = None
    if best_iter is not None and iterations:
        best_record = next((rec for rec in iterations if rec.iteration == best_iter), iterations[-1])
        if evaluation_mode == "labeled":
            improved = bool(
                best_record.benchmark
                and best_record.benchmark.improved
                and best_record.validation.passed
            )
        else:
            improved = bool(
                best_record.unsupervised_utility
                and best_record.unsupervised_utility.passed
                and best_record.validation.passed
            )
        best_record.decision = "accepted" if improved else "best_candidate_not_accepted"
        log(
            "Optimizer",
            (
                "Selected the best candidate after evaluating the configured search budget."
                if improved
                else "Selected the highest-scoring candidate, but it did not meet every acceptance threshold."
            ),
        )

    final_decision = "completed_improved" if improved else "completed_no_improvement"
    if best_synth is None:
        final_decision = "failed"
        log("Pipeline", "No synthetic dataset was produced")
    else:
        log("Pipeline", "Completed successfully")

    final_evaluation = {}
    if best_record is not None:
        final_evaluation = {
            "evaluation_mode": evaluation_mode,
            "validation": best_record.validation.model_dump(),
            "benchmark": best_record.benchmark.model_dump() if best_record.benchmark else None,
            "unsupervised_utility": (
                best_record.unsupervised_utility.model_dump()
                if best_record.unsupervised_utility
                else None
            ),
        }

    result = PipelineResult(
        run_id=run_id,
        timestamp=utc_now(),
        evaluation_mode=evaluation_mode,  # type: ignore[arg-type]
        target_column=target,
        dataset_filename=cfg.dataset_filename,
        dataset_analysis=analysis,
        baseline=baseline_metrics,
        iterations=iterations,
        optimization_history=decisions,
        agent_logs=logs,
        llm_summaries=llm_summaries,
        fairness=fairness,
        provenance=provenance,
        final_evaluation=final_evaluation,
        final_decision=final_decision,
        improved=improved,
        final_dataset_path=best_path,
        best_iteration=best_iter,
    )
    (output_dir / f"{run_id}_result.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    return PipelineOutput(
        result=result,
        synthetic_df=best_synth,
        original_df=df,
        train_df=train,
        test_df=test,
    )
