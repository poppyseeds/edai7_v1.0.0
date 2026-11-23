export type Page = "Overview" | "Dataset" | "Time Series" | "Generation" | "Validation" | "Benchmark" | "Optimization" | "Results";

export interface DatasetAnalysis {
  rows: number; columns: number; column_names: string[]; numerical_columns: string[];
  categorical_columns: string[]; missing_percentage: number; missing_by_column: Record<string, number>;
  duplicate_percentage: number; target_column?: string | null; target_detected: boolean;
  target_detection_reason?: string | null; task_type?: "classification" | "regression" | null;
  class_distribution?: Record<string, number> | null; numerical_stats: Record<string, Record<string, number>>;
  categorical_distributions: Record<string, Record<string, number>>; correlations: Record<string, Record<string, number>>;
  issues: string[]; issue_details: Record<string, string>;
}
export interface GenerationPlan {
  generator: string; num_samples: number; reason: string; generation_needed: boolean; generation_mode: string;
  target_column?: string | null; target_class?: string | number | null; original_rows?: number | null;
  current_target_count?: number | null; desired_target_count?: number | null; sample_count_details: Record<string, unknown>;
}
export interface ValidationResult { fidelity_score: number; distribution_score: number; correlation_score: number; diversity_score: number; privacy_score: number; overall_score: number; passed: boolean; }
export interface ModelMetrics { model_name: string; primary_metric: string; primary_value: number; accuracy?: number | null; precision?: number | null; recall?: number | null; f1?: number | null; roc_auc?: number | null; r2?: number | null; mae?: number | null; rmse?: number | null; }
export interface BenchmarkResult { baseline: ModelMetrics; augmented: ModelMetrics; improvement: Record<string, number | null>; improved: boolean; relative_improvement?: number; absolute_improvement?: number; min_relative_improvement?: number; min_absolute_improvement?: number; notes: string; }
export interface UnlabeledBenchmark { overall_score: number; distribution_score: number; correlation_score: number; diversity_score: number; structural_score: number; knn_similarity_score?: number | null; knn_discriminator_auc?: number | null; passed: boolean; metric_notes?: Record<string, string>; }
export interface IterationRecord { iteration: number; plan: GenerationPlan; validation: ValidationResult; benchmark?: BenchmarkResult | null; unsupervised_utility?: UnlabeledBenchmark | null; samples_requested: number; samples_generated: number; cumulative_synthetic_rows: number; decision: string; }
export interface PipelineResult { final_benchmark?: BenchmarkResult | null; artifact_eligible?: boolean; run_id: string; evaluation_mode: "labeled" | "unlabeled"; target_column?: string | null; dataset_filename?: string | null; dataset_analysis: DatasetAnalysis; baseline?: ModelMetrics | null; iterations: IterationRecord[]; optimization_history: { reason: string; accept: boolean; next_generator?: string | null; num_samples?: number | null }[]; agent_logs: { agent: string; message: string; timestamp: string }[]; llm_summaries?: Record<string, string>; final_report?: string; final_evaluation: Record<string, unknown>; final_decision: string; improved: boolean; best_iteration?: number | null; }
export interface PlanResponse { filename: string; analysis: DatasetAnalysis; plan: GenerationPlan; preview: Record<string, unknown>[]; }
export interface TemporalProfileResponse { filename: string; columns: string[]; schema: Record<string, unknown>; profile: { rows: number; irregular: boolean; entities: { entity: string; rows: number; interval_ns: number; irregular_intervals: number }[]; missing_by_feature: Record<string, number>; }; window_shapes: Record<string, number[]>; preview: Record<string, unknown>[]; }
export interface TemporalResult { modality: "time_series"; generator: string; artifact_eligible: boolean; quality: { per_feature: { feature_index: number; ks_statistic: number; acf_error?: number | null; spectral_js_divergence?: number | null }[]; cross_correlation_error?: number | null; unique_sequence_ratio: number; }; privacy: Record<string, number | string>; utility?: Record<string, unknown> | null; efficiency: Record<string, number | string | null>; split_counts: Record<string, number>; synthetic_sequence_preview: Record<string, unknown>[]; real_sequence_preview: Record<string, unknown>[]; preprocessing: { schema: { features: string[] } }; selection: Record<string, unknown>; }
export interface ResearchModelStatus { modality?: string; available: boolean; reason?: string | null; supports_cpu?: boolean; supports_gpu?: boolean; implementation?: string; role?: string; }
export interface ResearchModelsResponse { models: Record<string, ResearchModelStatus>; worker_hardware: Record<string, unknown>; worker_dependencies: Record<string, unknown>; resource_note: string; }
export interface PipelineJob { job_id: string; status: "queued" | "running" | "completed" | "failed"; filename: string; modality?: "tabular" | "time_series"; stage?: string; events?: { at: string; stage: string; message: string }[]; run_id?: string; result?: PipelineResult; temporal_result?: TemporalResult; synthetic_preview?: Record<string, unknown>[]; error?: string; }
