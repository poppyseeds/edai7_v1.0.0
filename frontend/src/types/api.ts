export type Page = "Overview" | "Dataset" | "Generation" | "Validation" | "Benchmark" | "Optimization" | "Results";

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
export interface PipelineResult { run_id: string; evaluation_mode: "labeled" | "unlabeled"; target_column?: string | null; dataset_filename?: string | null; dataset_analysis: DatasetAnalysis; baseline?: ModelMetrics | null; iterations: IterationRecord[]; optimization_history: { reason: string; accept: boolean; next_generator?: string | null; num_samples?: number | null }[]; agent_logs: { agent: string; message: string; timestamp: string }[]; llm_summaries?: Record<string, string>; final_evaluation: Record<string, unknown>; final_decision: string; improved: boolean; best_iteration?: number | null; }
export interface PlanResponse { filename: string; analysis: DatasetAnalysis; plan: GenerationPlan; preview: Record<string, unknown>[]; }
export interface PipelineJob { job_id: string; status: "queued" | "running" | "completed" | "failed"; filename: string; run_id?: string; result?: PipelineResult; synthetic_preview?: Record<string, unknown>[]; error?: string; }
