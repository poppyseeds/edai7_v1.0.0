import type { PipelineJob, PlanResponse, ResearchModelsResponse, TemporalProfileResponse } from "../types/api";

const API_URL = import.meta.env.VITE_API_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new Error("Cannot reach the API. Confirm the backend is running on port 8000.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed with status ${response.status}.`);
  }
  return response.json() as Promise<T>;
}

function form(file: File, options: Record<string, string | number | boolean | undefined>) {
  const body = new FormData(); body.append("file", file);
  Object.entries(options).forEach(([key, value]) => { if (value !== undefined) body.append(key, String(value)); });
  return body;
}

export const api = {
  plan: (file: File, targetColumn: string | null, autoDetect: boolean) => request<PlanResponse>("/plan", { method: "POST", body: form(file, { target_column: targetColumn ?? "", auto_detect_target: autoDetect }) }),
  startRun: (file: File, options: { targetColumn: string | null; autoDetect: boolean; maxIterations: number; preferredGenerator?: string; enableLlm: boolean }) => request<PipelineJob>("/pipeline-runs", { method: "POST", body: form(file, { target_column: options.targetColumn ?? "", auto_detect_target: options.autoDetect, max_iterations: options.maxIterations, preferred_generator: options.preferredGenerator, enable_llm: options.enableLlm }) }),
  profileTimeSeries: (file: File, options: { timestamp: string; entity: string; features: string[]; context: string[]; sequenceLength: number; stride: number; missingPolicy: string }) => request<TemporalProfileResponse>("/time-series/profile", { method: "POST", body: form(file, { timestamp_column: options.timestamp, entity_column: options.entity, feature_columns: JSON.stringify(options.features), context_columns: JSON.stringify(options.context), sequence_length: options.sequenceLength, stride: options.stride, missing_policy: options.missingPolicy }) }),
  startTimeSeriesRun: (file: File, options: { timestamp: string; entity: string; features: string[]; context: string[]; sequenceLength: number; stride: number; missingPolicy: string; generator: string; numSequences: number; epochs: number; batchSize: number; device: string; hiddenDim: number; layers: number; heads: number; timesteps: number; samplingSteps: number }) => request<PipelineJob>("/pipeline-runs", { method: "POST", body: form(file, { modality: "time_series", preferred_generator: options.generator, timestamp_column: options.timestamp, entity_column: options.entity, feature_columns: JSON.stringify(options.features), context_columns: JSON.stringify(options.context), sequence_length: options.sequenceLength, stride: options.stride, missing_policy: options.missingPolicy, num_sequences: options.numSequences, temporal_epochs: options.epochs, temporal_batch_size: options.batchSize, device: options.device, hidden_dim: options.hiddenDim, layers: options.layers, heads: options.heads, timesteps: options.timesteps, sampling_steps: options.samplingSteps }) }),
  job: (jobId: string) => request<PipelineJob>(`/pipeline-runs/${jobId}`),
  downloadUrl: (runId: string) => `${API_URL}/runs/${runId}/synthetic-data`,
  experimentDownloadUrl: (runId: string) => `${API_URL}/experiments/${runId}/synthetic-data`,
  researchModels: () => request<ResearchModelsResponse>("/research/models"),
  health: () => request<{ status: string }>("/health"),
  sampleUrl: (sampleId: "labeled" | "unlabeled" | "time_series") => `${API_URL}/samples/${sampleId}`,
  sample: async (sampleId: "labeled" | "unlabeled" | "time_series") => {
    const response = await fetch(`${API_URL}/samples/${sampleId}`);
    if (!response.ok) throw new Error("Sample dataset is unavailable.");
    const filename = response.headers.get("content-disposition")?.match(/filename="?([^";]+)"?/)?.[1] ?? `${sampleId}_sample.csv`;
    return new File([await response.blob()], filename, { type: "text/csv" });
  },
};
