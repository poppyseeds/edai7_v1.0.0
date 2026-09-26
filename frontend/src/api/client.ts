import type { PipelineJob, PlanResponse } from "../types/api";

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
  job: (jobId: string) => request<PipelineJob>(`/pipeline-runs/${jobId}`),
  downloadUrl: (runId: string) => `${API_URL}/runs/${runId}/synthetic-data`,
  health: () => request<{ status: string }>("/health"),
  sample: async (sampleId: "labeled" | "unlabeled") => {
    const response = await fetch(`${API_URL}/samples/${sampleId}`);
    if (!response.ok) throw new Error("Sample dataset is unavailable.");
    const filename = response.headers.get("content-disposition")?.match(/filename="?([^";]+)"?/)?.[1] ?? `${sampleId}_sample.csv`;
    return new File([await response.blob()], filename, { type: "text/csv" });
  },
};
