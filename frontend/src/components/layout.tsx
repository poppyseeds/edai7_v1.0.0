import { Activity, BarChart3, BrainCircuit, Database, FlaskConical, Layers3, Settings2, Sparkles } from "lucide-react";
import type { ReactNode } from "react";
import type { Page } from "../types/api";
import { StatusPill } from "./ui";

const items: { name: Page; icon: typeof Activity }[] = [
  { name: "Overview", icon: Layers3 }, { name: "Dataset", icon: Database }, { name: "Generation", icon: Sparkles },
  { name: "Validation", icon: Activity }, { name: "Benchmark", icon: BarChart3 }, { name: "Optimization", icon: BrainCircuit }, { name: "Results", icon: FlaskConical },
];

export function Shell({ page, setPage, filename, status, online, children }: { page: Page; setPage: (page: Page) => void; filename?: string; status: string; online: boolean; children: ReactNode }) {
  return <div className="min-h-screen bg-canvas text-ink"><aside className="fixed inset-y-0 left-0 z-20 hidden w-60 border-r border-line bg-[#0d121a] p-4 lg:block"><div className="flex items-center gap-3 px-2 py-3"><span className="grid size-8 place-items-center rounded-md bg-electric text-[#07101f]"><Sparkles size={17} /></span><span className="font-semibold tracking-wide">Synthetic AI</span></div><nav className="mt-8 space-y-1">{items.map(({ name, icon: Icon }) => <button key={name} onClick={() => setPage(name)} className={`flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm transition ${page === name ? "bg-electric/15 text-electric" : "text-slate-400 hover:bg-white/5 hover:text-slate-100"}`}><Icon size={17} />{name}</button>)}</nav><div className="absolute bottom-5 left-4 right-4 flex items-center gap-2 border-t border-line pt-4 text-xs text-muted"><Settings2 size={14} />API-driven workspace</div></aside><main className="lg:ml-60"><header className="sticky top-0 z-10 flex min-h-16 items-center justify-between border-b border-line bg-canvas/90 px-5 backdrop-blur lg:px-8"><div><p className="text-xs text-muted">Current dataset</p><p className="max-w-64 truncate text-sm font-medium">{filename ?? "No dataset selected"}</p></div><div className="flex items-center gap-3"><StatusPill tone={status === "completed" ? "success" : status === "failed" ? "danger" : "neutral"}>{status}</StatusPill><StatusPill tone={online ? "success" : "danger"}>{online ? "System online" : "API unavailable"}</StatusPill></div></header><div className="mx-auto max-w-7xl p-5 lg:p-8">{children}</div></main></div>;
}
