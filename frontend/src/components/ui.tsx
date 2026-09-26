import type { ButtonHTMLAttributes, PropsWithChildren } from "react";

export function Card({ children, className = "" }: PropsWithChildren<{ className?: string }>) {
  return <section className={`rounded-lg border border-line bg-panel/90 shadow-panel ${className}`}>{children}</section>;
}
export function Button({ children, className = "", ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button className={`inline-flex items-center justify-center gap-2 rounded-md px-4 py-2.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-45 ${className}`} {...props}>{children}</button>;
}
export function Metric({ label, value, detail }: { label: string; value: string | number; detail?: string }) {
  return <Card className="p-4"><p className="text-xs uppercase tracking-[0.12em] text-muted">{label}</p><p className="mt-2 text-2xl font-semibold text-ink">{value}</p>{detail && <p className="mt-1 text-xs text-muted">{detail}</p>}</Card>;
}
export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return <Card className="grid min-h-64 place-items-center p-8 text-center"><div><h2 className="text-lg font-semibold text-ink">{title}</h2><p className="mt-2 max-w-md text-sm text-muted">{detail}</p></div></Card>;
}
export function StatusPill({ children, tone = "neutral" }: PropsWithChildren<{ tone?: "success" | "warn" | "danger" | "neutral" }>) {
  const tones = { success: "bg-emerald-400/10 text-emerald-300", warn: "bg-amber-400/10 text-amber-200", danger: "bg-rose-400/10 text-rose-200", neutral: "bg-slate-500/15 text-slate-300" };
  return <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium ${tones[tone]}`}>{children}</span>;
}
