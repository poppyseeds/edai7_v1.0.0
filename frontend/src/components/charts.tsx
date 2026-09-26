import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DatasetAnalysis, IterationRecord } from "../types/api";
import { Card } from "./ui";

const tooltip = { contentStyle: { background: "#111720", border: "1px solid #273142", borderRadius: 6 }, labelStyle: { color: "#e7edf7" } };
export function CategoryChart({ analysis }: { analysis: DatasetAnalysis }) {
  const item = Object.entries(analysis.categorical_distributions)[0];
  if (!item) return null;
  const [column, values] = item;
  const data = Object.entries(values).slice(0, 8).map(([name, value]) => ({ name, value: Number(value) }));
  return <Card className="p-5"><h3 className="mb-4 text-sm font-medium">{column} distribution</h3><div className="h-60"><ResponsiveContainer><BarChart data={data}><CartesianGrid stroke="#273142" vertical={false} /><XAxis dataKey="name" stroke="#94a3b8" fontSize={11} /><YAxis stroke="#94a3b8" fontSize={11} /><Tooltip {...tooltip} /><Bar dataKey="value" fill="#66a6ff" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div></Card>;
}
export function OptimizationChart({ iterations, baseline }: { iterations: IterationRecord[]; baseline?: number }) {
  const data = iterations.map((item) => ({ iteration: `#${item.iteration}`, baseline, metric: item.benchmark?.augmented.primary_value ?? item.unsupervised_utility?.overall_score }));
  return <Card className="p-5"><h3 className="mb-4 text-sm font-medium">Utility across iterations</h3><div className="h-72"><ResponsiveContainer><LineChart data={data}><CartesianGrid stroke="#273142" vertical={false} /><XAxis dataKey="iteration" stroke="#94a3b8" /><YAxis stroke="#94a3b8" domain={["auto", "auto"]} /><Tooltip {...tooltip} /><Legend /><Line type="monotone" dataKey="baseline" name="Baseline" stroke="#94a3b8" strokeDasharray="5 5" dot={false} /><Line type="monotone" dataKey="metric" name="Candidate" stroke="#35d4a3" strokeWidth={2} /></LineChart></ResponsiveContainer></div></Card>;
}
