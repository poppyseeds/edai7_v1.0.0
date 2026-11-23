"""Generate a readable structural AST and call-flow guide for this project."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "app"
JSON_OUTPUT = ROOT / "PROJECT_AST.json"
MARKDOWN_OUTPUT = ROOT / "PROJECT_AST.md"


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def calls_in(node: ast.AST) -> list[str]:
    calls = {
        name
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and (name := dotted_name(child.func))
    }
    return sorted(calls)


def function_record(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
    arguments = [argument.arg for argument in [*node.args.posonlyargs, *node.args.args]]
    if node.args.vararg:
        arguments.append(f"*{node.args.vararg.arg}")
    arguments.extend(argument.arg for argument in node.args.kwonlyargs)
    if node.args.kwarg:
        arguments.append(f"**{node.args.kwarg.arg}")
    return {
        "name": node.name,
        "kind": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
        "line": node.lineno,
        "arguments": arguments,
        "calls": calls_in(node),
    }


def module_record(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    module = path.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")
    imports = []
    functions = []
    classes = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            imports.extend(f"{prefix}.{alias.name}".strip(".") for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(function_record(node))
        elif isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "bases": [name for base in node.bases if (name := dotted_name(base))],
                "methods": [
                    function_record(child)
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                ],
            })
    return {
        "module": module,
        "path": path.relative_to(ROOT).as_posix(),
        "imports": sorted(set(imports)),
        "internal_imports": sorted({name for name in imports if name.startswith("app")}),
        "functions": functions,
        "classes": classes,
    }


def mermaid_flows() -> str:
    return """## 1. Main tabular API flow

```mermaid
flowchart TD
    UI[React dashboard] --> API[FastAPI /pipeline-runs]
    API --> START[start_pipeline_run]
    START --> PIPE[run_pipeline / _run_pipeline]
    PIPE --> ANALYZE[DatasetAnalyzer.analyze]
    ANALYZE --> SPLIT[BenchmarkAgent.research_split]
    SPLIT --> PLAN[GenerationPlanner.plan]
    PLAN --> SAMPLEPLAN[SamplePlanner.plan]
    PLAN --> AGENT[GeneratorAgent.generate]
    AGENT --> REGISTRY[require_generator / create_generator]
    REGISTRY --> GEN{Selected adapter}
    GEN --> BOOT[BootstrapGenerator]
    GEN --> SDV[SDVGenerator: Copula / CTGAN / TVAE]
    GEN --> TABSYN[TabSyn worker]
    BOOT --> SYNTH[Synthetic dataframe]
    SDV --> SYNTH
    TABSYN --> SYNTH
    SYNTH --> VALIDATE[ValidationAgent.validate]
    VALIDATE --> METRICS[Fidelity / constraints / diversity / privacy]
    SYNTH --> BENCH[BenchmarkAgent.evaluate_split]
    BENCH --> SELECT{Quality + utility + hard gates pass?}
    SELECT -->|no| NEXT[Next dataset-ranked compatible generator]
    NEXT --> AGENT
    SELECT -->|yes: early stop| FINAL[Final held-out benchmark]
    NEXT -->|ranking exhausted| FINAL
    FINAL --> REPORT[reporting + manifest + downloadable CSV]
```

## 2. Generator lifecycle

```mermaid
flowchart LR
    CONFIG[GeneratorConfig] --> FIT[fit training rows]
    FIT --> MODEL[Fitted generator state]
    CONDITION[SamplingConfig] --> SAMPLE[sample requested rows]
    MODEL --> SAMPLE
    SAMPLE --> CHECK[Schema / count / condition validation]
    CHECK --> META[get_metadata]
    CHECK --> OUTPUT[Synthetic rows]
```

## 3. Time-series research flow

```mermaid
flowchart TD
    CSV[Time-series CSV] --> SCHEMA[TimeSeriesSchema]
    SCHEMA --> PROFILE[profile_sequences]
    PROFILE --> PREP[prepare_sequences]
    PREP --> CHRONO[Per-entity chronological split]
    CHRONO --> SCALE[Training-only scaling / missing policy]
    SCALE --> WINDOWS[sequence x timestep x feature tensors]
    WINDOWS --> SELECT[select_experiment_candidate]
    SELECT --> DIFF[Diffusion-TS worker]
    SELECT --> TGAN[timegan_torch worker]
    SELECT --> WBOOT[Window bootstrap]
    DIFF --> GENERATED[Generated sequences]
    TGAN --> GENERATED
    WBOOT --> GENERATED
    GENERATED --> QUALITY[sequence_metrics]
    GENERATED --> PRIVACY[distance_privacy]
    GENERATED --> FORECAST[forecasting_utility]
    QUALITY --> STORE[ExperimentStore SQLite + reports]
    PRIVACY --> STORE
    FORECAST --> STORE
```

## 4. Local text flow

```mermaid
flowchart LR
    RECORDS[Structured records] --> TEMPLATE[TemplateTextGenerator.generate]
    TEMPLATE --> TEXT[Generated local text]
    RECORDS --> RULES[evaluate_consistency rules]
    TEXT --> RULES
    RULES --> RESULT[Per-record violations + consistency rate]
```
"""


def module_section(records: list[dict]) -> str:
    lines = [
        "## 5. Structural AST index",
        "",
        "This section is generated from Python's `ast` module. Lines point to source definitions; calls are syntactic call names and may include library functions.",
        "",
    ]
    for record in records:
        definitions = len(record["functions"]) + sum(len(item["methods"]) for item in record["classes"])
        if not definitions and not record["classes"]:
            continue
        lines.extend([f"### `{record['module']}`", "", f"Source: `{record['path']}`", ""])
        for function in record["functions"]:
            args = ", ".join(function["arguments"])
            calls = ", ".join(f"`{call}`" for call in function["calls"][:12]) or "—"
            lines.append(f"- `{function['name']}({args})` — line {function['line']}; calls: {calls}")
        for klass in record["classes"]:
            bases = f" : {', '.join(klass['bases'])}" if klass["bases"] else ""
            lines.append(f"- class `{klass['name']}{bases}` — line {klass['line']}")
            for method in klass["methods"]:
                args = ", ".join(method["arguments"])
                calls = ", ".join(f"`{call}`" for call in method["calls"][:12]) or "—"
                lines.append(f"  - `{method['name']}({args})` — line {method['line']}; calls: {calls}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    paths = sorted(path for path in SOURCE_ROOT.rglob("*.py") if "__pycache__" not in path.parts)
    records = [module_record(path) for path in paths]
    payload = {
        "format": "project-structural-ast-v1",
        "source_root": "app",
        "module_count": len(records),
        "modules": records,
    }
    JSON_OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    heading = """# Project AST and function-flow guide

Generated from the current `app/` Python source. This is a readable structural AST: modules, classes, functions, methods, imports and syntactic calls. `PROJECT_AST.json` contains the machine-readable index.

Start with the diagrams, then search the structural index for a function name when debugging.

"""
    MARKDOWN_OUTPUT.write_text(heading + mermaid_flows() + "\n" + module_section(records), encoding="utf-8")
    print(json.dumps({"modules": len(records), "json": str(JSON_OUTPUT), "markdown": str(MARKDOWN_OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
