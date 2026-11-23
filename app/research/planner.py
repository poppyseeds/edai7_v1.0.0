"""Transparent, bounded modality/model recommendation; no global-optimality claim."""
from functools import lru_cache
import json
import subprocess

from app.agents.analyzer import DatasetAnalyzer
from app.generators.registry import get_available_generators
from app.research.sources import source_status, worker_python
from app.timeseries.data import profile_sequences


@lru_cache(maxsize=4)
def worker_hardware(interpreter: str):
    code = "import json,torch,importlib.util; print(json.dumps({'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'einops':importlib.util.find_spec('einops') is not None,'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}))"
    try:
        result = subprocess.run([interpreter, "-c", code], capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout) if result.returncode == 0 else {"error": result.stderr[-500:]}
    except Exception as exc:
        return {"error": str(exc)}


@lru_cache(maxsize=4)
def worker_dependencies(interpreter: str):
    """Fast availability check that deliberately does not import torch/CUDA."""
    code = "import json,importlib.util; print(json.dumps({'torch':importlib.util.find_spec('torch') is not None,'einops':importlib.util.find_spec('einops') is not None}))"
    try:
        result = subprocess.run([interpreter, "-c", code], capture_output=True, text=True, timeout=5)
        return json.loads(result.stdout) if result.returncode == 0 else {"error": result.stderr[-500:]}
    except Exception as exc:
        return {"error": str(exc)}


def model_inventory(allow_experimental=False, probe_hardware=False):
    interpreter = worker_python()
    dependencies = worker_dependencies(interpreter)
    hardware = worker_hardware(interpreter) if probe_hardware else {
        "status": "not_probed",
        "worker_python": interpreter,
        "note": "Use probe_hardware=true for the slower torch/CUDA probe.",
    }
    tables = get_available_generators(include_experimental=True)
    for name, status in tables.items():
        status["modality"] = "tabular"
        if status["capabilities"]["experimental"] and not allow_experimental:
            status.update(available=False, reason="experimental_opt_in_required")
    sequence = {
        "diffusion_ts": {"modality": "time_series", "available": source_status("diffusion_ts") is None and dependencies.get("einops", False),
                         "reason": source_status("diffusion_ts") or (None if dependencies.get("einops") else "worker_einops_missing"),
                         "supports_cpu": True, "supports_gpu": True, "input_shape": "N x fixed L x numerical features"},
        "timegan_torch": {"modality": "time_series", "available": dependencies.get("torch", False),
                          "reason": dependencies.get("error") or (None if dependencies.get("torch") else "worker_torch_missing"), "supports_cpu": True, "supports_gpu": True,
                          "implementation": "PyTorch reimplementation, not original TensorFlow runtime"},
        "window_bootstrap": {"modality": "time_series", "available": True, "reason": None,
                             "supports_cpu": True, "supports_gpu": False, "role": "exact-window copy baseline"},
    }
    return {"models": {**tables, **sequence}, "worker_hardware": hardware,
            "worker_dependencies": dependencies,
            "resource_note": "Full-run memory and training costs are unmeasured; the tiny smoke run does not establish 8 GB feasibility for arbitrary schemas."}


def plan_experiment(frame, config):
    inventory = model_inventory(config.allow_experimental)
    if config.modality == "time_series":
        profile = profile_sequences(frame, config.sequence_schema)
        if profile["irregular"]:
            return {**inventory, "profile": profile, "recommended": None,
                    "reason": "Irregular intervals need an explicit regularization policy before these fixed-interval models can run."}
        names = [name for name in ("diffusion_ts", "timegan_torch", "window_bootstrap") if inventory["models"][name]["available"]]
        recommended = names[0]
        reason = f"{recommended} supports fixed-length multivariate sequences; entity-aware chronological windows use length {config.sequence_schema.sequence_length}."
    else:
        analysis = DatasetAnalyzer().analyze(frame, target_column=config.target)
        profile = analysis.model_dump()
        recommended = "tabsyn" if len(frame) >= 1000 and config.allow_experimental and inventory["models"]["tabsyn"]["available"] else "gaussian_copula"
        reason = ("TabSyn is available for a larger mixed-type table; compare measured quality/cost against statistical baselines."
                  if recommended == "tabsyn" else "Gaussian Copula is a lower-cost, available baseline for initial table experiments.")
    return {**inventory, "profile": profile, "recommended": recommended, "reason": reason,
            "selection_limit": "Rule-based recommendation, not a proven globally optimal generator."}
