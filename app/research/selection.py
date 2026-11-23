"""Validation-only quality/cost Pareto search over an explicit, finite candidate list."""
import time


def select_experiment_candidate(config, names, evaluate):
    trials, completed = [], []
    for name in dict.fromkeys(names):
        started = time.perf_counter()
        try:
            candidate = evaluate(name)
            cost = candidate["fit_time"] + candidate["sample_time"]
            trial = {"generator": name, "status": "completed", "validation_score": candidate["score"],
                     "cost_seconds": cost, "failed_gates": candidate["failed_gates"], "selected": False,
                     "generator_metadata": candidate["metadata"]}
            completed.append((candidate, trial))
        except Exception as exc:
            trial = {"generator": name, "status": "failed", "failure": {"type": type(exc).__name__, "message": str(exc)},
                     "cost_seconds": time.perf_counter()-started, "selected": False}
            if config.selection != "adaptive":
                raise
        trials.append(trial)
    if not completed:
        raise ValueError(f"All configured candidates failed: {trials}")
    eligible = [item for item in completed if not item[0]["failed_gates"]]
    pool = eligible or completed  # retain a rejected audit artifact without claiming acceptance
    frontier = [item for item in pool if not any(
        other[0]["score"] >= item[0]["score"] and other[1]["cost_seconds"] <= item[1]["cost_seconds"]
        and (other[0]["score"] > item[0]["score"] or other[1]["cost_seconds"] < item[1]["cost_seconds"])
        for other in pool if other is not item)]
    maximum_cost = max(item[1]["cost_seconds"] for item in frontier) or 1
    weight = config.cost_weight if config.selection == "adaptive" else 0
    for _, trial in frontier:
        trial["pareto_optimal"] = True
    winner = max(frontier, key=lambda item: item[0]["score"]-weight*item[1]["cost_seconds"]/maximum_cost)
    winner[1]["selected"] = True
    return winner[0], {"strategy": config.selection, "cost_weight": weight, "trials": trials,
        "selection_role": "real_validation_only", "quality_gates_passed": bool(eligible),
        "limitations": "Bounded candidate search, not a proof of global optimality. Failed trials are retained. Final-test metrics cannot change the selected candidate."}
