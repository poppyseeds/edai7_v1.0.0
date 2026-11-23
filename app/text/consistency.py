"""Small, deterministic cross-modal checks for generated structured records and text."""
import re
from typing import Any, Mapping, Sequence


def evaluate_consistency(records: Sequence[Mapping[str, Any]], texts: Sequence[str], rules: Mapping[str, Mapping[str, str]] | None = None):
    if len(records) != len(texts):
        raise ValueError("records and texts must have the same length")
    rules = rules or {}
    results = []
    for record, text in zip(records, texts):
        text_lower = text.lower()
        violations = []
        matched = []
        for field, mapping in rules.items():
            if field not in record:
                continue
            value = str(record[field])
            expected = mapping.get(value)
            if expected is None:
                continue
            if re.search(re.escape(expected.lower()), text_lower):
                matched.append(field)
            else:
                violations.append({"field": field, "value": value, "expected_text": expected})
        results.append({"consistent": not violations, "matched_fields": matched, "violations": violations})
    consistent = sum(item["consistent"] for item in results)
    return {"records": results, "count": len(results), "consistent_count": consistent,
            "consistency_rate": consistent / len(results) if results else 1.0,
            "method": "case-insensitive exact rule matching; no embedding or hosted model"}
