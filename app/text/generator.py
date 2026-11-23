"""Deterministic text baseline; deliberately does not call a hosted model."""
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TextGenerationConfig:
    template: str = "Record {record_id}: "
    fields: tuple[str, ...] = field(default_factory=tuple)
    missing_value: str = "unknown"


class TemplateTextGenerator:
    """Render structured records into auditable text using a local template."""

    name = "template_local"

    def __init__(self, config: TextGenerationConfig | None = None):
        self.config = config or TextGenerationConfig()

    def generate(self, records: list[Mapping[str, Any]]) -> list[str]:
        fields = self.config.fields or tuple(records[0].keys()) if records else ()
        rendered = []
        for index, record in enumerate(records):
            values = {key: record.get(key, self.config.missing_value) for key in fields}
            values["record_id"] = record.get("record_id", index)
            details = ", ".join(f"{key}={values[key]}" for key in fields)
            rendered.append(self.config.template.format(**values) + details)
        return rendered
