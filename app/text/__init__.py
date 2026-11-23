"""Optional local text generation and cross-modal consistency checks."""

from .generator import TemplateTextGenerator, TextGenerationConfig
from .consistency import evaluate_consistency

__all__ = ["TemplateTextGenerator", "TextGenerationConfig", "evaluate_consistency"]
