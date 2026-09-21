"""Domain errors with user-facing messages."""


class SyntheticAIError(Exception):
    """Base error for the platform."""


class InvalidDatasetError(SyntheticAIError):
    pass


class GeneratorError(SyntheticAIError):
    pass


class BenchmarkError(SyntheticAIError):
    pass


class PipelineError(SyntheticAIError):
    pass
