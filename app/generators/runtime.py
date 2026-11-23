"""Local device validation and serialized, scoped random-state control."""
from contextlib import contextmanager
import random
from threading import RLock

import numpy as np

from app.utils.exceptions import GeneratorError

MODEL_LOCK = RLock()


def resolve_device(requested: str, supports_gpu: bool, supports_cpu: bool = True) -> str:
    if requested == "cuda" and not supports_gpu:
        raise GeneratorError("This generator supports CPU execution only.")
    if not supports_gpu or requested == "cpu":
        if not supports_cpu:
            raise GeneratorError("This generator does not support CPU execution.")
        return "cpu"
    import torch

    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise GeneratorError("CUDA was requested but is unavailable in this PyTorch environment. Select cpu or configure a CUDA-enabled environment.")
    if not available and not supports_cpu:
        raise GeneratorError("This generator requires CUDA, which is unavailable.")
    return "cuda" if available else "cpu"


@contextmanager
def seeded_execution(seed: int, neural: bool = False, device: str = "cpu"):
    """Seed before fitting and restore RNGs; bitwise CUDA determinism is not promised."""
    with MODEL_LOCK:
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            if neural:
                import torch

                devices = [torch.cuda.current_device()] if device == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(seed)
                    if devices:
                        torch.cuda.manual_seed(seed)
                    yield
            else:
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
