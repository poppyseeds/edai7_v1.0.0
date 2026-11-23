"""Central generator discovery, capability metadata, and optional availability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from importlib.util import find_spec
from typing import Any, Callable

from app.generators.bootstrap_generator import BootstrapGenerator
from app.generators.copula_generator import GaussianCopulaGenerator
from app.generators.ctgan_generator import CTGANGenerator
from app.generators.tvae_generator import TVAEGenerator
from app.generators.tabsyn_generator import TabSynGenerator
from app.research.sources import source_status
from app.utils.exceptions import GeneratorError


@dataclass(frozen=True)
class GeneratorCapabilities:
    family: str
    supports_conditional_sampling: bool
    supports_mixed_data: bool = True
    supports_cpu: bool = True
    gpu_recommended: bool = False
    cost: str = "medium"
    experimental: bool = False
    modalities: tuple[str, ...] = ("tabular",)
    data_types: tuple[str, ...] = ("numerical", "categorical", "boolean", "datetime")
    input_shape: str = "rows x columns (pandas DataFrame)"
    missing_values: str = "adapter preprocessing; retain missingness where supported"
    preprocessing: str = "training-only SDV metadata and reversible transforms"
    supports_gpu: bool = False
    conditional_method: str = "none"
    configuration_options: tuple[str, ...] = ("random_state", "device")
    execution: str = "in_process"


@dataclass(frozen=True)
class GeneratorRegistration:
    factory: Callable[[], Any] | None
    dependency: str | None
    capabilities: GeneratorCapabilities
    documentation: str
    availability_check: Callable[[], str | None] | None = None


GENERATOR_REGISTRY: dict[str, GeneratorRegistration] = {
    "bootstrap": GeneratorRegistration(BootstrapGenerator, None, GeneratorCapabilities("resampling", True, cost="low", preprocessing="empirical resampling and numerical perturbation", conditional_method="filter_training_rows"), "Built-in empirical bootstrap adapter."),
    "gaussian_copula": GeneratorRegistration(GaussianCopulaGenerator, "sdv", GeneratorCapabilities("statistical", True, cost="low", conditional_method="sdv_conditional"), "SDV GaussianCopulaSynthesizer."),
    "ctgan": GeneratorRegistration(CTGANGenerator, "sdv", GeneratorCapabilities("gan", True, gpu_recommended=True, supports_gpu=True, conditional_method="sdv_rejection_sampling", configuration_options=("random_state", "device", "epochs", "batch_size")), "SDV CTGANSynthesizer."),
    "tvae": GeneratorRegistration(TVAEGenerator, "sdv", GeneratorCapabilities("vae", True, gpu_recommended=True, supports_gpu=True, conditional_method="sdv_rejection_sampling", configuration_options=("random_state", "device", "epochs", "batch_size")), "SDV TVAESynthesizer; conditional requests use rejection sampling, not a conditional VAE."),
    # These packages are deliberately optional. No fallback is used under their names.
    "tabddpm": GeneratorRegistration(None, "tab_ddpm", GeneratorCapabilities("diffusion", False, gpu_recommended=True, cost="high"), "Optional TabDDPM implementation; adapter unavailable until a compatible package is installed."),
    "forest_flow": GeneratorRegistration(None, "forest_diffusion", GeneratorCapabilities("tree-flow", False, cost="medium"), "Optional ForestDiffusion/ForestFlow implementation; compatibility adapter required."),
    "forest_diffusion": GeneratorRegistration(None, "forest_diffusion", GeneratorCapabilities("tree-diffusion", False, cost="medium"), "Optional ForestDiffusion implementation; compatibility adapter required."),
    "ctabgan_plus": GeneratorRegistration(None, "ctabganplus", GeneratorCapabilities("gan", False, gpu_recommended=True, cost="high"), "Optional CTAB-GAN+ implementation; compatibility adapter required."),
    "tabsyn": GeneratorRegistration(TabSynGenerator, "torch", GeneratorCapabilities("latent-diffusion", False, gpu_recommended=True, cost="high", experimental=True, supports_gpu=True, execution="worker_process", configuration_options=("random_state", "device", "epochs", "batch_size", "advanced")), "Pinned upstream TabSyn core in an isolated worker; local training-only codec and explicit schedule.", lambda: source_status("tabsyn")),
}


def register_generator(name: str, registration: GeneratorRegistration) -> None:
    """Register before serving requests; never overwrite an existing model silently."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise GeneratorError("Generator names must be lowercase letters, digits, and underscores.")
    if name in GENERATOR_REGISTRY:
        raise GeneratorError(f"Generator '{name}' is already registered.")
    GENERATOR_REGISTRY[name] = registration


def require_generator(name: str, include_experimental: bool = False) -> GeneratorRegistration:
    registration = GENERATOR_REGISTRY.get(name)
    if registration is None:
        raise GeneratorError(f"Unknown generator '{name}'.")
    status = get_available_generators(include_experimental=True)[name]
    if registration.capabilities.experimental and not include_experimental:
        raise GeneratorError(f"Generator '{name}' is experimental and disabled.")
    if not status["available"]:
        raise GeneratorError(f"Generator '{name}' is unavailable: {status['reason']}.")
    return registration


def get_available_generators(include_experimental: bool = False) -> dict[str, dict[str, Any]]:
    """Return capability and dependency status without importing optional libraries."""

    results: dict[str, dict[str, Any]] = {}
    for name, registration in GENERATOR_REGISTRY.items():
        if registration.capabilities.experimental and not include_experimental:
            continue
        dependency_available = registration.dependency is None or find_spec(registration.dependency) is not None
        available = registration.factory is not None and dependency_available
        reason = None
        if registration.capabilities.experimental and not include_experimental:
            reason = "experimental_generators_disabled"
        elif registration.factory is None:
            reason = "compatible_adapter_not_installed"
        elif not dependency_available:
            reason = f"optional_dependency_missing:{registration.dependency}"
        elif registration.availability_check:
            reason = registration.availability_check()
            available = reason is None
        results[name] = {
            "available": available,
            "reason": reason,
            "capabilities": asdict(registration.capabilities),
            "documentation": registration.documentation,
        }
    return results


def create_generator(name: str, include_experimental: bool = False):
    registration = require_generator(name, include_experimental)
    model = registration.factory()
    required = ("fit", "sample", "generate", "get_metadata", "validate_input", "get_capabilities")
    if getattr(model, "name", None) != name or not all(callable(getattr(model, method, None)) for method in required):
        raise GeneratorError(f"Registered adapter '{name}' does not implement its declared lifecycle/name.")
    return model
