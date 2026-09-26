"""Central generator discovery, capability metadata, and optional availability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Any, Callable

from app.generators.bootstrap_generator import BootstrapGenerator
from app.generators.copula_generator import GaussianCopulaGenerator
from app.generators.ctgan_generator import CTGANGenerator
from app.generators.tvae_generator import TVAEGenerator
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


@dataclass(frozen=True)
class GeneratorRegistration:
    factory: Callable[[], Any] | None
    dependency: str | None
    capabilities: GeneratorCapabilities
    documentation: str


GENERATOR_REGISTRY: dict[str, GeneratorRegistration] = {
    "bootstrap": GeneratorRegistration(BootstrapGenerator, None, GeneratorCapabilities("resampling", True, cost="low"), "Built-in empirical bootstrap adapter."),
    "gaussian_copula": GeneratorRegistration(GaussianCopulaGenerator, "sdv", GeneratorCapabilities("statistical", True, cost="low"), "SDV GaussianCopulaSynthesizer."),
    "ctgan": GeneratorRegistration(CTGANGenerator, "sdv", GeneratorCapabilities("gan", True, gpu_recommended=True), "SDV CTGANSynthesizer."),
    "tvae": GeneratorRegistration(TVAEGenerator, "sdv", GeneratorCapabilities("vae", False, gpu_recommended=True), "SDV TVAESynthesizer."),
    # These packages are deliberately optional. No fallback is used under their names.
    "tabddpm": GeneratorRegistration(None, "tab_ddpm", GeneratorCapabilities("diffusion", False, gpu_recommended=True, cost="high"), "Optional TabDDPM implementation; adapter unavailable until a compatible package is installed."),
    "forest_flow": GeneratorRegistration(None, "forest_diffusion", GeneratorCapabilities("tree-flow", False, cost="medium"), "Optional ForestDiffusion/ForestFlow implementation; compatibility adapter required."),
    "forest_diffusion": GeneratorRegistration(None, "forest_diffusion", GeneratorCapabilities("tree-diffusion", False, cost="medium"), "Optional ForestDiffusion implementation; compatibility adapter required."),
    "ctabgan_plus": GeneratorRegistration(None, "ctabganplus", GeneratorCapabilities("gan", False, gpu_recommended=True, cost="high"), "Optional CTAB-GAN+ implementation; compatibility adapter required."),
    "tabsyn": GeneratorRegistration(None, "tabsyn", GeneratorCapabilities("latent-diffusion", False, gpu_recommended=True, cost="high", experimental=True), "Experimental TabSyn implementation; disabled unless explicitly enabled and adapted."),
}


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
        results[name] = {
            "available": available,
            "reason": reason,
            "capabilities": asdict(registration.capabilities),
            "documentation": registration.documentation,
        }
    return results


def create_generator(name: str):
    registration = GENERATOR_REGISTRY.get(name)
    if registration is None:
        raise GeneratorError(f"Unknown generator '{name}'.")
    status = get_available_generators(include_experimental=True)[name]
    if not status["available"]:
        raise GeneratorError(f"Generator '{name}' is unavailable: {status['reason']}.")
    return registration.factory()
