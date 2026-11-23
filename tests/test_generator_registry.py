from app.generators.registry import create_generator, get_available_generators
from app.utils.exceptions import GeneratorError


def test_builtin_generators_are_available_through_central_registry() -> None:
    status = get_available_generators()

    assert status["bootstrap"]["available"] is True
    assert status["ctgan"]["available"] is True
    assert create_generator("bootstrap").name == "bootstrap"


def test_optional_generator_is_reported_without_a_silent_fallback() -> None:
    status = get_available_generators(include_experimental=True)

    assert status["tabddpm"]["available"] is False
    assert status["tabddpm"]["reason"] == "compatible_adapter_not_installed"
    try:
        create_generator("tabddpm")
    except GeneratorError as exc:
        assert "tabddpm" in str(exc)
    else:  # pragma: no cover - protects against an accidental fake fallback
        raise AssertionError("Optional generator must not silently fall back.")
