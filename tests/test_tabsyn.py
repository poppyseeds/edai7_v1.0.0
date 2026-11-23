import numpy as np
import pandas as pd
import pytest

from app.generators.base import AdvancedConfig, GeneratorConfig, SamplingConfig
from app.generators.tabsyn_generator import TabSynGenerator
from app.research.codec import MixedTableCodec
from app.research.sources import source_status
from app.utils.exceptions import GeneratorError


def test_mixed_codec_roundtrip_missing_types():
    data = pd.DataFrame({"amount": np.linspace(0, 10, 32), "kind": ["a", None, "b", "a"] * 8,
                         "flag": [True, False] * 16})
    data.loc[2, "amount"] = np.nan
    codec = MixedTableCodec()
    num, cat = codec.fit_transform(data)
    restored = codec.inverse(num, cat)
    assert list(restored) == list(data)
    assert restored.kind.fillna("MISSING").tolist() == data.kind.fillna("MISSING").tolist()
    assert restored.amount.isna().equals(data.amount.isna())
    assert np.allclose(restored.amount.fillna(0), data.amount.fillna(0), atol=1e-5)


@pytest.mark.skipif(source_status("tabsyn") is not None, reason="Install pinned optional TabSyn source first")
def test_real_tabsyn_training_checkpoint_reload_sampling(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    data = pd.DataFrame({"amount": np.linspace(10, 100, 32), "kind": ["a", "b"] * 16})
    config = GeneratorConfig(epochs=1, batch_size=16, advanced=AdvancedConfig(
        hidden_dim=16, layers=1, factor=2, sampling_steps=3))
    model = TabSynGenerator().fit(data, config)
    metadata = model.get_metadata()
    assert metadata["vae_final_batch_loss"] >= 0
    restored = TabSynGenerator.load(metadata["checkpoint_dir"])
    output = restored.sample(5)
    assert output.shape == (5, 2) and set(output.kind) <= {"a", "b"}
    assert np.isfinite(output.amount).all()
    with pytest.raises(GeneratorError, match="unconditional"):
        restored.sample(2, SamplingConfig(condition_column="kind", condition_value="a"))
