"""Small local worker using only selected upstream model modules, never its CLI dataset loader."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def tabsyn(operation, directory, count):
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from app.generators.base import GeneratorConfig, AdvancedConfig
    from app.generators.runtime import resolve_device, seeded_execution
    from app.research.codec import MixedTableCodec

    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    cfg = GeneratorConfig(**request["config"])
    advanced = cfg.advanced or AdvancedConfig()
    torch.set_num_threads(advanced.threads)
    device = resolve_device(cfg.device, True)
    if advanced.hidden_dim % 2:
        raise ValueError("TabSyn hidden_dim must be even.")
    sys.path.insert(0, request["source"])
    from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
    from tabsyn.model import Model, MLPDiffusion
    from tabsyn.diffusion_utils import sample
    codec = MixedTableCodec()
    with seeded_execution(cfg.random_state, neural=True, device=device):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if operation == "fit":
            numerical, categorical = codec.fit_transform(pd.read_csv(directory / "training.csv"))
            (directory / "codec.json").write_text(json.dumps(codec.state), encoding="utf-8")
        else:
            codec.state = json.loads((directory / "codec.json").read_text(encoding="utf-8"))
        n_num = len(codec.state["numerical"])
        categories = [len(c["values"]) for c in codec.state["categorical"]]
        args = (advanced.layers, n_num, categories, advanced.token_dim)
        vae = Model_VAE(*args, n_head=1, factor=advanced.factor).to(device)
        encoder = Encoder_model(*args, n_head=1, factor=advanced.factor).to(device)
        decoder = Decoder_model(*args, n_head=1, factor=advanced.factor).to(device)
        dimension = (n_num + len(categories)) * advanced.token_dim
        diffusion = Model(MLPDiffusion(dimension, advanced.hidden_dim), dimension).to(device)
        if operation == "fit":
            loader = DataLoader(TensorDataset(torch.from_numpy(numerical), torch.from_numpy(categorical)),
                                batch_size=cfg.batch_size or 256, shuffle=True)
            optimizer = torch.optim.Adam(vae.parameters(), lr=advanced.learning_rate)
            losses = []
            for _ in range(cfg.epochs or advanced.vae_epochs):
                for num, cat in loader:
                    num, cat = num.to(device), cat.to(device)
                    reconstructed, logits, mean, logvar = vae(num, cat)
                    loss = (F.mse_loss(reconstructed, num) if n_num else mean.sum() * 0)
                    if categories:
                        loss = loss + sum(F.cross_entropy(logit, cat[:, i]) for i, logit in enumerate(logits)) / len(categories)
                    loss = loss - advanced.beta * 0.5 * (1 + logvar - mean.square() - logvar.exp()).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite TabSyn VAE loss.")
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    losses.append(float(loss.detach()))
            encoder.load_weights(vae); decoder.load_weights(vae)
            encoder.eval()
            latents = []
            with torch.no_grad():
                for num, cat in DataLoader(loader.dataset, batch_size=cfg.batch_size or 256):
                    latents.append(encoder(num.to(device), cat.to(device))[:, 1:].flatten(1).cpu())
            latent = torch.cat(latents)
            latent_mean = latent.mean(0)
            latent = (latent - latent_mean) / 2  # upstream TabSyn normalization
            optimizer = torch.optim.Adam(diffusion.parameters(), lr=advanced.learning_rate)
            diffusion_losses = []
            for _ in range(cfg.epochs or advanced.diffusion_epochs):
                for (batch,) in DataLoader(TensorDataset(latent), batch_size=cfg.batch_size or 256, shuffle=True):
                    loss = diffusion(batch.to(device))
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite TabSyn diffusion loss.")
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    diffusion_losses.append(float(loss.detach()))
            torch.save({"decoder": decoder.cpu().state_dict(), "diffusion": diffusion.cpu().state_dict(),
                        "latent_mean": latent_mean}, directory / "checkpoint.pt")
            metadata = {"generator": "tabsyn", "source_commit": request["source_commit"], "license": "Apache-2.0",
                "config": cfg.model_dump(), "device": device, "training_seconds": time.perf_counter()-started,
                "vae_final_batch_loss": losses[-1], "diffusion_final_batch_loss": diffusion_losses[-1],
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                "model_size_bytes": (directory / "checkpoint.pt").stat().st_size,
                "adaptations": "upstream core networks/loss/sampler; local training-only quantile codec, fixed epoch schedule, unlabeled support; not paper reproduction"}
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        else:
            checkpoint = torch.load(directory / "checkpoint.pt", map_location=device, weights_only=True)
            decoder.load_state_dict(checkpoint["decoder"]); decoder.eval()
            diffusion.load_state_dict(checkpoint["diffusion"]); diffusion.eval()
            rows = []
            with torch.no_grad():
                for start in range(0, count, cfg.batch_size or 256):
                    latent = sample(diffusion.denoise_fn_D, min(cfg.batch_size or 256, count-start), dimension,
                                    num_steps=advanced.sampling_steps, device=device)
                    latent = latent * 2 + checkpoint["latent_mean"].to(device)
                    numerical, logits = decoder(latent.reshape(-1, n_num+len(categories), advanced.token_dim))
                    cat = torch.stack([x.argmax(1) for x in logits], 1).cpu().numpy() if logits else np.empty((len(latent), 0), dtype=int)
                    rows.append(codec.inverse(numerical.cpu().numpy(), cat))
            pd.concat(rows, ignore_index=True).to_csv(directory / "synthetic.csv", index=False)


def diffusion_ts(operation, directory, count):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from app.generators.runtime import resolve_device, seeded_execution
    from app.timeseries.generators import SequenceConfig

    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    cfg = SequenceConfig(**request["config"])
    torch.set_num_threads(cfg.threads)
    device = resolve_device(cfg.device, True)
    sys.path.insert(0, request["source"])
    from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
    with seeded_execution(cfg.random_state, neural=True, device=device):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = Diffusion_TS(seq_length=request["shape"][1], feature_size=request["shape"][2],
            n_layer_enc=cfg.layers, n_layer_dec=cfg.layers, d_model=cfg.hidden_dim,
            n_heads=cfg.heads, timesteps=cfg.timesteps, sampling_timesteps=cfg.sampling_steps,
            kernel_size=1, padding_size=0, use_ff=True).to(device)
        if operation == "fit":
            data = torch.from_numpy(np.load(directory / "training.npy", allow_pickle=False))
            loader = DataLoader(TensorDataset(data), batch_size=cfg.batch_size, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
            for _ in range(cfg.epochs):
                for (batch,) in loader:
                    loss = model(batch.to(device))
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite Diffusion-TS training loss.")
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            torch.save(model.cpu().state_dict(), directory / "checkpoint.pt")
            metadata = {"generator": "diffusion_ts", "source_commit": request["source_commit"],
                "license": "MIT", "config": cfg.model_dump(), "device": device,
                "training_seconds": time.perf_counter()-started, "final_batch_loss": float(loss.detach()),
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                "model_size_bytes": (directory / "checkpoint.pt").stat().st_size,
                "adaptations": "pinned upstream Diffusion_TS network, Fourier loss and sampler; local fixed-epoch Adam loop without upstream EMA/scheduler; fixed sequence length"}
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        else:
            model.load_state_dict(torch.load(directory / "checkpoint.pt", map_location=device, weights_only=True))
            model.eval()
            batches = []
            with torch.no_grad():
                for start in range(0, count, cfg.batch_size):
                    batches.append(model.generate_mts(batch_size=min(cfg.batch_size, count-start)).cpu().numpy())
            np.save(directory / "synthetic.npy", np.concatenate(batches))


def timegan(operation, directory, count):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from app.generators.runtime import resolve_device, seeded_execution
    from app.timeseries.generators import SequenceConfig
    from app.timeseries.timegan import TimeGAN

    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    cfg = SequenceConfig(**request["config"])
    torch.set_num_threads(cfg.threads)
    device = resolve_device(cfg.device, True)
    with seeded_execution(cfg.random_state, neural=True, device=device):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = TimeGAN(request["shape"][2], cfg.hidden_dim, cfg.layers).to(device)
        if operation == "fit":
            loader = DataLoader(TensorDataset(torch.from_numpy(np.load(directory / "training.npy", allow_pickle=False))),
                                batch_size=cfg.batch_size, shuffle=True)
            losses = model.train_phases(loader, cfg.epochs, cfg.learning_rate, device)
            if device == "cuda":
                torch.cuda.synchronize()
            torch.save(model.cpu().state_dict(), directory / "checkpoint.pt")
            metadata = {"generator": "timegan_torch", "implementation": "local PyTorch fixed-window GRU reimplementation",
                "reference_commit": "8f6181cb9b9d2fa0c930cd902411d9ac8a308e07", "reference_license": "Apache-2.0",
                "config": cfg.model_dump(), "device": device, "losses": losses,
                "training_seconds": time.perf_counter()-started,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                "model_size_bytes": (directory / "checkpoint.pt").stat().st_size,
                "adaptations": "three phases and supervised/adversarial/moment objectives; fixed-length GRUs; PyTorch optimizer/initialization; not bitwise TensorFlow reproduction"}
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        else:
            model.load_state_dict(torch.load(directory / "checkpoint.pt", map_location=device, weights_only=True))
            model.eval()
            with torch.no_grad():
                rows = [model.sample(min(cfg.batch_size, count-start), request["shape"][1], device).cpu().numpy()
                        for start in range(0, count, cfg.batch_size)]
            np.save(directory / "synthetic.npy", np.concatenate(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--operation", choices=["fit", "sample"], required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--count", type=int, default=0)
    args = parser.parse_args()
    if args.model == "tabsyn":
        tabsyn(args.operation, args.directory, args.count)
    elif args.model == "diffusion_ts":
        diffusion_ts(args.operation, args.directory, args.count)
    elif args.model == "timegan_torch":
        timegan(args.operation, args.directory, args.count)
    else:
        raise ValueError("Unknown worker model.")
    if args.operation == "fit":
        import torch
        import numpy as np
        import platform
        metadata_path = args.directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["worker_environment"] = {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name() if metadata.get("device") == "cuda" else platform.processor()}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
