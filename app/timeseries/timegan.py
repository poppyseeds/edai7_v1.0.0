"""PyTorch TimeGAN comparative baseline, implemented from the published objectives.

Reference implementation inspected: jsyoon0823/TimeGAN@8f6181c (Apache-2.0).
This is a fixed-window GRU reimplementation, not a TensorFlow checkpoint port.
"""
import torch
from torch import nn
import torch.nn.functional as F


class RecurrentBlock(nn.Module):
    def __init__(self, inputs, hidden, outputs, layers, sigmoid=True):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, layers, batch_first=True)
        self.projection = nn.Linear(hidden, outputs)
        self.sigmoid = sigmoid

    def forward(self, values):
        values = self.projection(self.gru(values)[0])
        return values.sigmoid() if self.sigmoid else values


class TimeGAN(nn.Module):
    def __init__(self, features, hidden=64, layers=2):
        super().__init__()
        self.features = features
        self.networks = nn.ModuleDict({
            "embedder": RecurrentBlock(features, hidden, hidden, layers),
            "recovery": RecurrentBlock(hidden, hidden, features, layers),
            "generator": RecurrentBlock(features, hidden, hidden, layers),
            "supervisor": RecurrentBlock(hidden, hidden, hidden, max(1, layers-1)),
            "discriminator": RecurrentBlock(hidden, hidden, 1, layers, sigmoid=False),
        })

    def sample(self, count, length, device):
        net = self.networks
        noise = torch.rand(count, length, self.features, device=device)
        return net["recovery"](net["supervisor"](net["generator"](noise))) * 2-1

    def train_phases(self, loader, epochs, learning_rate, device):
        net = self.networks
        def parameters(names):
            return [parameter for name in names for parameter in net[name].parameters()]
        def trainable(names):
            for name, module in net.items():
                module.requires_grad_(name in names)
        def step(optimizer, loss):
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite TimeGAN loss.")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        def supervised(hidden):
            return F.mse_loss(net["supervisor"](hidden)[:, :-1], hidden[:, 1:])
        def adversarial(hidden, real=True):
            logits = net["discriminator"](hidden)
            return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits) if real else torch.zeros_like(logits))

        embed = torch.optim.Adam(parameters(["embedder", "recovery"]), lr=learning_rate)
        generate = torch.optim.Adam(parameters(["generator", "supervisor"]), lr=learning_rate)
        discriminate = torch.optim.Adam(parameters(["discriminator"]), lr=learning_rate)
        history = {}
        trainable(["embedder", "recovery"])
        for _ in range(epochs):
            for (batch,) in loader:
                real = (batch.to(device)+1)/2
                reconstruction = F.mse_loss(net["recovery"](net["embedder"](real)), real)
                step(embed, 10 * torch.sqrt(reconstruction+1e-8))
        history["reconstruction"] = float(reconstruction.detach())
        trainable(["supervisor"])
        for _ in range(epochs):
            for (batch,) in loader:
                hidden = net["embedder"]((batch.to(device)+1)/2).detach()
                supervision = supervised(hidden)
                step(generate, supervision)
        history["supervised"] = float(supervision.detach())
        for _ in range(epochs):
            for (batch,) in loader:
                real = (batch.to(device)+1)/2
                for _ in range(2):
                    trainable(["generator", "supervisor"])
                    embedded = net["embedder"](real).detach()
                    noise = torch.rand_like(real)
                    generated = net["generator"](noise)
                    supervised_generated = net["supervisor"](generated)
                    synthetic = net["recovery"](supervised_generated)
                    moments = (real.mean(0)-synthetic.mean(0)).abs().mean() + (
                        (real.var(0, unbiased=False)+1e-6).sqrt()-(synthetic.var(0, unbiased=False)+1e-6).sqrt()).abs().mean()
                    generator_loss = adversarial(supervised_generated) + adversarial(generated) + 100 * (
                        supervised(embedded)+1e-8).sqrt() + 100 * moments
                    step(generate, generator_loss)
                    trainable(["embedder", "recovery"])
                    embedded = net["embedder"](real)
                    embed_loss = 10 * (F.mse_loss(net["recovery"](embedded), real)+1e-8).sqrt() + 0.1*supervised(embedded)
                    step(embed, embed_loss)
                trainable(["discriminator"])
                with torch.no_grad():
                    embedded = net["embedder"](real)
                    generated = net["generator"](torch.rand_like(real))
                    synthetic = net["supervisor"](generated)
                discriminator_loss = adversarial(embedded) + adversarial(synthetic, False) + adversarial(generated, False)
                if discriminator_loss.detach() > 0.15:
                    step(discriminate, discriminator_loss)
        history.update(generator=float(generator_loss.detach()), discriminator=float(discriminator_loss.detach()),
                       embedding_joint=float(embed_loss.detach()))
        self.requires_grad_(True)
        return history
