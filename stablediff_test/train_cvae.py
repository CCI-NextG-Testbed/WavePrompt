import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from tqdm import tqdm
from stablediff.dataset import from_path
from WavePrompt.stablediff_test.params import params_simple
from WavePrompt.stablediff_test.CVAE import ComplexEncoder, ComplexDecoder

try:
    from rfml.nn.F import evm as rfml_evm
except Exception:
    rfml_evm = None


GIF_MODS = ["BPSK", "QPSK", "8PSK", "16QAM"]


class IQPlusSymLoss(nn.Module):

    def __init__(self, w_time=0.5, eps=1e-8):
        super().__init__()
        self.w_time = float(w_time)
        self.w_evm = 1.0 - self.w_time
        self.eps = eps

    def complex_mse(self, target, est):
        return torch.mean(torch.abs(target - est) ** 2)

    def _symbol_evm_loss(self, est_c, target_c, sps):
        B, N = est_c.shape
        losses = []

        for i in range(B):

            sps_i = max(1, int(sps[i]))
            T = N // sps_i

            if T <= 0:
                continue

            est_i = est_c[i, :T*sps_i].view(T, sps_i).mean(dim=1)
            tgt_i = target_c[i, :T*sps_i].view(T, sps_i).mean(dim=1)

            est_ri = torch.stack(
                (est_i.real, est_i.imag), dim=0
            ).unsqueeze(0).unsqueeze(0)

            tgt_ri = torch.stack(
                (tgt_i.real, tgt_i.imag), dim=0
            ).unsqueeze(0).unsqueeze(0)

            if rfml_evm is not None:
                losses.append(torch.mean(
                    rfml_evm(est_ri, tgt_ri)
                ))
            else:
                num = torch.mean(torch.abs(est_i - tgt_i) ** 2)
                den = torch.mean(torch.abs(tgt_i) ** 2).clamp(min=self.eps)
                losses.append(torch.sqrt(num / den))

        if not losses:
            return torch.tensor(
                0.0,
                device=est_c.device,
                dtype=est_c.real.dtype
            )

        return torch.stack(losses).mean()

    def forward(self, target, est, sps=None):

        if not torch.is_complex(target):
            target = torch.view_as_complex(target)

        if not torch.is_complex(est):
            est = torch.view_as_complex(est)

        if target.ndim == 3:
            target = target.squeeze(1)

        if est.ndim == 3:
            est = est.squeeze(1)

        l_iq = self.complex_mse(target, est)

        if sps is not None:
            l_evm = self._symbol_evm_loss(
                est, target, sps
            )
        else:
            l_evm = torch.tensor(
                0.0,
                device=est.device
            )

        return self.w_time * l_iq + self.w_evm * l_evm

def get_n_per_modulation(dataloader, mods, n):

    counts = {mod: 0 for mod in mods}
    subset = []

    for batch in dataloader:

        modulation = batch["modulation"]

        if isinstance(modulation, str):
            modulation = [modulation]

        for i, mod in enumerate(modulation):

            if counts[mod] >= n:
                continue

            subset.append({
                "data": batch["data"][i:i+1],
                "modulation": [mod]
            })

            counts[mod] += 1

        if all(counts[mod] >= n for mod in mods):
            break

    print("\nWaveforms selected:")
    for mod in mods:
        print(f"  {mod}: {counts[mod]}")

    return subset

def to_complex_samples(x):

    x = x.detach().cpu()

    if torch.is_complex(x):
        return x.flatten()

    if x.shape[-1] == 2:
        x = x.reshape(-1, 2)
        return torch.complex(x[:, 0], x[:, 1])

    return x.flatten().to(torch.complex64)

def discover_modulations(dataloader):
    mods = set()

    for batch in dataloader:
        modulation = batch["modulation"]

        if isinstance(modulation, str):
            modulation = [modulation]

        mods.update(
            str(m) for m in modulation
            if m is not None and str(m).strip()
        )

    return sorted(mods)

def calculate_evm(tx, pred, sps=1):

    n = min(len(tx), len(pred))
    T = n // sps

    tx = tx[:T*sps].view(T, sps).mean(dim=1)
    pred = pred[:T*sps].view(T, sps).mean(dim=1)

    tx_ri = torch.stack(
        (tx.real, tx.imag), dim=0
    ).unsqueeze(0).unsqueeze(0)

    pred_ri = torch.stack(
        (pred.real, pred.imag), dim=0
    ).unsqueeze(0).unsqueeze(0)

    if rfml_evm is not None:
        return torch.mean(
            rfml_evm(pred_ri, tx_ri)
        ).item()

    return (
        torch.mean(torch.abs(pred - tx) ** 2) /
        torch.mean(torch.abs(tx) ** 2)
    ).sqrt().item()


def train_epoch(encoder, decoder, device, dataloader, optimizer, recon_loss):

    encoder.train()
    decoder.train()
    total_loss = 0.0

    for step, batch in enumerate(tqdm(dataloader)):

        x = batch["data"].to(device)

        optimizer.zero_grad()

        z = encoder(x)

        x_hat = decoder(z)

        sps = [1] * x.shape[0]

        loss = recon_loss(
            x,
            x_hat,
            sps=sps
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_evms(encoder, decoder, device, dataloader, mods):

    encoder.eval()
    decoder.eval()

    evms = {mod: [] for mod in mods}

    with torch.no_grad():
        for batch in dataloader:

            x = batch["data"].to(device)
            z = encoder(x)
            x_hat = decoder(z)

            modulation = batch["modulation"]

            if isinstance(modulation, str):
                modulation = [modulation]

            for i, mod in enumerate(modulation):

                if mod is None:
                    continue

                tx = to_complex_samples(x[i:i+1])
                pred = to_complex_samples(x_hat[i:i+1])

                evm = calculate_evm(tx, pred, sps=1)

                if mod in evms:
                    evms[mod].append(evm)

    averages = {
        mod: sum(evms[mod]) / len(evms[mod])
        if evms[mod] else float("nan")
        for mod in mods
    }

    return averages, evms


def get_fixed_samples(dataloader):

    fixed = {}

    for batch in dataloader:

        modulation = batch["modulation"]

        if isinstance(modulation, str):
            modulation = [modulation]

        for i, mod in enumerate(modulation):

            if mod in GIF_MODS and mod not in fixed:
                fixed[mod] = {
                    "data": batch["data"][i:i+1],
                    "modulation": mod
                }

        if len(fixed) == len(GIF_MODS):
            break

    return fixed


def save_epoch_frame(
    encoder,
    decoder,
    device,
    samples,
    epoch,
    save_dir,
    sps=1
):

    encoder.eval()
    decoder.eval()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 10)
    )

    plots = [
        (axes[0, 0], "BPSK"),
        (axes[0, 1], "QPSK"),
        (axes[1, 0], "8PSK"),
        (axes[1, 1], "16QAM"),
    ]

    with torch.no_grad():

        for ax, mod in plots:

            if mod not in samples:
                ax.set_title(f"{mod} - Not Found")
                continue

            x = samples[mod]["data"].to(device)
            z = encoder(x)
            x_hat = decoder(z)

            tx = to_complex_samples(x)
            pred = to_complex_samples(x_hat)

            n = min(len(tx), len(pred))
            T = n // sps

            tx = tx[:T*sps].view(T, sps).mean(dim=1)
            pred = pred[:T*sps].view(T, sps).mean(dim=1)

            ax.scatter(
                tx.real.numpy(),
                tx.imag.numpy(),
                s=18,
                alpha=0.35,
                label="Tx"
            )

            ax.scatter(
                pred.real.numpy(),
                pred.imag.numpy(),
                s=18,
                alpha=0.8,
                label="Pred"
            )

            ax.set_title(f"{mod} Symbols")
            ax.set_xlabel("I")
            ax.set_ylabel("Q")

            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect("equal")

            ax.axhline(
                0,
                linestyle="--",
                linewidth=0.8
            )

            ax.axvline(
                0,
                linestyle="--",
                linewidth=0.8
            )

            ax.grid(True, alpha=0.3)
            ax.legend()

    fig.suptitle(
        f"Epoch {epoch}",
        fontsize=16
    )

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        f"epoch_{epoch:03d}.png"
    )

    plt.savefig(path, dpi=120)
    plt.close(fig)

    return path


def save_evm_plot(history, save_dir):

    plt.figure(figsize=(9, 6))

    for mod in GIF_MODS:
        plt.plot(
            history[mod],
            marker="o",
            label=mod
        )

    plt.xlabel("Epoch")
    plt.ylabel("Average EVM")
    plt.title("Average Reconstruction EVM")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        "evm_vs_epoch.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    return path


def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Selected device: {device}")

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)

    encoder = ComplexEncoder(
        latent_dims=args.latent_dims
    ).to(device)

    decoder = ComplexDecoder(
        latent_dims=args.latent_dims
    ).to(device)

    if args.resume_from is not None:

        checkpoint = torch.load(
            args.resume_from,
            map_location=device
        )

        encoder.load_state_dict(
            checkpoint["encoder_state_dict"]
        )

        decoder.load_state_dict(
            checkpoint["decoder_state_dict"]
        )

        print(
            f"Loaded previous model from: "
            f"{args.resume_from}"
        )

    recon_loss = IQPlusSymLoss(
        w_time=params_simple.loss_w_time
    )

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) +
        list(decoder.parameters()),
        lr=params_simple.learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    train_loader = from_path(params_simple)

    MODS = discover_modulations(train_loader)

    if args.waveforms_per_modulation is not None:
        train_loader = get_n_per_modulation(
            train_loader,
            MODS,
            args.waveforms_per_modulation
        )

    fixed_samples = get_fixed_samples(train_loader)

    print("\nDataset modulations:")
    for mod in MODS:
        print(f"  {mod}")

    frame_dir = os.path.join(
        args.gif_dir,
        "frames"
    )

    os.makedirs(
        frame_dir,
        exist_ok=True
    )

    frames = []

    history = {
        "train_loss": [],
        **{mod: [] for mod in MODS}
    }

    for epoch in range(args.epochs):

        train_loss = train_epoch(
            encoder,
            decoder,
            device,
            train_loader,
            optimizer,
            recon_loss
        )

        # Calculate EVM for EVERY waveform.
        avg_evm, all_evms = evaluate_evms(
            encoder,
            decoder,
            device,
            train_loader,
            MODS
        )

        history["train_loss"].append(train_loss)

        for mod in MODS:
            history[mod].append(
                avg_evm[mod]
            )

        print(
            f"\nEpoch {epoch + 1}/{args.epochs}"
            f"\n  Train Loss: {train_loss:.4f}"
        )

        for mod in MODS:
            print(
                f"  {mod:6s} EVM: "
                f"{avg_evm[mod]:.6f}"
                f"  (N={len(all_evms[mod])})"
            )

        frame_path = save_epoch_frame(
            encoder,
            decoder,
            device,
            fixed_samples,
            epoch + 1,
            frame_dir
        )

        frames.append(frame_path)

        # Use average EVM as scheduler metric.
        mean_evm = sum(
            avg_evm[m] for m in MODS
            if not torch.isnan(
                torch.tensor(avg_evm[m])
            )
        ) / len(MODS)

        scheduler.step(mean_evm)

    # GIF
    gif_path = os.path.join(
        args.gif_dir,
        "cvae_reconstruction.gif"
    )

    images = [
        imageio.imread(frame)
        for frame in frames
    ]

    imageio.mimsave(
        gif_path,
        images,
        duration=int(args.gif_duration * 1000),
        loop=0
    )

    # EVM plot
    evm_plot = save_evm_plot(
        history,
        args.gif_dir
    )

    # Save history
    torch.save(
        history,
        os.path.join(
            args.gif_dir,
            "evm_history.pt"
        )
    )

    print(f"\nGIF saved to: {gif_path}")
    print(f"EVM plot saved to: {evm_plot}")

    # Save model
    torch.save(
        {
            "epoch": args.epochs,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "latent_dims": args.latent_dims
        },
        args.save_dir
    )

    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--latent_dims",
        type=int,
        default=250
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=params_simple.cvae_model_dir
    )

    parser.add_argument(
        "--gif_dir",
        type=str,
        default=params_simple.cvae_gif_dir
    )

    parser.add_argument(
        "--gif_duration",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--waveforms_per_modulation",
        type=int,
        default=None
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=params_simple.cvae_model_dir
    )
    args = parser.parse_args()

    main(args)