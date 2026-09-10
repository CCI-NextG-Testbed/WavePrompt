# prediction_cvae.py

import torch
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import glob
import random


from stablediff.CVAE import CVAE
import argparse
from stablediff.params import params_simple

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load model
    # -----------------------------
    model = CVAE(latent_dims=args.latent_dims).to(device)

    checkpoint = torch.load(
        args.model_dir,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # -----------------------------
    # Load complex sample
    # -----------------------------
    files = glob.glob(f"{args.data_path}/**/*.mat", recursive=True)

    data_path = random.choice(files)

    mat = scio.loadmat(data_path)
    x = np.asarray(mat["data"])

    if not np.iscomplexobj(x):
        x = x[..., 0] + 1j * x[..., 1]

    x = x.reshape(-1)[:args.N]

    # Normalize like training
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))

    # [1, 1, 4096]
    x = torch.from_numpy(x.astype(np.complex64))
    x = x.view(1, 1, args.N).to(device)

    # -----------------------------
    # Reconstruction
    # -----------------------------
    with torch.no_grad():
        x_hat = model(x)

    # -----------------------------
    # Metrics
    # -----------------------------
    mse = torch.mean(torch.abs(x - x_hat) ** 2).item()

    nmse = (
        torch.sum(torch.abs(x - x_hat) ** 2)
        / torch.sum(torch.abs(x) ** 2)
    ).item()

    correlation = (
        torch.abs(torch.sum(torch.conj(x) * x_hat))
        / (
            torch.sqrt(torch.sum(torch.abs(x) ** 2))
            * torch.sqrt(torch.sum(torch.abs(x_hat) ** 2))
        )
    ).item()

    print(f"MSE:         {mse:.6f}")
    print(f"NMSE:        {nmse:.6f}")
    print(f"NMSE (dB):   {10*np.log10(nmse):.3f} dB")
    print(f"Correlation: {correlation:.6f}")

    # -----------------------------
    # Plot constellation
    # -----------------------------
    x = x.cpu().numpy().flatten()
    x_hat = x_hat.cpu().numpy().flatten()

    plt.figure(figsize=(7, 7))

    plt.scatter(x.real, x.imag, s=5, alpha=0.5, label="Original")
    plt.scatter(x_hat.real, x_hat.imag, s=5, alpha=0.5, label="Reconstructed")

    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title("CVAE Complex Waveform Reconstruction")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--latent_dims", type=int, default=250)
    parser.add_argument("--model_dir", type=str, default=params_simple.cvae_model_dir)
    parser.add_argument("--data_path", type=str, default=params_simple.data_dir[0])
    
    args = parser.parse_args()
    main(args)
