#!/usr/bin/env python3
"""One-hot autoencoder: AWGN vs Rayleigh block fading with 4 pilots."""

# ============================================================
# Configuration and Imports
# ============================================================

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sionna.phy
from sionna.phy.channel import AWGN, RayleighBlockFading
from sionna.phy.mapping import BinarySource
from sionna.phy.utils import ebnodb2no, expand_to_rank, sim_ber

sionna.phy.config.seed = 42
device = sionna.phy.config.device


# ============================================================
# Simulation Parameters
# ============================================================

# Autoencoder
M = 16
k = int(np.log2(M))
n_channel = 7
num_pilots = 4
coderate_awgn = k / n_channel
coderate_rayleigh = k / (n_channel + num_pilots)

# SNR
train_ebno_min = 4.0
train_ebno_max = 8.0
eval_ebno_db = torch.arange(-4.0, 12.5, 0.5, device=device)

# Training
num_training_iterations = 10_000
training_batch_size = 256
learning_rate = 1e-3

# Evaluation
evaluation_batch_size = 512
max_mc_iter = 1_000
num_target_bit_errors = 2_000
num_target_block_errors = 1_000

# Output
result_dir = Path("results/ae_awgn_vs_rayleigh_4pilot")
result_dir.mkdir(parents=True, exist_ok=True)
awgn_weights_path = result_dir / "awgn_weights.pt"
rayleigh_weights_path = result_dir / "rayleigh_4pilot_weights.pt"
results_path = result_dir / "ber_bler.csv"
figure_path = result_dir / "ber_bler_comparison.png"


# ============================================================
# Bit / Message Conversion
# ============================================================

def bits_to_index(b):
    """[batch, k] bits -> [batch] message indices."""
    weights = 2 ** torch.arange(k - 1, -1, -1, device=b.device)
    return (b.long() * weights).sum(dim=-1)


def index_to_bits(s):
    """[batch] message indices -> [batch, k] bits."""
    shifts = torch.arange(k - 1, -1, -1, device=s.device)
    return ((s.unsqueeze(-1) >> shifts) & 1).float()


# ============================================================
# Neural Transmitter
# ============================================================

class NeuralTransmitter(nn.Module):
    """Map one-hot messages to normalized complex codewords."""

    def __init__(self):
        super().__init__()
        self._dense_1 = nn.Linear(M, M)
        self._dense_2 = nn.Linear(M, 2 * n_channel)

    def forward(self, s_onehot):
        z = F.relu(self._dense_1(s_onehot))
        z = self._dense_2(z)

        x_real, x_imag = torch.chunk(z, 2, dim=-1)
        x = torch.complex(x_real, x_imag)

        # mean(|x|^2) = 1 for every codeword
        energy = torch.mean(torch.abs(x) ** 2, dim=-1, keepdim=True)
        return x / torch.sqrt(energy + 1e-12)


# ============================================================
# Neural Receiver
# ============================================================

class NeuralReceiver(nn.Module):
    """Recover one of M messages from a received codeword."""

    def __init__(self):
        super().__init__()
        self._dense_1 = nn.Linear(2 * n_channel + 1, 128)
        self._dense_2 = nn.Linear(128, 128)
        self._dense_3 = nn.Linear(128, M)

    def forward(self, y, no):
        no_db = torch.log10(no.clamp_min(1e-12))
        z = torch.cat([y.real, y.imag, no_db], dim=-1)
        z = F.relu(self._dense_1(z))
        z = F.relu(self._dense_2(z))
        return self._dense_3(z)


# ============================================================
# Rayleigh Channel with 4 Pilots
# ============================================================

class Rayleigh4Pilot(nn.Module):
    """Sionna Rayleigh block fading + LS estimation + ZF."""

    def __init__(self):
        super().__init__()
        self._rayleigh = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=1,
        )
        self._awgn = AWGN()
        self.register_buffer(
            "_pilots",
            torch.ones(1, num_pilots, dtype=torch.complex64),
        )

    def forward(self, x, no):
        batch_size = x.shape[0]

        ################
        # Transmitter: 4 pilots + 7 learned symbols
        ################

        pilots = self._pilots.expand(batch_size, -1)
        x_block = torch.cat([pilots, x], dim=-1)

        ################
        # Channel: Sionna RayleighBlockFading + AWGN
        ################

        a, _ = self._rayleigh(
            batch_size=batch_size,
            num_time_steps=num_pilots + n_channel,
        )
        # [batch, rx, rx_ant, tx, tx_ant, path, time]
        h = a[:, 0, 0, 0, 0, 0, :]
        y = self._awgn(h * x_block, no)

        ################
        # Receiver: LS channel estimation
        ################

        y_pilot = y[:, :num_pilots]
        y_data = y[:, num_pilots:]

        # h_hat = sum(conj(p)y_p) / sum(|p|^2)
        h_hat = (
            (torch.conj(pilots) * y_pilot).sum(dim=-1, keepdim=True)
            / (torch.abs(pilots) ** 2).sum(dim=-1, keepdim=True)
        )

        ################
        # Receiver: regularized ZF equalization
        ################

        denominator = torch.abs(h_hat) ** 2 + 1e-6
        y_equalized = torch.conj(h_hat) * y_data / denominator
        no_equalized = no / denominator
        return y_equalized, no_equalized


# ============================================================
# End-to-End System
# ============================================================

class E2ESystem(nn.Module):
    """One-hot autoencoder for AWGN or Rayleigh + 4 pilots."""

    def __init__(self, channel_type, training):
        super().__init__()
        self._channel_type = channel_type
        self._training = training

        ################
        # Transmitter
        ################

        self._binary_source = BinarySource()
        self._transmitter = NeuralTransmitter()

        ################
        # Channel
        ################

        if channel_type == "awgn":
            self._channel = AWGN()
            self._coderate = coderate_awgn
        elif channel_type == "rayleigh":
            self._channel = Rayleigh4Pilot()
            self._coderate = coderate_rayleigh
        else:
            raise ValueError("channel_type must be 'awgn' or 'rayleigh'")

        ################
        # Receiver
        ################

        self._receiver = NeuralReceiver()

    def forward(self, batch_size, ebno_db):
        if ebno_db.dim() == 0:
            ebno_db = ebno_db.expand(batch_size)

        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=1,
            coderate=self._coderate,
        )
        no = expand_to_rank(no, 2)

        ################
        # Transmitter
        ################

        b = self._binary_source([batch_size, k])
        s = bits_to_index(b)
        s_onehot = F.one_hot(s, num_classes=M).float()
        x = self._transmitter(s_onehot)

        ################
        # Channel
        ################

        if self._channel_type == "awgn":
            y = self._channel(x, no)
            receiver_no = no
        else:
            y, receiver_no = self._channel(x, no)

        ################
        # Receiver
        ################

        logits = self._receiver(y, receiver_no)

        if self._training:
            log_prob = F.log_softmax(logits, dim=-1)
            return -(s_onehot * log_prob).sum(dim=-1).mean()

        s_hat = torch.argmax(logits, dim=-1)
        return b, index_to_bits(s_hat)


# ============================================================
# Training
# ============================================================

def train_model(channel_type, weights_path):
    model = E2ESystem(channel_type, training=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for i in range(num_training_iterations):
        optimizer.zero_grad()
        ebno_db = torch.empty(training_batch_size, device=device).uniform_(
            train_ebno_min,
            train_ebno_max,
        )
        loss = model(training_batch_size, ebno_db)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                f"{channel_type:8s} iteration {i:5d}/"
                f"{num_training_iterations} loss: {loss.item():.4f}"
            )

    torch.save(model.state_dict(), weights_path)


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(channel_type, weights_path):
    model = E2ESystem(channel_type, training=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with torch.no_grad():
        ber, bler = sim_ber(
            model,
            eval_ebno_db,
            batch_size=evaluation_batch_size,
            max_mc_iter=max_mc_iter,
            num_target_bit_errors=num_target_bit_errors,
            num_target_block_errors=num_target_block_errors,
            early_stop=False,
        )

    return ber.cpu().numpy(), bler.cpu().numpy()


# ============================================================
# Run Experiment
# ============================================================

print("Training AWGN autoencoder")
train_model("awgn", awgn_weights_path)

print("\nTraining Rayleigh + 4-pilot autoencoder")
train_model("rayleigh", rayleigh_weights_path)

print("\nEvaluating AWGN")
awgn_ber, awgn_bler = evaluate_model("awgn", awgn_weights_path)

print("\nEvaluating Rayleigh + 4 pilots")
rayleigh_ber, rayleigh_bler = evaluate_model(
    "rayleigh",
    rayleigh_weights_path,
)


# ============================================================
# Save and Plot Results
# ============================================================

ebno_db_numpy = eval_ebno_db.cpu().numpy()
results = np.column_stack(
    [ebno_db_numpy, awgn_ber, awgn_bler, rayleigh_ber, rayleigh_bler]
)

np.savetxt(
    results_path,
    results,
    delimiter=",",
    header=(
        "ebno_db,awgn_ber,awgn_bler,"
        "rayleigh_4pilot_ber,rayleigh_4pilot_bler"
    ),
    comments="",
)

plt.figure(figsize=(8, 5.5))
plt.semilogy(ebno_db_numpy, awgn_ber, marker="o", label="AWGN BER")
plt.semilogy(ebno_db_numpy, awgn_bler, linestyle="--", label="AWGN BLER")
plt.semilogy(
    ebno_db_numpy,
    rayleigh_ber,
    marker="s",
    label="Rayleigh + 4 pilots BER",
)
plt.semilogy(
    ebno_db_numpy,
    rayleigh_bler,
    linestyle="--",
    label="Rayleigh + 4 pilots BLER",
)
plt.xlabel("Eb/N0 [dB]")
plt.ylabel("Error Rate")
plt.title("One-Hot Autoencoder: AWGN vs Rayleigh + 4 Pilots")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(figure_path, dpi=200)
plt.close()

print("\nFinished")
print(f"Results: {results_path}")
print(f"Figure:  {figure_path}")
