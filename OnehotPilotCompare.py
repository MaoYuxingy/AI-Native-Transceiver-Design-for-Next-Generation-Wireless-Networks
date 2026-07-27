#!/usr/bin/env python3
"""Compare Rayleigh perfect CSI with 2, 4, and 8 pilot symbols.

Outputs:
- ber_comparison.csv
- efficiency_comparison.csv
- rayleigh_ber_comparison.png
- rayleigh_efficiency_comparison.png
- one weight file per configuration
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sionna.phy
from sionna.phy.channel import AWGN, RayleighBlockFading
from sionna.phy.mapping import BinarySource
from sionna.phy.utils import ebnodb2no, expand_to_rank, sim_ber


# ============================================================
# Reproducibility and Device
# ============================================================

RANDOM_SEED = 42


def reset_random_seed(seed: int) -> None:
    """Reset NumPy, PyTorch, CUDA, and Sionna random generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sionna.phy.config.seed = seed


reset_random_seed(RANDOM_SEED)
device = sionna.phy.config.device


# ============================================================
# Simulation Parameters
# ============================================================

# One-hot autoencoder
M = 16
K = int(np.log2(M))
N_DATA_SYMBOLS = 7
HIDDEN_DIM = 128

# The four Rayleigh receiver configurations.
# Perfect CSI uses no pilots because the true h is directly available.
CONFIGURATIONS: Dict[str, Dict[str, object]] = {
    "perfect_csi": {
        "label": "Perfect CSI",
        "num_pilots": 0,
        "perfect_csi": True,
    },
    "pilot_2": {
        "label": "2 pilots",
        "num_pilots": 2,
        "perfect_csi": False,
    },
    "pilot_4": {
        "label": "4 pilots",
        "num_pilots": 4,
        "perfect_csi": False,
    },
    "pilot_8": {
        "label": "8 pilots",
        "num_pilots": 8,
        "perfect_csi": False,
    },
}

# Fixed reference coderate used by every BER model.
# This deliberately excludes pilot overhead so that BER differences
# mainly reflect channel-estimation quality rather than a changing N0.
COMPARISON_CODERATE = K / N_DATA_SYMBOLS

# Training Eb/N0 range
TRAIN_EBNO_MIN_DB = 4.0
TRAIN_EBNO_MAX_DB = 8.0

# Evaluation Eb/N0 range
EVAL_EBNO_DB = torch.arange(-4.0, 12.5, 0.5, device=device)

# Training
NUM_TRAINING_ITERATIONS = 10_000
TRAINING_BATCH_SIZE = 256
LEARNING_RATE = 1e-3

# Evaluation
EVALUATION_BATCH_SIZE = 512
MAX_MC_ITER = 1_000
NUM_TARGET_BIT_ERRORS = 2_000
NUM_TARGET_BLOCK_ERRORS = 1_000

# Numerical regularization used by the ZF equalizer
ZF_EPSILON = 1e-6

# Output
RESULT_DIR = Path(
    "results/rayleigh_fixed_coderate_perfect_csi_vs_2_4_8_pilots"
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BER_RESULTS_PATH = RESULT_DIR / "ber_comparison.csv"
EFFICIENCY_RESULTS_PATH = RESULT_DIR / "efficiency_comparison.csv"
BER_FIGURE_PATH = RESULT_DIR / "rayleigh_ber_comparison.png"
EFFICIENCY_FIGURE_PATH = RESULT_DIR / "rayleigh_efficiency_comparison.png"
PARAMETERS_PATH = RESULT_DIR / "parameters.txt"


# ============================================================
# Helper Functions
# ============================================================


def bits_to_index(bits: torch.Tensor) -> torch.Tensor:
    """Convert [batch, K] bits to [batch] message indices."""
    weights = 2 ** torch.arange(K - 1, -1, -1, device=bits.device)
    return (bits.long() * weights).sum(dim=-1)



def index_to_bits(indices: torch.Tensor) -> torch.Tensor:
    """Convert [batch] message indices to [batch, K] bits."""
    shifts = torch.arange(K - 1, -1, -1, device=indices.device)
    return ((indices.unsqueeze(-1) >> shifts) & 1).float()



def effective_information_rate(num_pilots: int) -> float:
    """Information bits per transmitted complex channel use."""
    return K / (N_DATA_SYMBOLS + num_pilots)



def pilot_payload_efficiency(num_pilots: int) -> float:
    """Fraction of transmitted symbols that carry learned data."""
    return N_DATA_SYMBOLS / (N_DATA_SYMBOLS + num_pilots)


# ============================================================
# Neural Transmitter
# ============================================================


class NeuralTransmitter(nn.Module):
    """Map one-hot messages to normalized complex codewords."""

    def __init__(self) -> None:
        super().__init__()
        self._dense_1 = nn.Linear(M, M)
        self._dense_2 = nn.Linear(M, 2 * N_DATA_SYMBOLS)

    def forward(self, one_hot_messages: torch.Tensor) -> torch.Tensor:
        z = F.relu(self._dense_1(one_hot_messages))
        z = self._dense_2(z)

        # 14 real outputs -> 7 real parts + 7 imaginary parts
        x_real, x_imag = torch.chunk(z, 2, dim=-1)
        x = torch.complex(x_real, x_imag)

        # Normalize every learned codeword so that mean(|x|^2) = 1.
        energy = torch.mean(torch.abs(x) ** 2, dim=-1, keepdim=True)
        return x / torch.sqrt(energy + 1e-12)


# ============================================================
# Neural Receiver
# ============================================================


class NeuralReceiver(nn.Module):
    """Jointly classify one complete received codeword."""

    def __init__(self) -> None:
        super().__init__()
        input_dim = 2 * N_DATA_SYMBOLS + 1
        self._dense_1 = nn.Linear(input_dim, HIDDEN_DIM)
        self._dense_2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self._dense_3 = nn.Linear(HIDDEN_DIM, M)

    def forward(
        self,
        y_equalized: torch.Tensor,
        receiver_no: torch.Tensor,
    ) -> torch.Tensor:
        # receiver_no is the post-equalization effective noise variance,
        # not the actual random noise realization.
        no_feature = torch.log10(receiver_no.clamp_min(1e-12))

        # 7 complex values -> 14 real values, then append one noise feature.
        z = torch.cat(
            [y_equalized.real, y_equalized.imag, no_feature],
            dim=-1,
        )
        z = F.relu(self._dense_1(z))
        z = F.relu(self._dense_2(z))
        return self._dense_3(z)


# ============================================================
# Rayleigh Block-Fading Channel
# ============================================================


class RayleighCSIChannel(nn.Module):
    """Rayleigh block fading with perfect CSI or pilot-based LS CSI.

    For perfect CSI, no pilots are transmitted and the true channel
    coefficient h is used by the ZF equalizer.

    For pilot-based CSI, unit pilots are prepended, one LS estimate h_hat
    is formed for the whole block, and regularized ZF is applied.
    """

    def __init__(self, num_pilots: int, perfect_csi: bool) -> None:
        super().__init__()

        if perfect_csi and num_pilots != 0:
            raise ValueError("Perfect CSI must use num_pilots=0.")
        if not perfect_csi and num_pilots <= 0:
            raise ValueError("Pilot-based CSI requires at least one pilot.")

        self._num_pilots = num_pilots
        self._perfect_csi = perfect_csi

        self._rayleigh = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=1,
        )
        self._awgn = AWGN()

        # Unit-power pilots. For perfect CSI this is an empty buffer.
        self.register_buffer(
            "_pilots",
            torch.ones(1, num_pilots, dtype=torch.complex64),
        )

    def _generate_block_channel(self, batch_size: int) -> torch.Tensor:
        """Generate one complex Rayleigh coefficient per codeword."""
        a, _ = self._rayleigh(
            batch_size=batch_size,
            num_time_steps=1,
        )
        # Sionna shape:
        # [batch, rx, rx_ant, tx, tx_ant, path, time]
        return a[:, 0, 0, 0, 0, 0, :]  # [batch, 1]

    @staticmethod
    def _regularized_zf(
        y_data: torch.Tensor,
        h_used: torch.Tensor,
        no: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Equalize data and compute the post-equalization noise variance."""
        denominator = torch.abs(h_used) ** 2 + ZF_EPSILON
        y_equalized = torch.conj(h_used) * y_data / denominator
        no_equalized = no / denominator
        return y_equalized, no_equalized

    def forward(
        self,
        x_data: torch.Tensor,
        no: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_data.shape[0]

        # One constant fading coefficient for pilots and data in this codeword.
        h = self._generate_block_channel(batch_size)

        if self._perfect_csi:
            # No pilot overhead. The receiver directly uses the true h.
            y_data = self._awgn(h * x_data, no)
            return self._regularized_zf(y_data, h, no)

        # Pilot-assisted case: prepend P unit pilots to 7 learned data symbols.
        pilots = self._pilots.expand(batch_size, -1)
        x_block = torch.cat([pilots, x_data], dim=-1)
        y_block = self._awgn(h * x_block, no)

        y_pilot = y_block[:, : self._num_pilots]
        y_data = y_block[:, self._num_pilots :]

        # Least-squares estimate for a constant block-fading coefficient:
        # h_hat = sum(conj(p) * y_p) / sum(|p|^2)
        h_hat = (
            (torch.conj(pilots) * y_pilot).sum(dim=-1, keepdim=True)
            / (torch.abs(pilots) ** 2).sum(dim=-1, keepdim=True)
        )

        return self._regularized_zf(y_data, h_hat, no)


# ============================================================
# End-to-End System
# ============================================================


class E2ESystem(nn.Module):
    """One-hot autoencoder over one selected Rayleigh CSI configuration."""

    def __init__(self, config_name: str, training: bool) -> None:
        super().__init__()

        if config_name not in CONFIGURATIONS:
            raise ValueError(
                f"Unknown configuration '{config_name}'. "
                f"Choose from {list(CONFIGURATIONS)}."
            )

        config = CONFIGURATIONS[config_name]
        self._training_mode = training
        self._binary_source = BinarySource()
        self._transmitter = NeuralTransmitter()
        self._channel = RayleighCSIChannel(
            num_pilots=int(config["num_pilots"]),
            perfect_csi=bool(config["perfect_csi"]),
        )
        self._receiver = NeuralReceiver()

        # Use the same coderate for all configurations so that the same
        # reference Eb/N0 produces the same channel-noise variance.
        # Actual pilot overhead is evaluated separately as efficiency.
        self._coderate = COMPARISON_CODERATE

    def forward(
        self,
        batch_size: int,
        ebno_db: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if ebno_db.dim() == 0:
            ebno_db = ebno_db.expand(batch_size)

        # All configurations use the same reference coderate K/7 here.
        # Thus 2/4/8 pilots receive the same N0 at the same Eb/N0 point.
        # Pilot overhead is intentionally excluded from this BER comparison.
        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=1,
            coderate=self._coderate,
        )
        no = expand_to_rank(no, 2)  # [batch, 1]

        # Generate K random bits and convert the message to one-hot form.
        bits = self._binary_source([batch_size, K])
        message_indices = bits_to_index(bits)
        one_hot_messages = F.one_hot(
            message_indices,
            num_classes=M,
        ).float()

        # Transmitter -> channel/equalizer -> neural receiver.
        x_data = self._transmitter(one_hot_messages)
        y_equalized, receiver_no = self._channel(x_data, no)
        logits = self._receiver(y_equalized, receiver_no)

        if self._training_mode:
            # Same categorical cross-entropy expression as the compact script.
            log_prob = F.log_softmax(logits, dim=-1)
            return -(one_hot_messages * log_prob).sum(dim=-1).mean()

        estimated_indices = torch.argmax(logits, dim=-1)
        estimated_bits = index_to_bits(estimated_indices)
        return bits, estimated_bits


# ============================================================
# Training and Evaluation
# ============================================================



def weights_path(config_name: str) -> Path:
    return RESULT_DIR / f"{config_name}_weights.pt"



def train_model(config_name: str) -> None:
    """Train one independent model for one CSI/pilot configuration."""
    # Starting every configuration from the same seed gives comparable
    # initialization and random-data sequences.
    reset_random_seed(RANDOM_SEED)

    config = CONFIGURATIONS[config_name]
    model = E2ESystem(config_name, training=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    print("-" * 72)
    print(
        f"Training {config['label']} | "
        f"pilots={config['num_pilots']} | "
        f"BER coderate={COMPARISON_CODERATE:.6f} | "
        f"actual efficiency={effective_information_rate(int(config['num_pilots'])):.6f}"
    )

    for iteration in range(NUM_TRAINING_ITERATIONS):
        optimizer.zero_grad()

        ebno_db = torch.empty(
            TRAINING_BATCH_SIZE,
            device=device,
        ).uniform_(TRAIN_EBNO_MIN_DB, TRAIN_EBNO_MAX_DB)

        loss = model(TRAINING_BATCH_SIZE, ebno_db)
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(
                f"{config_name:12s} iteration "
                f"{iteration:5d}/{NUM_TRAINING_ITERATIONS} "
                f"loss={loss.item():.6f}"
            )

    torch.save(model.state_dict(), weights_path(config_name))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def evaluate_model(config_name: str) -> np.ndarray:
    """Evaluate BER for one trained configuration."""
    config = CONFIGURATIONS[config_name]
    model = E2ESystem(config_name, training=False).to(device)
    model.load_state_dict(
        torch.load(weights_path(config_name), map_location=device)
    )
    model.eval()

    print("-" * 72)
    print(f"Evaluating {config['label']}")

    with torch.no_grad():
        ber, _ = sim_ber(
            model,
            EVAL_EBNO_DB,
            batch_size=EVALUATION_BATCH_SIZE,
            max_mc_iter=MAX_MC_ITER,
            num_target_bit_errors=NUM_TARGET_BIT_ERRORS,
            num_target_block_errors=NUM_TARGET_BLOCK_ERRORS,
            early_stop=False,
        )

    ber_numpy = ber.detach().cpu().numpy()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ber_numpy


# ============================================================
# Saving and Plotting
# ============================================================



def save_parameters() -> None:
    lines = [
        f"random_seed={RANDOM_SEED}",
        f"M={M}",
        f"K={K}",
        f"N_DATA_SYMBOLS={N_DATA_SYMBOLS}",
        f"HIDDEN_DIM={HIDDEN_DIM}",
        f"TRAIN_EBNO_MIN_DB={TRAIN_EBNO_MIN_DB}",
        f"TRAIN_EBNO_MAX_DB={TRAIN_EBNO_MAX_DB}",
        f"EVAL_EBNO_DB={EVAL_EBNO_DB.detach().cpu().tolist()}",
        f"NUM_TRAINING_ITERATIONS={NUM_TRAINING_ITERATIONS}",
        f"TRAINING_BATCH_SIZE={TRAINING_BATCH_SIZE}",
        f"LEARNING_RATE={LEARNING_RATE}",
        f"EVALUATION_BATCH_SIZE={EVALUATION_BATCH_SIZE}",
        f"MAX_MC_ITER={MAX_MC_ITER}",
        f"NUM_TARGET_BIT_ERRORS={NUM_TARGET_BIT_ERRORS}",
        f"NUM_TARGET_BLOCK_ERRORS={NUM_TARGET_BLOCK_ERRORS}",
        f"ZF_EPSILON={ZF_EPSILON}",
        f"COMPARISON_CODERATE={COMPARISON_CODERATE}",
        "BER comparison excludes pilot overhead from Eb/N0-to-N0 conversion.",
        f"device={device}",
        "",
        "Configuration summary:",
    ]

    for config_name, config in CONFIGURATIONS.items():
        p = int(config["num_pilots"])
        lines.append(
            f"{config_name}: label={config['label']}, pilots={p}, "
            f"perfect_csi={config['perfect_csi']}, "
            f"ber_coderate={COMPARISON_CODERATE:.8f}, "
            f"actual_effective_rate={effective_information_rate(p):.8f}, "
            f"payload_efficiency={pilot_payload_efficiency(p):.8f}"
        )

    PARAMETERS_PATH.write_text("\n".join(lines), encoding="utf-8")



def save_ber_results(ber_results: Dict[str, np.ndarray]) -> None:
    ebno_numpy = EVAL_EBNO_DB.detach().cpu().numpy()
    columns = [ebno_numpy]
    header = ["ebno_db"]

    for config_name in CONFIGURATIONS:
        columns.append(ber_results[config_name])
        header.append(f"{config_name}_ber")

    np.savetxt(
        BER_RESULTS_PATH,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
    )



def save_efficiency_results() -> None:
    rows = []
    for config_name, config in CONFIGURATIONS.items():
        num_pilots = int(config["num_pilots"])
        total_symbols = N_DATA_SYMBOLS + num_pilots
        rows.append(
            [
                config_name,
                str(config["label"]),
                num_pilots,
                N_DATA_SYMBOLS,
                total_symbols,
                effective_information_rate(num_pilots),
                pilot_payload_efficiency(num_pilots),
                num_pilots / total_symbols,
            ]
        )

    with EFFICIENCY_RESULTS_PATH.open("w", encoding="utf-8") as file:
        file.write(
            "config,label,num_pilots,data_symbols,total_symbols,"
            "effective_information_rate,payload_efficiency,pilot_overhead\n"
        )
        for row in rows:
            file.write(",".join(map(str, row)) + "\n")



def plot_ber(ber_results: Dict[str, np.ndarray]) -> None:
    ebno_numpy = EVAL_EBNO_DB.detach().cpu().numpy()

    plt.figure(figsize=(8.5, 5.8))
    markers = ["o", "s", "^", "d"]

    for marker, (config_name, config) in zip(
        markers,
        CONFIGURATIONS.items(),
    ):
        plt.semilogy(
            ebno_numpy,
            ber_results[config_name],
            marker=marker,
            markevery=2,
            label=str(config["label"]),
        )

    plt.xlabel("Reference Eb/N0 [dB] (pilot overhead excluded)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(
        "Rayleigh Autoencoder: Fixed-Noise Pilot Estimation Comparison"
    )
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BER_FIGURE_PATH, dpi=200)
    plt.close()



def plot_efficiency() -> None:
    labels = [str(config["label"]) for config in CONFIGURATIONS.values()]
    efficiencies = [
        effective_information_rate(int(config["num_pilots"]))
        for config in CONFIGURATIONS.values()
    ]

    plt.figure(figsize=(8.0, 5.5))
    bars = plt.bar(labels, efficiencies)

    for bar, value in zip(bars, efficiencies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel("Effective information rate [bits / complex channel use]")
    plt.title("Transmission Efficiency Including Pilot Overhead")
    plt.ylim(0.0, max(efficiencies) * 1.20)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(EFFICIENCY_FIGURE_PATH, dpi=200)
    plt.close()


# ============================================================
# Main Experiment
# ============================================================



def main() -> None:
    print("=" * 72)
    print("Rayleigh Fixed-Coderate Perfect CSI vs 2/4/8 Pilots")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Result directory: {RESULT_DIR.resolve()}")
    print(f"Data symbols per message: {N_DATA_SYMBOLS}")
    print(f"Information bits per message: {K}")
    print(f"Shared BER coderate: {COMPARISON_CODERATE:.6f}")
    print("Pilot overhead is excluded from BER noise conversion.")

    for config_name, config in CONFIGURATIONS.items():
        p = int(config["num_pilots"])
        print(
            f"{str(config['label']):12s}: pilots={p}, "
            f"total uses={N_DATA_SYMBOLS + p}, "
            f"BER coderate={COMPARISON_CODERATE:.6f}, "
            f"actual efficiency={effective_information_rate(p):.6f}"
        )

    save_parameters()

    # Each configuration is trained independently.
    for config_name in CONFIGURATIONS:
        train_model(config_name)

    ber_results: Dict[str, np.ndarray] = {}
    for config_name in CONFIGURATIONS:
        ber_results[config_name] = evaluate_model(config_name)

    save_ber_results(ber_results)
    save_efficiency_results()
    plot_ber(ber_results)
    plot_efficiency()

    print("=" * 72)
    print("Finished")
    print(f"BER CSV:        {BER_RESULTS_PATH}")
    print(f"Efficiency CSV: {EFFICIENCY_RESULTS_PATH}")
    print(f"BER figure:     {BER_FIGURE_PATH}")
    print(f"Efficiency fig: {EFFICIENCY_FIGURE_PATH}")
    print(f"Parameters:     {PARAMETERS_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()