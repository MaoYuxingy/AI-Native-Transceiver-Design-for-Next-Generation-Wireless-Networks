#!/usr/bin/env python3
"""Compare three training Eb/N0 ranges for a pure trainable constellation model.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sionna.phy
from sionna.phy.channel import AWGN, RayleighBlockFading
from sionna.phy.mapping import BinarySource, Constellation, Mapper
from sionna.phy.utils import ebnodb2no, expand_to_rank


# ============================================================
# Reproducibility and device
# ============================================================

RANDOM_SEED = 42
EVALUATION_SEED = 20_000


def reset_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sionna.phy.config.seed = seed


reset_random_seed(RANDOM_SEED)
device = sionna.phy.config.device


# ============================================================
# Communication parameters
# ============================================================

NUM_BITS_PER_SYMBOL = 6
MODULATION_ORDER = 2**NUM_BITS_PER_SYMBOL  # 64
NUM_BITS_PER_FRAME = 1500
NUM_DATA_SYMBOLS = NUM_BITS_PER_FRAME // NUM_BITS_PER_SYMBOL  # 250
NUM_PILOTS = 4

if NUM_BITS_PER_FRAME % NUM_BITS_PER_SYMBOL != 0:
    raise ValueError("NUM_BITS_PER_FRAME must be divisible by NUM_BITS_PER_SYMBOL.")

# No LDPC is used here.
OUTER_CODERATE = 1.0

# Pilot overhead is included when converting Eb/N0 to N0.
PILOT_EFFICIENCY = NUM_DATA_SYMBOLS / (NUM_DATA_SYMBOLS + NUM_PILOTS)
EFFECTIVE_CODERATE = OUTER_CODERATE * PILOT_EFFICIENCY
EFFECTIVE_INFORMATION_RATE = (
    NUM_BITS_PER_FRAME / (NUM_DATA_SYMBOLS + NUM_PILOTS)
)

HIDDEN_DIM = 128
ZF_EPSILON = 1e-6


# ============================================================
# Training and evaluation configurations
# ============================================================

MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "mapping_uniform_0_4": {
        "label": "Trainable mapping trained on Uniform(0, 4) dB",
        "train_min_db": 0.0,
        "train_max_db": 4.0,
        "marker": "o",
    },
    "mapping_uniform_4_8": {
        "label": "Trainable mapping trained on Uniform(4, 8) dB",
        "train_min_db": 4.0,
        "train_max_db": 8.0,
        "marker": "s",
    },
    "mapping_uniform_8_12": {
        "label": "Trainable mapping trained on Uniform(8, 12) dB",
        "train_min_db": 8.0,
        "train_max_db": 12.0,
        "marker": "^",
    },
}

NUM_TRAINING_ITERATIONS = 10_000
TRAINING_BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LOG_INTERVAL = 100

EVAL_EBNO_DB = torch.arange(0.0, 20.5, 0.5, device=device)
EVALUATION_BATCH_SIZE = 256
MIN_MC_ITER = 20
MAX_MC_ITER = 1_000
NUM_TARGET_BIT_ERRORS = 10_000

# Set TRAIN_MODELS=False after the weights have been generated if only testing.
TRAIN_MODELS = True
EVALUATE_MODELS = True


# ============================================================
# Output paths
# ============================================================

RESULT_DIR = Path("results/rayleigh_trainable_mapping_training_snr_ranges")
WEIGHTS_DIR = RESULT_DIR / "weights"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_LOG_PATH = RESULT_DIR / "training_loss.csv"
EVALUATION_RESULTS_PATH = RESULT_DIR / "evaluation_results.csv"
PARAMETERS_PATH = RESULT_DIR / "parameters.txt"

BER_FIGURE_PATH = RESULT_DIR / "ber_comparison.png"
SER_FIGURE_PATH = RESULT_DIR / "ser_comparison.png"
BLER_FIGURE_PATH = RESULT_DIR / "bler_comparison.png"
TRAINING_LOSS_FIGURE_PATH = RESULT_DIR / "training_loss.png"
CONSTELLATION_COMPARISON_PATH = RESULT_DIR / "constellation_comparison.png"


# ============================================================
# Bit conversion utilities
# ============================================================


def group_bits(bits: torch.Tensor) -> torch.Tensor:
    """[batch, 1500] -> [batch, 250, 6]."""
    return bits.reshape(bits.shape[0], NUM_DATA_SYMBOLS, NUM_BITS_PER_SYMBOL)


def grouped_bits_to_indices(grouped_bits: torch.Tensor) -> torch.Tensor:
    """[batch, 250, 6] -> [batch, 250] symbol indices 0,...,63."""
    weights = 2 ** torch.arange(
        NUM_BITS_PER_SYMBOL - 1,
        -1,
        -1,
        device=grouped_bits.device,
    )
    return (grouped_bits.long() * weights).sum(dim=-1)


# ============================================================
# Trainable constellation mapper
# ============================================================


def standard_qam_points() -> torch.Tensor:
    """Return normalized 64-QAM points for initialization."""
    constellation = Constellation("qam", NUM_BITS_PER_SYMBOL)
    return constellation.points.detach().clone().to(torch.complex64)


def center_and_normalize_points(points: torch.Tensor) -> torch.Tensor:
    """Zero-center points and force unit average symbol energy."""
    centered = points - torch.mean(points)
    average_energy = torch.mean(torch.abs(centered) ** 2)
    return centered / torch.sqrt(average_energy + 1e-12)


class TrainableConstellationMapper(nn.Module):
    """Map every 6 input bits directly to one trainable complex point."""

    def __init__(self) -> None:
        super().__init__()

        initial_points = standard_qam_points()
        self.points_r = nn.Parameter(initial_points.real.clone())
        self.points_i = nn.Parameter(initial_points.imag.clone())

        self.constellation = Constellation(
            "custom",
            NUM_BITS_PER_SYMBOL,
            points=torch.complex(self.points_r, self.points_i),
            normalize=True,
            center=True,
        )
        self._mapper = Mapper(constellation=self.constellation)

    def constellation_points(self) -> torch.Tensor:
        raw_points = torch.complex(self.points_r, self.points_i)
        return center_and_normalize_points(raw_points)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        # Refresh the constellation with the latest trainable parameters so the
        # current forward pass remains connected to the autograd graph.
        self.constellation.points = torch.complex(self.points_r, self.points_i)

        # [batch, 1500] bits -> [batch, 250] complex symbols
        return self._mapper(bits)


# ============================================================
# Neural bit-wise demapper
# ============================================================


def build_receiver_features(
    y_equalized: torch.Tensor,
    no_equalized: torch.Tensor,
) -> torch.Tensor:
    """Create [Re(y_eq), Im(y_eq), log10(N0_eq)] for every data symbol."""
    no_feature = torch.log10(no_equalized.clamp_min(1e-12))
    no_feature = no_feature.expand(-1, NUM_DATA_SYMBOLS)

    return torch.stack(
        [y_equalized.real, y_equalized.imag, no_feature],
        dim=-1,
    )  # [batch, 250, 3]


class NeuralDemapper(nn.Module):
    """Output six bit logits for each received complex data symbol."""

    def __init__(self) -> None:
        super().__init__()
        self._dense_1 = nn.Linear(3, HIDDEN_DIM)
        self._dense_2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self._dense_3 = nn.Linear(HIDDEN_DIM, NUM_BITS_PER_SYMBOL)

    def forward(
        self,
        y_equalized: torch.Tensor,
        no_equalized: torch.Tensor,
    ) -> torch.Tensor:
        z = build_receiver_features(y_equalized, no_equalized)
        z = F.relu(self._dense_1(z))
        z = F.relu(self._dense_2(z))
        return self._dense_3(z)  # [batch, 250, 6]


# ============================================================
# Rayleigh block-fading channel with four pilots
# ============================================================


class Rayleigh4PilotChannel(nn.Module):
    """Rayleigh block fading + LS estimation + regularized ZF."""

    def __init__(self) -> None:
        super().__init__()

        self._rayleigh = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=1,
        )
        self._awgn = AWGN()

        # Four known unit-power pilots: 1+0j.
        self.register_buffer(
            "_pilots",
            torch.ones(1, NUM_PILOTS, dtype=torch.complex64),
        )

    def _generate_block_channel(self, batch_size: int) -> torch.Tensor:
        """Generate one complex h for every frame: [batch, 1]."""
        a, _ = self._rayleigh(
            batch_size=batch_size,
            num_time_steps=1,
        )

        # Sionna output shape:
        # [batch, rx, rx_ant, tx, tx_ant, path, time]
        return a[:, 0, 0, 0, 0, 0, :]  # [batch, 1]

    @staticmethod
    def _regularized_zf(
        y_data: torch.Tensor,
        h_hat: torch.Tensor,
        no: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        denominator = torch.abs(h_hat) ** 2 + ZF_EPSILON
        equalizer_gain = torch.conj(h_hat) / denominator

        y_equalized = equalizer_gain * y_data
        no_equalized = no * torch.abs(equalizer_gain) ** 2
        return y_equalized, no_equalized

    def forward(
        self,
        x_data: torch.Tensor,
        no: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_data.shape[0]

        pilots = self._pilots.expand(batch_size, -1)
        x_frame = torch.cat([pilots, x_data], dim=-1)

        # One h is shared by all pilots and all data symbols in a frame.
        h = self._generate_block_channel(batch_size)
        y_frame = self._awgn(h * x_frame, no)

        y_pilot = y_frame[:, :NUM_PILOTS]
        y_data = y_frame[:, NUM_PILOTS:]

        # LS estimate: h_hat = sum(conj(p_i)y_i) / sum(|p_i|^2)
        h_hat = (
            (torch.conj(pilots) * y_pilot).sum(dim=-1, keepdim=True)
            / (torch.abs(pilots) ** 2).sum(dim=-1, keepdim=True)
        )

        return self._regularized_zf(y_data, h_hat, no)


# ============================================================
# Complete end-to-end system
# ============================================================


class TrainableMappingE2ESystem(nn.Module):
    """Trainable 64-point mapper followed by a six-output neural demapper."""

    def __init__(self, training: bool) -> None:
        super().__init__()
        self._training_mode = training

        self._binary_source = BinarySource()
        self._transmitter = TrainableConstellationMapper()
        self._channel = Rayleigh4PilotChannel()
        self._receiver = NeuralDemapper()

    def constellation_points(self) -> torch.Tensor:
        return self._transmitter.constellation_points()

    def forward(
        self,
        batch_size: int,
        ebno_db: torch.Tensor,
    ) -> torch.Tensor | Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if ebno_db.dim() == 0:
            ebno_db = ebno_db.expand(batch_size)

        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
            coderate=EFFECTIVE_CODERATE,
        )
        no = expand_to_rank(no, 2)  # [batch, 1]

        bits = self._binary_source([batch_size, NUM_BITS_PER_FRAME])
        grouped_bits = group_bits(bits)
        true_indices = grouped_bits_to_indices(grouped_bits)

        # The transmitter receives bits only. It does not receive pilots, h,
        # h_hat, or any other CSI.
        x_data = self._transmitter(bits)

        y_equalized, no_equalized = self._channel(x_data, no)
        bit_logits = self._receiver(y_equalized, no_equalized)

        if self._training_mode:
            return F.binary_cross_entropy_with_logits(
                bit_logits,
                grouped_bits,
            )

        estimated_grouped_bits = (bit_logits > 0.0).float()
        estimated_bits = estimated_grouped_bits.reshape(
            batch_size,
            NUM_BITS_PER_FRAME,
        )
        estimated_indices = grouped_bits_to_indices(estimated_grouped_bits)

        return bits, estimated_bits, true_indices, estimated_indices


# ============================================================
# Model construction
# ============================================================


def build_model(training: bool) -> TrainableMappingE2ESystem:
    return TrainableMappingE2ESystem(training=training)


def weights_path(model_name: str) -> Path:
    return WEIGHTS_DIR / f"{model_name}_weights.pt"


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# Training
# ============================================================


def train_model(model_name: str) -> List[Dict[str, object]]:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_CONFIGS[model_name]
    train_min_db = float(config["train_min_db"])
    train_max_db = float(config["train_max_db"])

    # Reset before every model so architecture initialization and random streams
    # are aligned as closely as possible. Only the Eb/N0 interval changes.
    reset_random_seed(RANDOM_SEED)

    model = build_model(training=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    records: List[Dict[str, object]] = []

    print("-" * 100)
    print(f"Training {config['label']}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    for iteration in range(NUM_TRAINING_ITERATIONS):
        optimizer.zero_grad()

        # Every sample in a batch receives an independently sampled Eb/N0.
        ebno_db = torch.empty(
            TRAINING_BATCH_SIZE,
            device=device,
        ).uniform_(train_min_db, train_max_db)

        loss = model(TRAINING_BATCH_SIZE, ebno_db)
        loss.backward()
        optimizer.step()

        if iteration % LOG_INTERVAL == 0 or iteration == NUM_TRAINING_ITERATIONS - 1:
            record = {
                "model": model_name,
                "label": str(config["label"]),
                "iteration": iteration,
                "configured_train_min_db": train_min_db,
                "configured_train_max_db": train_max_db,
                "sampled_mean_ebno_db": float(ebno_db.mean().item()),
                "sampled_min_ebno_db": float(ebno_db.min().item()),
                "sampled_max_ebno_db": float(ebno_db.max().item()),
                "loss": float(loss.item()),
            }
            records.append(record)

            print(
                f"{model_name:24s} | "
                f"iteration {iteration:5d}/{NUM_TRAINING_ITERATIONS} | "
                f"mean Eb/N0={record['sampled_mean_ebno_db']:5.2f} dB | "
                f"loss={record['loss']:.6f}"
            )

    torch.save(model.state_dict(), weights_path(model_name))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return records


# ============================================================
# Evaluation
# ============================================================


def evaluate_model(model_name: str) -> Dict[str, np.ndarray]:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    weight_file = weights_path(model_name)
    if not weight_file.exists():
        raise FileNotFoundError(
            f"Missing weights: {weight_file}. "
            "Set TRAIN_MODELS=True or provide the trained file."
        )

    config = MODEL_CONFIGS[model_name]

    model = build_model(training=False).to(device)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.eval()

    results: Dict[str, List[float | int]] = {
        "ebno_db": [],
        "bit_errors": [],
        "total_bits": [],
        "ber": [],
        "symbol_errors": [],
        "total_symbols": [],
        "ser": [],
        "block_errors": [],
        "total_blocks": [],
        "bler": [],
        "mc_iterations": [],
    }

    print("-" * 100)
    print(f"Evaluating {config['label']}")

    with torch.inference_mode():
        for snr_index, ebno_scalar in enumerate(EVAL_EBNO_DB):
            ebno_value = float(ebno_scalar.item())

            # The same seed at the same SNR is used for every model, so all
            # models see the same bits, Rayleigh h values, and Gaussian samples.
            reset_random_seed(EVALUATION_SEED + snr_index)

            bit_errors = 0
            total_bits = 0
            symbol_errors = 0
            total_symbols = 0
            block_errors = 0
            total_blocks = 0
            completed_iterations = 0

            fixed_ebno = torch.full(
                (EVALUATION_BATCH_SIZE,),
                ebno_value,
                dtype=torch.float32,
                device=device,
            )

            for mc_iteration in range(MAX_MC_ITER):
                (
                    bits,
                    estimated_bits,
                    true_indices,
                    estimated_indices,
                ) = model(EVALUATION_BATCH_SIZE, fixed_ebno)

                bit_error_mask = bits != estimated_bits
                symbol_error_mask = true_indices != estimated_indices

                bit_errors += int(bit_error_mask.sum().item())
                total_bits += int(bits.numel())

                symbol_errors += int(symbol_error_mask.sum().item())
                total_symbols += int(true_indices.numel())

                block_errors += int(bit_error_mask.any(dim=-1).sum().item())
                total_blocks += int(bits.shape[0])
                completed_iterations = mc_iteration + 1

                if (
                    completed_iterations >= MIN_MC_ITER
                    and bit_errors >= NUM_TARGET_BIT_ERRORS
                ):
                    break

            ber = bit_errors / total_bits
            ser = symbol_errors / total_symbols
            bler = block_errors / total_blocks

            results["ebno_db"].append(ebno_value)
            results["bit_errors"].append(bit_errors)
            results["total_bits"].append(total_bits)
            results["ber"].append(ber)
            results["symbol_errors"].append(symbol_errors)
            results["total_symbols"].append(total_symbols)
            results["ser"].append(ser)
            results["block_errors"].append(block_errors)
            results["total_blocks"].append(total_blocks)
            results["bler"].append(bler)
            results["mc_iterations"].append(completed_iterations)

            print(
                f"{model_name:24s} | Eb/N0={ebno_value:5.1f} dB | "
                f"BER={ber:.6e} ({bit_errors}/{total_bits}) | "
                f"SER={ser:.6e} | BLER={bler:.6e} | "
                f"MC={completed_iterations}"
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {key: np.asarray(value) for key, value in results.items()}


# ============================================================
# Saving and plotting
# ============================================================


def save_training_logs(records: List[Dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "label",
        "iteration",
        "configured_train_min_db",
        "configured_train_max_db",
        "sampled_mean_ebno_db",
        "sampled_min_ebno_db",
        "sampled_max_ebno_db",
        "loss",
    ]

    with TRAINING_LOG_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_evaluation_results(
    evaluation: Dict[str, Dict[str, np.ndarray]],
) -> None:
    fieldnames = [
        "model",
        "label",
        "train_min_db",
        "train_max_db",
        "ebno_db",
        "bit_errors",
        "total_bits",
        "ber",
        "symbol_errors",
        "total_symbols",
        "ser",
        "block_errors",
        "total_blocks",
        "bler",
        "mc_iterations",
    ]

    with EVALUATION_RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for model_name, model_results in evaluation.items():
            config = MODEL_CONFIGS[model_name]

            for index in range(len(model_results["ebno_db"])):
                writer.writerow(
                    {
                        "model": model_name,
                        "label": config["label"],
                        "train_min_db": config["train_min_db"],
                        "train_max_db": config["train_max_db"],
                        "ebno_db": float(model_results["ebno_db"][index]),
                        "bit_errors": int(model_results["bit_errors"][index]),
                        "total_bits": int(model_results["total_bits"][index]),
                        "ber": float(model_results["ber"][index]),
                        "symbol_errors": int(model_results["symbol_errors"][index]),
                        "total_symbols": int(model_results["total_symbols"][index]),
                        "ser": float(model_results["ser"][index]),
                        "block_errors": int(model_results["block_errors"][index]),
                        "total_blocks": int(model_results["total_blocks"][index]),
                        "bler": float(model_results["bler"][index]),
                        "mc_iterations": int(model_results["mc_iterations"][index]),
                    }
                )


def save_parameters() -> None:
    lines = [
        f"random_seed={RANDOM_SEED}",
        f"evaluation_seed={EVALUATION_SEED}",
        f"device={device}",
        "",
        "Data organization:",
        f"NUM_BITS_PER_SYMBOL={NUM_BITS_PER_SYMBOL}",
        f"MODULATION_ORDER={MODULATION_ORDER}",
        f"NUM_BITS_PER_FRAME={NUM_BITS_PER_FRAME}",
        f"NUM_DATA_SYMBOLS={NUM_DATA_SYMBOLS}",
        f"NUM_PILOTS={NUM_PILOTS}",
        "pilot_symbol=1+0j",
        f"OUTER_CODERATE={OUTER_CODERATE}",
        f"PILOT_EFFICIENCY={PILOT_EFFICIENCY:.10f}",
        f"EFFECTIVE_CODERATE={EFFECTIVE_CODERATE:.10f}",
        f"EFFECTIVE_INFORMATION_RATE={EFFECTIVE_INFORMATION_RATE:.10f}",
        "",
        "Channel:",
        "channel_type=RayleighBlockFading",
        "one_channel_coefficient_per_frame=True",
        "estimator=4-pilot least squares",
        "equalizer=regularized ZF",
        f"ZF_EPSILON={ZF_EPSILON}",
        "true_h_given_to_transmitter=False",
        "true_h_given_to_receiver=False",
        "",
        "Model:",
        "transmitter=bits -> Sionna Mapper -> trainable 64-point constellation",
        f"receiver=3->{HIDDEN_DIM}->{HIDDEN_DIM}->{NUM_BITS_PER_SYMBOL}",
        "receiver_input=Re(y_equalized), Im(y_equalized), log10(no_equalized)",
        "loss=bit-wise BCEWithLogits",
        "one_hot_used=False",
        "",
        "Training models:",
    ]

    for model_name, config in MODEL_CONFIGS.items():
        lines.extend(
            [
                f"{model_name}.label={config['label']}",
                f"{model_name}.train_min_db={config['train_min_db']}",
                f"{model_name}.train_max_db={config['train_max_db']}",
                f"{model_name}.sampling=independent uniform per sample",
            ]
        )

    parameter_count = count_trainable_parameters(build_model(training=True))

    lines.extend(
        [
            "",
            "Shared training parameters:",
            f"NUM_TRAINING_ITERATIONS={NUM_TRAINING_ITERATIONS}",
            f"TRAINING_BATCH_SIZE={TRAINING_BATCH_SIZE}",
            f"LEARNING_RATE={LEARNING_RATE}",
            "optimizer=Adam",
            f"trainable_parameters_per_model={parameter_count}",
            "",
            "Evaluation:",
            f"EVAL_EBNO_DB={EVAL_EBNO_DB.detach().cpu().tolist()}",
            f"EVALUATION_BATCH_SIZE={EVALUATION_BATCH_SIZE}",
            f"MIN_MC_ITER={MIN_MC_ITER}",
            f"MAX_MC_ITER={MAX_MC_ITER}",
            f"NUM_TARGET_BIT_ERRORS={NUM_TARGET_BIT_ERRORS}",
            "same_random_test_samples_across_models=True",
            "metrics=BER, SER, BLER, explicit error counts",
        ]
    )

    PARAMETERS_PATH.write_text("\n".join(lines), encoding="utf-8")


def plot_metric(
    evaluation: Dict[str, Dict[str, np.ndarray]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9.2, 6.1))

    for model_name, model_results in evaluation.items():
        config = MODEL_CONFIGS[model_name]
        plt.semilogy(
            model_results["ebno_db"],
            model_results[metric],
            marker=str(config["marker"]),
            markevery=2,
            label=str(config["label"]),
        )

    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_loss(records: List[Dict[str, object]]) -> None:
    plt.figure(figsize=(9.2, 6.1))

    for model_name, config in MODEL_CONFIGS.items():
        selected = [record for record in records if record["model"] == model_name]
        iterations = [int(record["iteration"]) for record in selected]
        losses = [float(record["loss"]) for record in selected]

        plt.plot(iterations, losses, label=str(config["label"]))

    plt.xlabel("Training iteration")
    plt.ylabel("Bit-wise BCE loss")
    plt.title("Trainable-Mapping Loss for Different Eb/N0 Ranges")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAINING_LOSS_FIGURE_PATH, dpi=200)
    plt.close()


def load_constellation_points(model_name: str) -> np.ndarray:
    model = build_model(training=False).to(device)
    model.load_state_dict(torch.load(weights_path(model_name), map_location=device))
    model.eval()

    with torch.no_grad():
        points = model.constellation_points().detach().cpu().numpy()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return points


def plot_constellation(
    points: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6.4, 6.4))
    plt.scatter(points.real, points.imag)
    plt.axhline(0.0, linewidth=0.8)
    plt.axvline(0.0, linewidth=0.8)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_constellation_figures() -> None:
    initial = standard_qam_points().cpu().numpy()
    plot_constellation(
        initial,
        "Initial Normalized 64-QAM Constellation",
        RESULT_DIR / "constellation_initial_64qam.png",
    )

    learned: Dict[str, np.ndarray] = {}

    for model_name, config in MODEL_CONFIGS.items():
        points = load_constellation_points(model_name)
        learned[model_name] = points

        plot_constellation(
            points,
            f"Learned Constellation: {config['label']}",
            RESULT_DIR / f"constellation_{model_name}.png",
        )

    plt.figure(figsize=(7.2, 7.2))
    for model_name, points in learned.items():
        config = MODEL_CONFIGS[model_name]
        plt.scatter(
            points.real,
            points.imag,
            marker=str(config["marker"]),
            label=str(config["label"]),
            alpha=0.75,
        )

    plt.axhline(0.0, linewidth=0.8)
    plt.axvline(0.0, linewidth=0.8)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title("Learned Constellations Across Training SNR Ranges")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONSTELLATION_COMPARISON_PATH, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 100)
    print("Rayleigh Trainable-Mapping Training-SNR Range Comparison")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Result directory: {RESULT_DIR.resolve()}")
    print(f"Information bits per frame: {NUM_BITS_PER_FRAME}")
    print(f"Data symbols per frame: {NUM_DATA_SYMBOLS}")
    print(f"Pilots per frame: {NUM_PILOTS}")
    print(f"Effective information rate: {EFFECTIVE_INFORMATION_RATE:.6f} bits/use")
    print("Training ranges: Uniform(0,4), Uniform(4,8), Uniform(8,12) dB")
    print("Evaluation range: 0--20 dB, step 0.5 dB")
    print()

    save_parameters()

    training_records: List[Dict[str, object]] = []

    if TRAIN_MODELS:
        for model_name in MODEL_CONFIGS:
            training_records.extend(train_model(model_name))

        save_training_logs(training_records)
        plot_training_loss(training_records)
    else:
        print("Training skipped because TRAIN_MODELS=False")

    evaluation: Dict[str, Dict[str, np.ndarray]] = {}

    if EVALUATE_MODELS:
        for model_name in MODEL_CONFIGS:
            evaluation[model_name] = evaluate_model(model_name)

        save_evaluation_results(evaluation)

        plot_metric(
            evaluation,
            metric="ber",
            ylabel="Bit Error Rate (BER)",
            title="Trainable Mapping BER for Different Training Eb/N0 Ranges",
            output_path=BER_FIGURE_PATH,
        )
        plot_metric(
            evaluation,
            metric="ser",
            ylabel="Symbol Error Rate (SER)",
            title="Trainable Mapping SER for Different Training Eb/N0 Ranges",
            output_path=SER_FIGURE_PATH,
        )
        plot_metric(
            evaluation,
            metric="bler",
            ylabel="Block Error Rate (BLER)",
            title="Trainable Mapping 1500-Bit Frame BLER",
            output_path=BLER_FIGURE_PATH,
        )

        save_constellation_figures()
    else:
        print("Evaluation skipped because EVALUATE_MODELS=False")

    print("=" * 100)
    print("Finished")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print(f"Training log:      {TRAINING_LOG_PATH}")
    print(f"Evaluation CSV:    {EVALUATION_RESULTS_PATH}")
    print(f"BER figure:        {BER_FIGURE_PATH}")
    print(f"SER figure:        {SER_FIGURE_PATH}")
    print(f"BLER figure:       {BLER_FIGURE_PATH}")
    print(f"Parameters:        {PARAMETERS_PATH}")
    print("=" * 100)


if __name__ == "__main__":
    main()
