#!/usr/bin/env python3
"""Compare fixed vs learned constellation under 4 pilots and 1 pilot.

This script trains and evaluates four models under the same Rayleigh block-fading
setting and puts all four BER curves into ONE figure:

1. 4 pilots  + Fixed 64-QAM + Neural Demapper
2. 4 pilots  + Trainable Constellation + Neural Demapper
3. 1 pilot   + Fixed 64-QAM + Neural Demapper
4. 1 pilot   + Trainable Constellation + Neural Demapper

Changes requested by the user:
- training iterations reduced to 20,000
- one combined BER figure with four curves
- BER y-axis manually tightened to make the gap easier to see
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

if NUM_BITS_PER_FRAME % NUM_BITS_PER_SYMBOL != 0:
    raise ValueError("NUM_BITS_PER_FRAME must be divisible by NUM_BITS_PER_SYMBOL.")

OUTER_CODERATE = 1.0

HIDDEN_DIM = 128
ZF_EPSILON = 1e-6


# ============================================================
# Experiment configuration
# ============================================================

MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "pilot4_fixed": {
        "label": "4 pilots · Fixed 64-QAM + Neural Demapper",
        "num_pilots": 4,
        "trainable_tx": False,
        "marker": "o",
        "linestyle": "-",
    },
    "pilot4_learned": {
        "label": "4 pilots · Trainable Constellation + Neural Demapper",
        "num_pilots": 4,
        "trainable_tx": True,
        "marker": "^",
        "linestyle": "-",
    },
    "pilot1_fixed": {
        "label": "1 pilot · Fixed 64-QAM + Neural Demapper",
        "num_pilots": 1,
        "trainable_tx": False,
        "marker": "s",
        "linestyle": "--",
    },
    "pilot1_learned": {
        "label": "1 pilot · Trainable Constellation + Neural Demapper",
        "num_pilots": 1,
        "trainable_tx": True,
        "marker": "D",
        "linestyle": "--",
    },
}

TRAIN_MIN_DB = 8.0
TRAIN_MAX_DB = 20.0

# User requested that future training iterations be 20,000.
NUM_TRAINING_ITERATIONS = 20_000
TRAINING_BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LOG_INTERVAL = 100

EVAL_EBNO_DB = torch.arange(0.0, 20.5, 0.5, device=device)
EVALUATION_BATCH_SIZE = 256
MIN_MC_ITER = 20
MAX_MC_ITER = 1_000
NUM_TARGET_BIT_ERRORS = 10_000

TRAIN_MODELS = True
EVALUATE_MODELS = True

# Tightened y-axis based on previous runs so the gap looks clearer.
BER_YMIN = 1.1e-2
BER_YMAX = 3.0e-1


# ============================================================
# Output paths
# ============================================================

RESULT_DIR = Path("results/rayleigh_fixed_vs_trainable_4pilot_1pilot_20k")
WEIGHTS_DIR = RESULT_DIR / "weights"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_LOG_PATH = RESULT_DIR / "training_loss.csv"
EVALUATION_RESULTS_PATH = RESULT_DIR / "evaluation_results.csv"
PARAMETERS_PATH = RESULT_DIR / "parameters.txt"

BER_FIGURE_PATH = RESULT_DIR / "ber_4pilot_1pilot_fixed_vs_trainable.png"
TRAINING_LOSS_FIGURE_PATH = RESULT_DIR / "training_loss.png"


# ============================================================
# Bit utilities
# ============================================================

def group_bits(bits: torch.Tensor) -> torch.Tensor:
    """[batch, 1500] -> [batch, 250, 6]."""
    return bits.reshape(
        bits.shape[0],
        NUM_DATA_SYMBOLS,
        NUM_BITS_PER_SYMBOL,
    )


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
# Constellation helpers
# ============================================================

def standard_qam_points() -> torch.Tensor:
    constellation = Constellation("qam", NUM_BITS_PER_SYMBOL)
    return constellation.points.detach().clone().to(torch.complex64)


def center_and_normalize_points(points: torch.Tensor) -> torch.Tensor:
    centered = points - torch.mean(points)
    average_energy = torch.mean(torch.abs(centered) ** 2)
    return centered / torch.sqrt(average_energy + 1e-12)


# ============================================================
# Transmitters
# ============================================================

class Fixed64QAMMapper(nn.Module):
    """Fixed normalized 64-QAM transmitter."""

    def __init__(self) -> None:
        super().__init__()
        self.constellation = Constellation("qam", NUM_BITS_PER_SYMBOL)
        self._mapper = Mapper(constellation=self.constellation)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        return self._mapper(bits)

    def constellation_points(self) -> torch.Tensor:
        return self.constellation.points.detach().clone().to(torch.complex64)


class TrainableConstellationMapper(nn.Module):
    """Trainable 64-point constellation initialized from the same 64-QAM."""

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

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        self.constellation.points = torch.complex(self.points_r, self.points_i)
        return self._mapper(bits)

    def constellation_points(self) -> torch.Tensor:
        raw_points = torch.complex(self.points_r, self.points_i)
        return center_and_normalize_points(raw_points)


# ============================================================
# Neural demapper
# ============================================================

def build_receiver_features(
    y_equalized: torch.Tensor,
    no_equalized: torch.Tensor,
) -> torch.Tensor:
    no_feature = torch.log10(no_equalized.clamp_min(1e-12))
    no_feature = no_feature.expand(-1, NUM_DATA_SYMBOLS)

    return torch.stack(
        [
            y_equalized.real,
            y_equalized.imag,
            no_feature,
        ],
        dim=-1,
    )


class NeuralDemapper(nn.Module):
    """3 -> 128 -> 128 -> 6 bit-logit neural demapper."""

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
        return self._dense_3(z)


# ============================================================
# Variable-pilot Rayleigh channel
# ============================================================

class RayleighPilotChannel(nn.Module):
    """Rayleigh block fading + AWGN + pilot-based LS + regularized ZF."""

    def __init__(self, num_pilots: int) -> None:
        super().__init__()

        self._num_pilots = num_pilots

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

    def _generate_block_channel(self, batch_size: int) -> torch.Tensor:
        a, _ = self._rayleigh(batch_size=batch_size, num_time_steps=1)
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

        h = self._generate_block_channel(batch_size)
        y_frame = self._awgn(h * x_frame, no)

        y_pilot = y_frame[:, :self._num_pilots]
        y_data = y_frame[:, self._num_pilots:]

        h_hat = (
            (torch.conj(pilots) * y_pilot).sum(dim=-1, keepdim=True)
            / (torch.abs(pilots) ** 2).sum(dim=-1, keepdim=True)
        )

        return self._regularized_zf(y_data, h_hat, no)


# ============================================================
# Complete system
# ============================================================

class E2ESystem(nn.Module):
    """Common system; only num_pilots and trainable/fixed Tx differ."""

    def __init__(
        self,
        training: bool,
        trainable_tx: bool,
        num_pilots: int,
    ) -> None:
        super().__init__()

        self._training_mode = training
        self._num_pilots = num_pilots

        self._binary_source = BinarySource()

        if trainable_tx:
            self._transmitter = TrainableConstellationMapper()
        else:
            self._transmitter = Fixed64QAMMapper()

        self._channel = RayleighPilotChannel(num_pilots=num_pilots)
        self._receiver = NeuralDemapper()

        pilot_efficiency = NUM_DATA_SYMBOLS / (NUM_DATA_SYMBOLS + num_pilots)
        self._effective_coderate = OUTER_CODERATE * pilot_efficiency

    def constellation_points(self) -> torch.Tensor:
        return self._transmitter.constellation_points()

    def forward(
        self,
        batch_size: int,
        ebno_db: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if ebno_db.dim() == 0:
            ebno_db = ebno_db.expand(batch_size)

        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
            coderate=self._effective_coderate,
        )
        no = expand_to_rank(no, 2)

        bits = self._binary_source([batch_size, NUM_BITS_PER_FRAME])
        grouped_bits = group_bits(bits)
        true_indices = grouped_bits_to_indices(grouped_bits)

        x_data = self._transmitter(bits)
        y_equalized, no_equalized = self._channel(x_data, no)
        bit_logits = self._receiver(y_equalized, no_equalized)

        if self._training_mode:
            return F.binary_cross_entropy_with_logits(bit_logits, grouped_bits)

        estimated_grouped_bits = (bit_logits > 0.0).float()
        estimated_bits = estimated_grouped_bits.reshape(batch_size, NUM_BITS_PER_FRAME)
        estimated_indices = grouped_bits_to_indices(estimated_grouped_bits)

        return bits, estimated_bits, true_indices, estimated_indices


# ============================================================
# Helpers
# ============================================================

def build_model(model_name: str, training: bool) -> E2ESystem:
    config = MODEL_CONFIGS[model_name]

    return E2ESystem(
        training=training,
        trainable_tx=bool(config["trainable_tx"]),
        num_pilots=int(config["num_pilots"]),
    )


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
    config = MODEL_CONFIGS[model_name]

    reset_random_seed(RANDOM_SEED)

    model = build_model(model_name=model_name, training=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    records: List[Dict[str, object]] = []

    print("-" * 100)
    print(f"Training {config['label']}")
    print(f"Num pilots: {config['num_pilots']}")
    print(f"Trainable transmitter: {config['trainable_tx']}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    for iteration in range(NUM_TRAINING_ITERATIONS):
        optimizer.zero_grad()

        ebno_db = torch.empty(TRAINING_BATCH_SIZE, device=device).uniform_(
            TRAIN_MIN_DB,
            TRAIN_MAX_DB,
        )

        loss = model(TRAINING_BATCH_SIZE, ebno_db)

        loss.backward()
        optimizer.step()

        if iteration % LOG_INTERVAL == 0 or iteration == NUM_TRAINING_ITERATIONS - 1:
            record = {
                "model": model_name,
                "label": str(config["label"]),
                "num_pilots": int(config["num_pilots"]),
                "trainable_tx": bool(config["trainable_tx"]),
                "iteration": iteration,
                "loss": float(loss.item()),
                "sampled_mean_ebno_db": float(ebno_db.mean().item()),
            }

            records.append(record)

            print(
                f"{model_name:14s} | "
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
    config = MODEL_CONFIGS[model_name]
    weight_file = weights_path(model_name)

    if not weight_file.exists():
        raise FileNotFoundError(f"Missing weights: {weight_file}")

    model = build_model(model_name=model_name, training=False).to(device)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.eval()

    results: Dict[str, List[float | int]] = {
        "ebno_db": [],
        "bit_errors": [],
        "total_bits": [],
        "ber": [],
        "mc_iterations": [],
    }

    print("-" * 100)
    print(f"Evaluating {config['label']}")

    with torch.inference_mode():
        for snr_index, ebno_scalar in enumerate(EVAL_EBNO_DB):
            ebno_value = float(ebno_scalar.item())

            reset_random_seed(EVALUATION_SEED + snr_index)

            bit_errors = 0
            total_bits = 0
            completed_iterations = 0

            fixed_ebno = torch.full(
                (EVALUATION_BATCH_SIZE,),
                ebno_value,
                dtype=torch.float32,
                device=device,
            )

            for mc_iteration in range(MAX_MC_ITER):
                bits, estimated_bits, _, _ = model(EVALUATION_BATCH_SIZE, fixed_ebno)

                bit_error_mask = bits != estimated_bits
                bit_errors += int(bit_error_mask.sum().item())
                total_bits += int(bits.numel())

                completed_iterations = mc_iteration + 1

                if completed_iterations >= MIN_MC_ITER and bit_errors >= NUM_TARGET_BIT_ERRORS:
                    break

            ber = bit_errors / total_bits

            results["ebno_db"].append(ebno_value)
            results["bit_errors"].append(bit_errors)
            results["total_bits"].append(total_bits)
            results["ber"].append(ber)
            results["mc_iterations"].append(completed_iterations)

            print(
                f"{model_name:14s} | "
                f"Eb/N0={ebno_value:5.1f} dB | "
                f"BER={ber:.6e} ({bit_errors}/{total_bits}) | "
                f"MC={completed_iterations}"
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        key: np.asarray(value)
        for key, value in results.items()
    }


# ============================================================
# Saving
# ============================================================

def save_training_logs(records: List[Dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "label",
        "num_pilots",
        "trainable_tx",
        "iteration",
        "loss",
        "sampled_mean_ebno_db",
    ]

    with TRAINING_LOG_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_evaluation_results(evaluation: Dict[str, Dict[str, np.ndarray]]) -> None:
    fieldnames = [
        "model",
        "label",
        "num_pilots",
        "trainable_tx",
        "ebno_db",
        "bit_errors",
        "total_bits",
        "ber",
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
                        "label": str(config["label"]),
                        "num_pilots": int(config["num_pilots"]),
                        "trainable_tx": bool(config["trainable_tx"]),
                        "ebno_db": float(model_results["ebno_db"][index]),
                        "bit_errors": int(model_results["bit_errors"][index]),
                        "total_bits": int(model_results["total_bits"][index]),
                        "ber": float(model_results["ber"][index]),
                        "mc_iterations": int(model_results["mc_iterations"][index]),
                    }
                )


def save_parameters() -> None:
    lines = [
        f"device={device}",
        f"random_seed={RANDOM_SEED}",
        f"evaluation_seed={EVALUATION_SEED}",
        "",
        "Experiment:",
        "comparison=4 pilots and 1 pilot; fixed vs learned constellation; neural demapper for all models",
        "",
        "Shared communication settings:",
        f"NUM_BITS_PER_SYMBOL={NUM_BITS_PER_SYMBOL}",
        f"MODULATION_ORDER={MODULATION_ORDER}",
        f"NUM_BITS_PER_FRAME={NUM_BITS_PER_FRAME}",
        f"NUM_DATA_SYMBOLS={NUM_DATA_SYMBOLS}",
        f"HIDDEN_DIM={HIDDEN_DIM}",
        f"ZF_EPSILON={ZF_EPSILON}",
        "",
        "Training:",
        f"TRAIN_MIN_DB={TRAIN_MIN_DB}",
        f"TRAIN_MAX_DB={TRAIN_MAX_DB}",
        f"NUM_TRAINING_ITERATIONS={NUM_TRAINING_ITERATIONS}",
        f"TRAINING_BATCH_SIZE={TRAINING_BATCH_SIZE}",
        f"LEARNING_RATE={LEARNING_RATE}",
        "",
        "Evaluation:",
        f"EVAL_EBNO_DB={EVAL_EBNO_DB.detach().cpu().tolist()}",
        f"EVALUATION_BATCH_SIZE={EVALUATION_BATCH_SIZE}",
        f"MIN_MC_ITER={MIN_MC_ITER}",
        f"MAX_MC_ITER={MAX_MC_ITER}",
        f"NUM_TARGET_BIT_ERRORS={NUM_TARGET_BIT_ERRORS}",
        "",
        "BER axis tuning:",
        f"BER_YMIN={BER_YMIN}",
        f"BER_YMAX={BER_YMAX}",
        "",
        "Models:",
    ]

    for model_name, config in MODEL_CONFIGS.items():
        lines.extend(
            [
                f"{model_name}.label={config['label']}",
                f"{model_name}.num_pilots={config['num_pilots']}",
                f"{model_name}.trainable_tx={config['trainable_tx']}",
            ]
        )

    PARAMETERS_PATH.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Plotting
# ============================================================

def plot_training_loss(records: List[Dict[str, object]]) -> None:
    plt.figure(figsize=(9.2, 6.1))

    for model_name, config in MODEL_CONFIGS.items():
        selected = [record for record in records if record["model"] == model_name]

        x = [int(record["iteration"]) for record in selected]
        y = [float(record["loss"]) for record in selected]

        plt.plot(
            x,
            y,
            linestyle=str(config["linestyle"]),
            label=str(config["label"]),
        )

    plt.xlabel("Training iteration")
    plt.ylabel("Bit-wise BCE loss")
    plt.title("Training Loss: 4 Pilots vs 1 Pilot, Fixed vs Learned")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(TRAINING_LOSS_FIGURE_PATH, dpi=250)
    plt.close()


def plot_ber(evaluation: Dict[str, Dict[str, np.ndarray]]) -> None:
    plt.figure(figsize=(10.0, 6.6))

    for model_name, model_results in evaluation.items():
        config = MODEL_CONFIGS[model_name]

        plt.semilogy(
            model_results["ebno_db"],
            model_results["ber"],
            marker=str(config["marker"]),
            markevery=2,
            linestyle=str(config["linestyle"]),
            label=str(config["label"]),
        )

    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER: 4 Pilots vs 1 Pilot, Fixed vs Learned Constellation")
    plt.grid(True, which="both")

    # User requested a more visually obvious figure based on previous results.
    plt.ylim(BER_YMIN, BER_YMAX)

    plt.legend()
    plt.tight_layout()

    plt.savefig(BER_FIGURE_PATH, dpi=250)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("Rayleigh BER Comparison: 4 Pilots vs 1 Pilot, Fixed vs Learned")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Result directory: {RESULT_DIR.resolve()}")
    print(f"Training Eb/N0: Uniform({TRAIN_MIN_DB:g},{TRAIN_MAX_DB:g}) dB")
    print(f"Training iterations: {NUM_TRAINING_ITERATIONS}")
    print("Models:")
    for model_name, config in MODEL_CONFIGS.items():
        print(
            f"  {model_name:14s} | pilots={config['num_pilots']} | "
            f"trainable_tx={config['trainable_tx']}"
        )
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
        plot_ber(evaluation)
    else:
        print("Evaluation skipped because EVALUATE_MODELS=False")

    print("=" * 100)
    print("Finished")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print(f"Training log:      {TRAINING_LOG_PATH}")
    print(f"Evaluation CSV:    {EVALUATION_RESULTS_PATH}")
    print(f"BER figure:        {BER_FIGURE_PATH}")
    print(f"Training loss:     {TRAINING_LOSS_FIGURE_PATH}")
    print(f"Parameters:        {PARAMETERS_PATH}")
    print("=" * 100)


if __name__ == "__main__":
    main()
