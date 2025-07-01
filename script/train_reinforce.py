# 2025 Steven Chiacchira
"""CLI script for training a paint chaser model."""

import argparse
import os
import sys
from pathlib import Path

import gymnasium
import torch
from torch import optim
from tqdm import tqdm

from paint_chaser.environment import UFO50PaintChaseEnv
from paint_chaser.model import ResidualAgent
from paint_chaser.train import EpisodeBuffer


SCREEN_REGION = {
    "top": 30,
    "left": 1,
    "height": 214,
    "width": 384,
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "train_policy_gradients",
        description="Trains a model to plain paint chase using policy gradients.",
        epilog="2025 Steven Chiacchira",
    )
    parser.add_argument("batches", type=int, help="Number of batches to train")
    parser.add_argument(
        "--model-out",
        "--mo",
        required=True,
        type=Path,
        help="Path to save model weights to",
    )
    parser.add_argument(
        "--model-in",
        "--mi",
        type=Path,
        help="Path to load model weights from. Defaults to no path, which initializes random weights",
    )
    parser.add_argument(
        "--log-out",
        "--lo",
        default=Path("/dev/null"),
        type=Path,
        help="Path to log output to. Defaults to /dev/null",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        "-l",
        default=os.getenv("PAINT_CHASER_LEARNING_RATE") or 1e-5,
        type=float,
        help="Learning rate for training. Reads PAINT_CHASER_LEARNING_RATE environment variable. Otherwise, defaults to 1e-5",
    )
    parser.add_argument(
        "--batch-size",
        "--bs",
        default=os.getenv("PAINT_CHASER_BATCH_SIZE") or 1,
        type=int,
        help="Number of episodes in each batch. Reads PAINT_CHASER_BATCH_SIZE environment variable. Otherwise, defaults to 1, implying no batching",
    )
    parser.add_argument(
        "--save-every-n-batch",
        default=os.getenv("PAINT_CHASER_SAVE_EVERY") or 10,
        type=int,
        help="Number of batches to save between. Reads PAINT_CHASER_SAVE_EVERY environment variable. Otherwise, defaults to 10 batches",
    )
    parser.add_argument(
        "--gamma-good",
        "--gg",
        default=os.getenv("PAINT_CHASER_GAMMA_GOOD") or 0.95,
        type=float,
        help="Reward discount to use for blue pixels. Reads PAINT_CHASER_GAMMA_GOOD environment variable. Otherwise, defaults to 0.95",
    )
    parser.add_argument(
        "--gamma-bad",
        "--gb",
        default=os.getenv("PAINT_CHASER_GAMMA_BAD") or 0.99,
        type=float,
        help="Penalty discount to use for red pixels. Reads PAINT_CHASER_GAMMA_BAD environment variable. Otherwise, defaults to 0.99",
    )
    parser.add_argument(
        "--bad-penalty-factor",
        "--bpf",
        default=os.getenv("PAINT_CHASER_BAD_PENALTY_FACTOR") or 0.3,
        type=float,
        help="Penalty factor for red pixels. Reads PAINT_CHASER_BAD_PENALTY_FACTOR environment variable. Otherwise, defaults to 0.30",
    )
    parser.add_argument(
        "--use-cpu",
        default=os.getenv("PAINT_CHASER_USE_CPU") or False,
        action="store_true",
        help="Flag to force CPU use. Reads PAINT_CHASER_USE_CPU environment variable. Otherwise, defaults to False",
    )

    return parser.parse_args()


def get_device(use_cpu: bool) -> torch.device:
    device = (
        torch.device("cpu")
        if use_cpu
        else torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    return device


def train_one_batch(
    agent: torch.nn.Module,
    env: gymnasium.Env,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma_good: float,
    gamma_bad: float,
    bad_penalty_factor: float,
    device: torch.device,
) -> float:
    batch_buffers = []
    for episode in range(batch_size):
        buffer = EpisodeBuffer(gamma_good, gamma_bad, bad_penalty_factor, device)
        buffer.collect_episode(agent, env)
        batch_buffers.append(buffer)

    policy_loss = torch.zeros((1)).to(device)
    for buffer in batch_buffers:
        _, log_probs, returns = buffer.get_tensors()
        policy_loss += (-log_probs * returns).sum()

    # Average loss across batch
    policy_loss /= batch_size

    # Update policy
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return policy_loss.item()


def save_model_and_optimizer_parameters(
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
) -> None:
    parameters = {}
    parameters["model"] = agent.state_dict()
    parameters["optim"] = optimizer.state_dict()

    torch.save(parameters, path)


def main() -> int:
    args = get_args()
    device = get_device(args.use_cpu)

    env = UFO50PaintChaseEnv(SCREEN_REGION)
    agent = ResidualAgent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    if args.model_in is not None:
        try:
            parameters = torch.load(args.model_in)
            agent.load_state_dict(parameters["model"])
            optimizer.load_state_dict(parameters["optim"])
        except FileNotFoundError:
            print(f"No model weights found at {args.model_in}", file=sys.stderr)
            return 1

    with open(args.log_out, "w") as log_file:
        log_file.write("Batch\tAverage Reward")

    pbar = tqdm(range(1, 1 + args.batches), desc="Batch: 0 | Avg reward: ?")

    for batch in pbar:
        average_reward = train_one_batch(
            agent,
            env,
            optimizer,
            args.batch_size,
            args.gamma_good,
            args.gamma_bad,
            args.bad_penalty_factor,
            device,
        )

        with open(args.log_out, "w") as log_file:
            log_file.write(f"{batch}\t{average_reward}")

        pbar.set_description(f"Batch: {batch} | Avg reward: {average_reward}")

        if batch % args.save_every_n_batch == 0:
            print("Saving model!", file=sys.stderr)
            save_model_and_optimizer_parameters(agent, optimizer, args.model_out)

    save_model_and_optimizer_parameters(agent, optimizer, args.model_out)

    return 0


if __name__ == "__main__":
    exit_code = main()
    print("Finished Training!", file=sys.stderr)
    sys.exit(exit_code)
