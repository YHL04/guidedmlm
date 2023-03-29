import argparse

from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--max_files", type=int, default=10000)
    parser.add_argument("--worker", type=int, default=2)

    args = parser.parse_args()

