import argparse
from src.train import run_training, parse_args
from src.evaluate import evaluate

if __name__ == "__main__":
    args = parse_args()
    test_dataset = run_training(args)
    evaluate(test_dataset, args.batch_size)
