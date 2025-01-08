from stara_astar.astar import AStar as AstarNaive
from time import time_ns
import pandas as pd
import argparse
from loguru import logger
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze-file", type=str, default="10x10_1000.pkl")
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    mazes = pd.read_pickle(args.maze_file)
    maze_size = mazes.shape[0]
    n_runs = args.n_runs
    logger.info(f"Maze size: {maze_size}")
    logger.info(f"Number of runs: {n_runs}")

    logger.info("df.head():")
    logger.info(mazes.head())

    return 0
