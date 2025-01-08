import argparse
from stara_maze_generator.vmaze import VMaze
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

MAX_SEED = 10_000_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze-size", type=int, default=10)
    parser.add_argument("--maze-count", type=int, default=10)
    parser.add_argument("--min-valid-paths", type=int, default=3)
    parser.add_argument("--output-path", type=str, default="mazes.pkl")
    args = parser.parse_args()

    if args.maze_count > MAX_SEED:
        logger.error("maze-count must be less than 10_000_000")
        return

    rng = np.random.default_rng()
    seeds = rng.choice(MAX_SEED, args.maze_count, replace=False)

    logger.info(f"Generating {args.maze_count} mazes")
    df = pd.DataFrame()

    for i in tqdm(range(args.maze_count)):
        seed = seeds[i]
        maze = VMaze(
            size=args.maze_size,
            seed=seed,
            min_valid_paths=args.min_valid_paths,
            start=np.array([0, 0]),
            goal=np.array([args.maze_size - 1, args.maze_size - 1]),
        )
        df = df._append(
            {
                "seed": seed,
                "maze": maze,
                "size": args.maze_size,
                "min_valid_paths": args.min_valid_paths,
                "start": maze.start.tolist(),
                "goal": maze.goal.tolist(),
            },
            ignore_index=True,
        )

    df.to_pickle(args.output_path)
    logger.info(f"Saved to {args.output_path}")
    logger.info(f"df.head():\n{df.head()}")


if __name__ == "__main__":
    main()
