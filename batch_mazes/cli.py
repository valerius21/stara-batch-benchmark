import argparse
from stara_maze_generator.vmaze import VMaze
import pandas as pd
import numpy as np
from loguru import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze-size", type=int, default=10)
    parser.add_argument("--maze-count", type=int, default=10)
    parser.add_argument("--min-valid-paths", type=int, default=3)
    parser.add_argument("--output-path", type=str, default="mazes.pkl")
    args = parser.parse_args()

    if args.maze_count > 10_000_000:
        logger.error("maze-count must be less than 10_000_000")
        return

    tries = 0
    while tries < 10:
        # create maze_count unique seeds with a max seed of <= 10_000_000
        rng = np.random.default_rng(0xDEADBEEF)
        seeds = rng.integers(0, 10_000_000, args.maze_count)
        unique_seeds = np.unique(seeds)
        if len(unique_seeds) != args.maze_count:
            logger.warning(f"Not enough unique seeds, retrying ({tries + 1}/10)")
            tries += 1
            continue
        break

    if tries == 10:
        logger.error("Failed to generate enough unique seeds")
        return

    logger.info(f"Found {len(unique_seeds)} unique seeds")

    logger.info(f"Generating {args.maze_count} mazes")
    df = pd.DataFrame()

    for seed in seeds:
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
