import argparse
from time import time_ns, time
from typing import Tuple, List, Optional
import subprocess


import pandas as pd
from loguru import logger
from tqdm import tqdm
from numpy.typing import NDArray


from stara_maze_generator.vmaze import VMaze
from stara_astar.astar import AStar as AstarNaive
from stara_astar_stdlib.a_star_stdlib import AStarStdLib
from stara_astar_numba.astar_numba import AStarNumba

from stara_maze_generator.pathfinder.base import PathfinderBase
from stara_rs.stara_rs import astar


class AStarRS(PathfinderBase):
    def __init__(self, maze):
        super().__init__(maze)
        self.maze_list: List[List[int]] = [list(row) for row in self.maze.maze_map]

    def find_path(
        self, start: NDArray | Tuple[int, int], goal: NDArray | Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        return astar(self.maze_list, start[0], start[1], goal[0], goal[1])


def process_row(row):
    logger.info(f"Processing row seed={row['seed']}")
    maze: VMaze = row["maze"]

    # Naive A* search
    start_time = time_ns()
    AstarNaive(maze).find_path(maze.start, maze.goal)
    end_time = time_ns()
    row["naive"] = end_time - start_time

    # StdLib A* search
    start_time = time_ns()
    AStarStdLib(maze).find_path(maze.start, maze.goal)
    end_time = time_ns()
    row["stdlib"] = end_time - start_time

    # Rust A* search
    start_time = time_ns()
    AStarRS(maze).find_path(maze.start, maze.goal)
    end_time = time_ns()
    row["pyo3"] = end_time - start_time

    # numba A* search
    start_time = time_ns()
    AStarNumba(maze).find_path(maze.start, maze.goal)
    end_time = time_ns()
    row["numba"] = end_time - start_time

    return row


def strip_df(df) -> pd.DataFrame:
    df = df.drop(columns=["maze", "min_valid_paths", "start", "goal"])
    df = df.dropna()
    # 'size' column is a category
    df["size"] = pd.Categorical(df["size"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuitka-bin", type=str, default="astar_nuitka.bin")
    parser.add_argument("--maze-file", type=str, default="10x10_1000.pkl")
    args = parser.parse_args()

    mazes: pd.DataFrame | pd.Series = pd.read_pickle(args.maze_file)
    maze_size = mazes.iloc[0]["size"]
    logger.info(f"Maze size: {maze_size}")

    # Initialize tqdm for pandas
    tqdm.pandas(desc="Processing mazes")

    mazes = mazes.progress_apply(process_row, axis=1)

    logger.info("Running Nuitka")

    process = subprocess.run([args.nuitka_bin, "--maze-file", args.maze_file])
    logger.info(f"Nuitka exit code: {process.returncode}")

    mazes = strip_df(mazes)
    logger.info("Merging results on 'seed' column")
    #  <maze-file>.pkl.json contains the results of the nuitka run
    #  so we can just read that file and merge the results
    #  on the seed column
    mazes = pd.read_json(f"{args.maze_file}.json").merge(mazes, on="seed")

    logger.info("df.head():\n{}".format(mazes.head()))

    logger.info("Saving results")
    timestamp = str(time()).split(".")[0]
    mazes.to_pickle(f"{args.maze_file}_run-{timestamp}.pkl")

    return 0


if __name__ == "__main__":
    main()
