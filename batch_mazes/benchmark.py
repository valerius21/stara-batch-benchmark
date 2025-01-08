import argparse
from time import time_ns, time
from typing import Tuple, List, Optional
import subprocess


import pandas as pd
from loguru import logger
from tqdm import tqdm


from stara_maze_generator.vmaze import VMaze
from stara_astar.astar import AStar as AstarNaive
from stara_astar_stdlib.a_star_stdlib import AStarStdLib
from stara_astar_numba.astar_numba import AStarNumba

from stara_maze_generator.pathfinder.base import PathfinderBase
from stara_rs.stara_rs import MazeSolver

# Number of runs, if the time is less than 1ns, we run N times and take the average
N = 1_000


class AStarRS(PathfinderBase):
    def __init__(self, maze):
        super().__init__(maze)
        self.solver = MazeSolver()
        self.solver.load(self.maze.maze_map)

    def find_path(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        return self.solver.astar(start, goal)


def stara_rs_preprocess(df):
    astar_mazes = []
    for row in tqdm(
        df.itertuples(), desc="[stara-rs] Create AStarRS objects", total=len(df)
    ):
        maze = row.maze
        astar_mazes.append(AStarRS(maze))

    res = []
    for astar_maze in tqdm(
        astar_mazes, desc="[stara-rs] Find Path for Mazes", total=len(astar_mazes)
    ):
        start = (astar_maze.maze.start[0], astar_maze.maze.start[1])
        goal = (astar_maze.maze.goal[0], astar_maze.maze.goal[1])
        start_time = time_ns()
        astar_maze.find_path(start, goal)
        end_time = time_ns()
        delta = end_time - start_time
        if end_time - start_time <= 1:
            start_time = time_ns()
            for _ in range(N):
                astar_maze.find_path(start, goal)
            end_time = time_ns()

            delta = (end_time - start_time) / N
        res.append(
            {
                "seed": astar_maze.maze.seed,
                "pyo3_pp": delta,
            }
        )
    return pd.DataFrame(res)


def process_row(row):
    maze: VMaze = row["maze"]

    # Naive A* search
    naive = AstarNaive(maze)
    start_time = time_ns()
    naive.find_path(maze.start, maze.goal)
    end_time = time_ns()
    delta = end_time - start_time
    if delta <= 1:
        start_time = time_ns()
        for _ in range(N):
            naive.find_path(maze.start, maze.goal)
        end_time = time_ns()
        delta = (end_time - start_time) / N
    row["naive"] = delta

    # StdLib A* search
    stdlib = AStarStdLib(maze)
    start_time = time_ns()
    stdlib.find_path(maze.start, maze.goal)
    end_time = time_ns()
    delta = end_time - start_time
    if delta <= 1:
        start_time = time_ns()
        for _ in range(N):
            stdlib.find_path(maze.start, maze.goal)
        end_time = time_ns()
        delta = (end_time - start_time) / N
    row["stdlib"] = delta

    # Rust A* search
    rs = AStarRS(maze)
    start = (maze.start[0], maze.start[1])
    goal = (maze.goal[0], maze.goal[1])
    start_time = time_ns()
    rs.find_path(start, goal)
    end_time = time_ns()
    delta = end_time - start_time
    if delta <= 1:
        start_time = time_ns()
        for _ in range(N):
            rs.find_path(start, goal)
        end_time = time_ns()
        delta = (end_time - start_time) / N
    row["pyo3"] = delta

    # numba A* search
    numba = AStarNumba(maze)
    start_time = time_ns()
    numba.find_path(maze.start, maze.goal)
    end_time = time_ns()
    delta = end_time - start_time
    if delta <= 1:
        start_time = time_ns()
        for _ in range(N):
            numba.find_path(maze.start, maze.goal)
        end_time = time_ns()
        delta = (end_time - start_time) / N
    row["numba"] = delta

    return row


def strip_df(df) -> pd.DataFrame:
    try:
        df = df.drop(columns=["maze"])
    except KeyError:
        logger.warning("maze column not found")
    try:
        df = df.drop(columns=["min_valid_paths"])
    except KeyError:
        logger.warning("min_valid_paths column not found")
    try:
        df = df.drop(columns=["start"])
    except KeyError:
        logger.warning("start column not found")
    try:
        df = df.drop(columns=["goal"])
    except KeyError:
        logger.warning("goal column not found")
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

    tqdm.pandas(desc="[Benchmark] Processing mazes")

    mazes = mazes.progress_apply(process_row, axis=1)

    logger.info("[stara-rs] preprocessing")
    stara_rs_mazes = stara_rs_preprocess(mazes)

    #  merge the results
    mazes = strip_df(mazes).merge(stara_rs_mazes, on="seed")

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

    timestamp = str(time()).split(".")[0]
    file_name = f"{args.maze_file}_run-{timestamp}.pkl"

    logger.info(f"Saving results to {file_name}")
    mazes.to_pickle(file_name)
    logger.info("Done")

    return 0


if __name__ == "__main__":
    main()
