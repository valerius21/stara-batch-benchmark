import argparse
from time import time_ns, time
from typing import Tuple, List, Optional
import subprocess
import random


import pandas as pd
import numpy as np
from numpy.typing import NDArray
from loguru import logger
from tqdm import tqdm


from stara_maze_generator.vmaze import VMaze
from stara_astar.astar import AStar as AstarNaive
from stara_astar_stdlib.a_star_stdlib import AStarStdLib
from stara_astar_numba.astar_numba import AStarNumba

from stara_maze_generator.pathfinder.base import PathfinderBase
from stara_rs.stara_rs import MazeSolver
from stara_cpp.stara_cpp import AStar as AstarCpp
from stara_cpp.stara_cpp import load_maze as load_cpp_maze

from batch_mazes.generator import generate_maze


class AstarCxx(PathfinderBase):
    def __init__(self, maze: VMaze) -> None:
        super().__init__(maze)
        mmap = maze.maze_map.copy()

        # cast mmap elements to np.int32

        mmap = mmap.astype(np.int32)

        maze_ptr = load_cpp_maze(mmap)
        self.solver = AstarCpp(maze_ptr)

    def find_path(
        self,
        start: NDArray[np.int32] | Tuple[int, int],
        goal: NDArray[np.int32] | Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        return self.solver.find_path(start, goal)


class AStarRS(PathfinderBase):
    def __init__(self, maze):
        super().__init__(maze)
        self.solver = MazeSolver()
        self.solver.load(self.maze.maze_map)

    def find_path(
        self,
        start: NDArray[np.int32] | Tuple[int, int],
        goal: NDArray[np.int32] | Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        return self.solver.astar(start, goal)


def stara_rs_preprocess(df, N=1_000):
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


def run_find_path(solver: PathfinderBase, maze: VMaze, N=1_000) -> float:
    start_time = time_ns()
    start = (maze.start[0], maze.start[1])
    goal = (maze.goal[0], maze.goal[1])
    solver.find_path(start, goal)
    end_time = time_ns()
    delta = end_time - start_time
    if delta <= 1:
        start_time = time_ns()
        for _ in range(N):
            solver.find_path(start, goal)
        end_time = time_ns()
        delta = (end_time - start_time) / N
    return delta


def shuffe_row_benchmark(maze: VMaze, runs=1_000):
    options = {
        "naive": {"runs": runs, "instance": AstarNaive(maze), "times": []},
        "stdlib": {"runs": runs, "instance": AStarStdLib(maze), "times": []},
        "rust": {"runs": runs, "instance": AStarRS(maze), "times": []},
        "numba": {"runs": runs, "instance": AStarNumba(maze), "times": []},
        "cpp": {"runs": runs, "instance": AstarCxx(maze), "times": []},
    }
    has_runs_left = any(option["runs"] > 0 for option in options.values())
    while has_runs_left:
        random_impl = random.choice(
            list([key for key in options.keys() if options[key]["runs"] > 0])
        )
        option = options[random_impl]
        option["runs"] -= 1
        time = run_find_path(option["instance"], maze)
        option["times"].append(time)
        has_runs_left = any(option["runs"] > 0 for option in options.values())

    return options


def get_maze_from_dataframe(row, maze_size, maze_valid_path_count):
    if row["seed"] is None:
        raise ValueError("seed is required")
    return generate_maze(maze_size, row["seed"], maze_valid_path_count)


def process_row_shuffle(row, maze_size, maze_valid_path_count):
    maze: VMaze = row["maze"]
    if maze is None:
        maze = get_maze_from_dataframe(row, maze_size, maze_valid_path_count)
    results = shuffe_row_benchmark(maze)
    for key, value in results.items():
        row[key] = np.mean(value["times"])
    return row


def process_row(row, maze_size, maze_valid_path_count):
    maze: VMaze = row["maze"]
    if maze is None:
        maze = get_maze_from_dataframe(row, maze_size, maze_valid_path_count)

    # Naive A* search
    naive = AstarNaive(maze)
    row["naive"] = run_find_path(naive, maze)

    # StdLib A* search
    stdlib = AStarStdLib(maze)
    row["stdlib"] = run_find_path(stdlib, maze)

    # Rust A* search
    rs = AStarRS(maze)
    row["pyo3"] = run_find_path(rs, maze)

    # numba A* search
    numba = AStarNumba(maze)
    row["numba"] = run_find_path(numba, maze)

    # C++ A* search
    cxx = AstarCxx(maze)
    row["cpp"] = run_find_path(cxx, maze)

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
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--nuitka-bin", type=str, default="astar_nuitka.bin")
    parser.add_argument("--maze-file", type=str, default="10x10_1000.pkl")
    args = parser.parse_args()

    mazes: pd.DataFrame | pd.Series = pd.read_pickle(args.maze_file)
    maze_size = mazes.iloc[0]["size"]
    logger.info(f"Maze size: {maze_size}")

    tqdm.pandas(desc="[Benchmark] Processing mazes")

    if args.shuffle:
        mazes = mazes.progress_apply(process_row_shuffle, axis=1)
    else:
        mazes = mazes.progress_apply(
            lambda row: process_row(row, row["size"], row["min_valid_paths"]), axis=1
        )

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
