
from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding
import matplotlib.pyplot as plt


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class ParallelRooms(Env):
    """
        two rooms environment 5x11 , 55 states,4 actions
        
           ["FFFF---FFFF",
            "FFFF---FFFF",
            "FFFFFSFFFFF",
            "FFFF---FFFF",
            "FFFG---FFFG"]

        corridors 10x10 , 100 states,4 actions
        [
            "-FFF------",
            "-FFF------",
            "--FF--FFFF",
            "F--F--F--F",
            "F--F--F--F",
            "FFFFFFSFFF",
            "F--F--F--F",
            "F--F--F--F",
            "---F--FFFF",
            "-FFF------"
        ]

        F = floor
        S = start
        G = goal
        H = hole

    """

    def __init__(self,
                 desc: List[str],
                 is_slippery: bool = False,
                 entropy_mode: bool = False):
        

        self.entropy_mode = entropy_mode

        self.desc = np.asarray([list(row) for row in desc], dtype="c")
        self.nrow, self.ncol = self.desc.shape
        self.is_slippery = is_slippery

        self.P = {}
        
        nA = 4
        nS = self.nrow * self.ncol
        self.valid_states = []
        self._identify_valid_states()

        self.LEFT, self.DOWN, self.RIGHT, self.UP = 0, 1, 2, 3

        self._populate_probability_matrix()
        self.initial_state_distrib = self._initialize_start_state()
        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

    def update_map(self, desc):
        self.desc = np.asarray([list(row) for row in desc], dtype="c")
        self.nrow, self.ncol = self.desc.shape
        self.valid_states = []
        self._identify_valid_states()
        self._populate_probability_matrix()
        self.initial_state_distrib = self._initialize_start_state()


    def _identify_valid_states(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                if self.desc[row, col] != b"-":
                    state = self.to_s(row, col)
                    self.valid_states.append(state)
                    self.P[state] = {a: [] for a in range(4)}
                    


    def to_s(self, row, col):
        return row * self.ncol + col

    def inc(self, row, col, a):
        if a == self.LEFT:
            col = max(col - 1, 0)
        elif a == self.DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == self.RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == self.UP:
            row = max(row - 1, 0)
        return (row, col)

    def update_probability_matrix(self, row, col, action):
        newrow, newcol = self.inc(row, col, action)
        if self.desc[newrow, newcol] == b"-":
            return self.to_s(row, col), 0, False  # Stay in place if hitting a wall
        newstate = self.to_s(newrow, newcol)
        newletter = self.desc[newrow, newcol]
        terminated = bytes(newletter) in b"GH"
        reward = float(newletter == b"G")
        return newstate, reward, terminated

    def _populate_probability_matrix(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                if self.desc[row, col] == b"-":
                    continue  # Skip walls
                s = self.to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = self.desc[row, col]
                    if letter in b"GH":
                        self.goal = (row, col)
                        li.append((1.0, s, 0, True))
                    else:
                        if self.is_slippery:
                            for b in range(4):
                                if b == a:
                                    li.append(
                                        (0.8, *self.update_probability_matrix(row, col, b))
                                    )
                                else:
                                    li.append(
                                        ((0.2 / 3), *self.update_probability_matrix(row, col, b))
                                    )
                        else:
                            li.append((1.0, *self.update_probability_matrix(row, col, a)))
    
    def _initialize_start_state(self):
        initial_distribution = np.zeros(self.nrow * self.ncol)
        for row in range(self.nrow):
            for col in range(self.ncol):
                if self.desc[row, col] == b"S":
                    initial_distribution[self.to_s(row, col)] = 1.0
        return initial_distribution / initial_distribution.sum()

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        if self.entropy_mode:
            return (int(s), r, False, False, {"prob": p})
        else:
            return (int(s), r, t, False, {"prob": p})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}



def plot_gridworld(map_text):
    # Define color mapping
    color_mapping = {
        'F': "white",   # Normal floor
        'S': "green",   # Starting position
        'C': "yellow",  # Checkpoint
        'H': "red",     # Hazard
        'G': "blue",    # Goal
        '-': "black"    # Wall
    }

    # Create a gridworld matrix based on the map
    grid = np.array([list(row) for row in map_text])

    # Determine the grid size
    rows, cols = grid.shape

    # Create the plot
    fig, ax = plt.subplots(figsize=(cols, rows))

    for r in range(rows):
        for c in range(cols):
            cell = grid[r, c]
            color = color_mapping.get(cell, "white")
            rect = plt.Rectangle((c, rows - r - 1), 1, 1, color=color)
            ax.add_patch(rect)

    # Set grid limits and aesthetics
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_yticks(np.arange(0, rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='gray', linestyle='-', linewidth=0.5)

    plt.show()

# # Define the map
# map_text = [
#     "FFFF---FFFF",
#     "FFFF---FFFF",
#     "FFFFCSCFFFF",
#     "FFFF---FHFF",
#     "FFFG---FFFG",
# ]

# # Create the gridworld
# create_gridworld(map_text)