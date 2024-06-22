"""
Project: WGU - CS Capstone - C964
File: test_model.py
Author: VD
Date: 6/22/2024
Description:

Test QModel Training
    Test model.train | check_list = all
    Plot model training stats
    Plot Q-Table Best Moves 
    Test moves from every cells
"""

from env.maze import Maze
from env.robot import Robot
from env.common import *
from env.qtable import QModel
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

sz = 10
target = (randint(0,sz - 1), randint(0,sz - 1))

maze = Maze(*Maze.generate(perfect_score = 95, w =sz, h=sz))
rob = Robot(maze = maze, fog=False)
rob.render()
model = QModel(rob)

round = sz * sz
h, w, _, _ = model.train(target = target, discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=round*10,
                             stop_at_convergence=True, check_convergence_every = 5)

# Plot model training stats
if True:
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
    fig.canvas.manager.set_window_title(model.name)
    ax1.plot(*zip(*w))
    ax1.set_xlabel("episode")
    ax1.set_ylabel("win rate")
    ax2.plot(h)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative reward")
    
    plt.show(block=False)

# Plot Q-Table Best Moves 
if True:
    fig, ax1 = plt.subplots(1, 1, tight_layout=True)
    fig.canvas.manager.set_window_title('Q-table (Best Moves)')
    maze.draw_maze(ax1)
    maze.cells[*target].draw_text(ax1, 'Target', fill_color = 'green')
    def clip(n):
        return max(min(1, n), 0)
    for _ in maze.cells:
        for cell in _:
            q = model.q(cell.pos)
            a = np.nonzero(q == np.max(q))[0]

            for action in a:
                dx = 0
                dy = 0
                if action == Direction.LEFT:
                    dx = -0.2
                if action == Direction.RIGHT:
                    dx = +0.2
                if action == Direction.UP:
                    dy = +0.2
                if action == Direction.DOWN:
                    dy = -0.2

                maxv = 1
                minv = -1
                color = clip((q[action] - minv) / (maxv - minv))  # normalize in [-1, 1]
                (x, y) = cell.pos
                ax1.arrow(x+0.5, y+0.5, dx, dy, color=(1 - color, color, 0), head_width=0.2, head_length=0.1)
    
    plt.show(block=False)

# Test moves from every cells (exclude target)
if True:
    rob.render_mode = Render.MOVES
    ls = [(x, y) for y in range(sz) for x in range(sz)]
    ls.remove(target)
    for pos in ls:
        start_cell = maze.cells[*pos]
        model.play(start_cell=start_cell, render_mode= Render.MOVES)

plt.show()
