"""
Project: WGU - CS Capstone - C964
File: test_robot_2.py
Author: VD
Date: 6/22/2024
Description:

Test Navigation System 2
    Task: Searching - fog: ON
"""

from env.maze import Maze
from env.robot import Robot
from env.common import *
from env.qtable import QModel
from random import randint
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

sz = 10

start = (randint(0,sz - 1), randint(0,sz - 1))
targets = []
for _ in range(10):
    target = start
    while target == start or target in targets:
        target = (randint(0,sz - 1), randint(0,sz - 1))
    targets.append(target)

maze = Maze(*Maze.generate(perfect_score = 95, w =sz, h=sz))
rob = Robot(maze = maze, fog=True)
rob.render()
paths = rob.play(start = start, targets= targets, task= Task.SEARCH)

plt.show()