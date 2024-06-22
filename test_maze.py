"""
Project: WGU - CS Capstone - C964
File: test_maze.py
Author: VD
Date: 6/22/2024
Description:

Test Environment
    Environment render | fog ON / OFF
    Manual play: use arrow keys to control the robot
"""

from env.maze import Maze
from env.robot import Robot
from env.common import *
from random import randint
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

sz = 10

start = target = (randint(0,sz - 1), randint(0,sz - 1))
while target == start:
    target = (randint(0,sz - 1), randint(0,sz - 1))

maze = Maze(*Maze.generate(perfect_score = 95, w = sz, h = sz)) #seed = random
rob = Robot(maze = maze, fog = True) # or fog = False
rob.render()
rob.play(task = Task.NONE, start = start, targets = [target])

plt.show()