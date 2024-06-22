"""
Project: WGU - CS Capstone - C964
File: env/common.py
Author: VD
Date: 6/22/2024
Description: 

This file defines common enumerations used throughout the environment code.
"""

__all__ = ['Direction', 'Render', 'Status', 'Task', 'View']

from enum import  IntEnum

class Direction(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

class Render(IntEnum):
    NOTHING = 0
    TRAINING = 1
    MOVES = 2

class Status(IntEnum):
    WIN = 0
    LOSE = 1
    PLAYING = 2

class Task(IntEnum):
    NONE = 0
    SEARCH = 1

class View(IntEnum):
    NONE = 0
    DISCOVERED = 1
    VISITED = 2
