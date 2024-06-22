"""
Project: WGU - CS Capstone - C964
File: env/maze.py
Author: VD
Date: 6/22/2024
Description:

This file defines the Maze class used to represent the environment.
"""

import random
import numpy as np
import logging
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from .common import *


class Cell:
    """
    Cell class represents a single cell within the maze environment.

    Attributes:
        pos (tuple): (x, y) coordinates of the cell.
        adj (list): List representing adjacent cells in each direction (up, down, left, right). None indicates a wall.
    """
    def __init__(self, pos, adj = [None, None, None, None]):
        self.pos = pos
        self.adj = adj

    def __eq__(self, other):
        """
        Defines equality comparison for cells based on their position.
        """
        return self.pos == other.pos

    def draw_text(self, ax, text, text_color = 'white', fill_color = 'red'):
        """
        Draws text within the cell on a matplotlib axis.

        Args:
            ax (matplotlib.axes._axes.Axes): Matplotlib axis object for drawing.
            text (str): Text to be displayed within the cell.
            text_color (str, optional): Color of the text. Defaults to 'white'.
            fill_color (str, optional): Fill color of the cell. Defaults to 'red'.
        """
        (x, y) = self.pos
        rectangle = plt.Rectangle(xy=(x + 0.1, y + 0.1),
                    width=0.8, height=0.8,
                    fill=True,
                    fc=fill_color)
        
        ax.add_patch(rectangle)
        ax.text(x + 0.5, y + 0.5, text, ha="center", va="center", color=text_color)

    def draw_cell(self, ax, color = 'gray', alpha = 1.0):
        """
        Draws the cell on a matplotlib axis.

        Args:
            ax (matplotlib.axes._axes.Axes): Matplotlib axis object for drawing.
            color (str, optional): Color of the cell walls. Defaults to 'gray'.
            alpha (float, optional): Transparency of the cell. Defaults to 1.0 (opaque).
        """
        (x, y) = self.pos
        rectangle = plt.Rectangle(xy=(x, y),
                    width=1, height=1,
                    fill=True,
                    fc=color, 
                    alpha = alpha)
        ax.add_patch(rectangle)
        edges = [
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((0, 1), (1, 1)),
            ((0, 0), (1, 0))
        ]

        for d in Direction:
            if self.adj[d] is None:
                e = edges[d]
                xs = [x + e[0][0], x + e[1][0]]
                ys = [y + e[0][1], y + e[1][1]]
                ax.plot(xs, ys, linewidth=3, color='black')


class Maze:
    """
    Maze class represents the environment for the agent. It generates a random maze
    and provides methods to interact with the maze structure.
    """

    @staticmethod
    def generate(w = 10, h = 10, start = (0, 0), seed = random.randrange(10000000), perfect_score = 100):
        """
        Generates a random maze with specified parameters.

        Args:
            w (int, optional): Width of the maze. Defaults to 10.
            h (int, optional): Height of the maze. Defaults to 10.
            start (tuple, optional): Starting position (x, y). Defaults to (0, 0).
            seed (int, optional): Random seed for maze generation. Defaults to a random value.
            perfect_score (int, optional): Threshold used to control the randomness of perfect maze walls during generation. Defaults to 100.

        Returns:
            tuple: A tuple containing a numpy array of Cells representing the maze and an information string about the maze generation.
        """

        logging.info('Maze generating... seed: {}, perfect_score: {}, width: {}, height: {} '.format(seed, perfect_score, w, h))
        
        adj = {(x, y): [None, None, None, None] for x in range(w) for y in range(h)}
        vis = [[0] * w + [2] for _ in range(h)] + [[2] * (w + 1)]
        ver = [["|   "] * w + ['|'] for _ in range(h)] + [[]]
        hor = [["+---"] * w + ['+'] for _ in range(h + 1)]

        def walk(x, y, seed):
            vis[y][x] = 1

            d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
            random.Random(seed).shuffle(d)
            seed = random.Random(seed).randrange(10000000)
            for (xx, yy) in d:
                b_walk = True
                if vis[yy][xx] == 2:
                    continue
                elif vis[yy][xx]: 
                    score = random.Random(seed).randrange(100)
                    seed = random.Random(seed).randrange(10000000)
                    if score < perfect_score:
                        continue
                    b_walk = False # break perfect maze walls but dont move

                if xx == x: 
                    hor[max(y, yy)][x] = "+   "
                    adj[(x, y)][Direction.UP if yy > y else Direction.DOWN] = (xx, yy)
                    adj[(xx, yy)][Direction.DOWN if yy > y else Direction.UP] = (x, y)

                if yy == y: 
                    ver[y][max(x, xx)] = "    "
                    adj[(x, y)][Direction.RIGHT if xx > x else Direction.LEFT] = (xx, yy)
                    adj[(xx, yy)][Direction.LEFT if xx > x else Direction.RIGHT] = (x, y)
                
                if b_walk:
                    walk(xx, yy, seed)

        walk(*start, seed)

        s = ""
        hor.reverse()
        ver.reverse()
        for (a, b) in  zip(ver, hor):
            s += ''.join(a + ['\n'] + b + ['\n'])
        logging.info(s)

        cells = np.array([[Cell((x,y), adj[(x,y)]) for y in range(h)] for x in range(w)])
        info = '[Maze - seed: {} | perfect_score: {} | width: {} | height: {}]'.format(seed, perfect_score, w, h)
        return cells, info
    

    def __init__(self, cells, info = ''):
        """
        Initializes a Maze object.

        Args:
            cells (numpy.array): 2D numpy array of Cell objects representing the maze structure.
            info (str, optional): Information string about the maze generation. Defaults to ''.
        """
        self.cells = cells
        self.info = info
        self.width, self.height = self.cells.shape

    def draw_maze(self, ax, alpha = 1.0):
        """
        Draws the maze on a provided matplotlib axis 'ax' with a transparency level 'alpha'.

        Args:
            ax (matplotlib.axes._axes.Axes): Matplotlib axis object for drawing.
            alpha (float, optional): Transparency level. Defaults to 1.0.
        """
        for x in range(self.cells.shape[0]):
            for y in range(self.cells.shape[1]):
                self.cells[x,y].draw_cell(ax, alpha = alpha)

        rectangle = plt.Rectangle(xy=(0, 0),
                    width=self.width, height=self.height,
                    fill=True,
                    fc='black', 
                    alpha = 1-alpha)
        ax.add_patch(rectangle)
        
        ax.set_xticks(np.arange(0.5, self.width + 1, step=1))
        ax.set_yticks(np.arange(0.5, self.height + 1, step=1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal', 'box')