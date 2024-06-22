"""
Project: WGU - CS Capstone - C964
File: env/robot.py
Author: VD
Date: 6/22/2024
Description:

This file defines the Robot class.
"""

from .common import *
from .maze import Maze, Cell
import matplotlib.pyplot as plt
import numpy as np
from .qtable import QModel
import random
import logging
from datetime import datetime

class Robot:
    """
    This class represents a robot that can navigate a maze.

    Attributes:
        render_mode (Render): The current rendering mode (NOTHING, MOVES, TRAINING)
        fog (bool): Whether fog of war is enabled (limited view)
        task (Task): The current task of the robot (NONE, SEARCH, EXHAUST)
        maze (Maze): The maze environment the robot is navigating.
        ax (matplotlib.pyplot.Axes): The matplotlib axis for rendering the maze (if applicable).
        current_cell (Cell): The current cell the robot is occupying.
        previous_cell (Cell): The previous cell the robot visited.
        views ({pos, View}): The vision of the robot 
    """

    def __init__(self, maze, fog = True):
        """
        Initializes a new Robot instance.

        Args:
            maze (Maze): The maze environment for the robot.
            fog (bool, optional): Whether fog layer is enabled (limited view). Defaults to True.
        """
        self.render_mode = Render.NOTHING
        self.fog = fog
        self.task = Task.NONE
        self.maze = maze
        self.ax = None
        self.reset()
        
    def reset_view(self):
        """
        Resets the robot's view of the maze.
        """

        self.views = dict()
        if self.fog:
            for dir in Direction:
                pos = self.current_cell.adj[dir]
                if pos is None: 
                    continue
                self.views[pos] = View.DISCOVERED
        else:
            for _ in self.maze.cells:
                for cell in _:
                    self.views[cell.pos] = View.DISCOVERED
        self.views[self.current_cell.pos] = View.VISITED

    def update_view(self, ax):
        """
        Updates the robot's view of the maze.

        Args:
            ax (matplotlib.pyplot.Axes): The matplotlib axis for rendering (if applicable).
        """

        for dir in Direction:
            pos = self.current_cell.adj[dir]
            if pos is None or pos in self.views.keys(): 
                continue
            self.views[pos] = View.DISCOVERED

            if ax and self.render_mode != Render.NOTHING:
                cell = self.maze.cells[*pos]
                cell.draw_cell(ax, color = 'gray')
                if cell in self.target_cells:
                    cell.draw_text(ax, '', fill_color = 'green')

        if self.views[self.current_cell.pos] != View.VISITED:
            self.views[self.current_cell.pos] = View.VISITED
            if ax and self.render_mode != Render.NOTHING:
                self.current_cell.draw_cell(ax, color = 'lightgreen')

        if ax and self.current_cell in self.target_cells:
            self.current_cell.draw_text(ax, 'x', fill_color = 'green')
        
        if ax and self.render_mode != Render.NOTHING:
            self.draw_path(ax)

    def draw_view(self, ax):
        """
        Renders the robot's view of the maze (based on fog of war).

        Args:
            ax (matplotlib.pyplot.Axes): The matplotlib axis for rendering.
        """

        for pos, v in self.views.items():
            if v in (View.DISCOVERED, View.VISITED):
                cell = self.maze.cells[*pos]
                cell.draw_cell(ax, color = 'lightgreen' if v == View.VISITED else 'gray')

        for _ in self.maze.cells:
            for cell in _:
                if cell == self.start_cell:
                    cell.draw_text(ax, 'Start', fill_color = 'red')
                if cell in self.target_cells:
                    if cell.pos in self.views.keys():
                        cell.draw_text(ax, '', fill_color = 'green')
    
    def possible_actions(self):
        """
        Returns a list of possible actions (directions) the robot can take 
        based on the current cell.

        Returns:
            tuple: A tuple containing two lists:
                - discovered (list): List of directions leading to discovered cells.
                - visited (list): List of directions leading to visited cells.
        """
        visited = []
        discovered = []
        for d in Direction:
            pos = self.current_cell.adj[d]
            if pos == None: continue

            if self.views[pos] == View.VISITED:
                visited.append(d)
            else:
                discovered.append(d)

        return discovered, visited
            
    def draw_path(self, ax):
        """
        Renders the path taken by the robot in the maze.

        Args:
            ax (matplotlib.pyplot.Axes): The matplotlib axis for rendering.
        """
        curr = np.array([*self.current_cell.pos]) + 0.5
        prev = np.array([*self.previous_cell.pos]) + 0.5

        ax.plot(*zip(*[prev, curr]), "bo-")
        ax.plot(*curr, "go")
        ax.get_figure().canvas.draw()
        ax.get_figure().canvas.flush_events()

    KEY_EVENTS = {
        'left': Direction.LEFT,
        'right': Direction.RIGHT,
        'up': Direction.UP,
        'down': Direction.DOWN
    }
    actions = [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]
    def on_key_press(self, event):
        """
        Handles keyboard events for user interaction with the robot (if applicable).

        Args:
            event (KeyEvent): The keyboard event object.
        """
        if self.task != Task.NONE:
            return

        if event.key not in Robot.KEY_EVENTS:
            return
        
        self.step(Robot.KEY_EVENTS[event.key])

    def step(self, action):
        """
        Takes a step in the given direction and updates the robot's state.

        Args:
            action (Direction): The direction to move in.

        Returns:
            tuple: The new position of the robot.
        """

        possible = list(filter(lambda d: self.current_cell.adj[d] is not None, Direction))
        if action in possible:
            next_cell = self.maze.cells[*self.current_cell.adj[action]]
            self.previous_cell = self.current_cell
            self.current_cell = next_cell
            self.update_view(self.ax)
        
        return self.current_cell.pos
    
    def step_to(self, pos):
        """
        Moves the robot directly to a adjacent cell position.

        Args:
            pos (tuple): The position of the target cell.

        Returns:
            tuple: The new position of the robot.
        """
        for d in Direction:
            if self.current_cell.adj[d] == pos:
                return self.step(d)
        return self.current_cell.pos

    def play(self, start, targets = [], task = Task.SEARCH, render_mode = Render.MOVES):
        """
        Starts playing the maze game.

        Args:
            start (tuple): The starting position of the robot.
            targets (list, optional): A list of target cell positions. Defaults to [].
            task (Task, optional): The task for the robot (NONE, SEARCH). Defaults to Task.SEARCH.
            render_mode (Render, optional): The rendering mode (NOTHING, MOVES). Defaults to Render.MOVES.
        """

        self.render_mode = render_mode
        if task == Task.SEARCH:
            if self.fog:
                return self.search_fog(start, targets)
            else:
                return self.search(start, targets)
        elif task == Task.NONE:
            start_cell = self.maze.cells[*start]
            target_cells = [self.maze.cells[*target]  for target in targets]
            self.reset(start_cell, target_cells)

    
    def search_fog(self, start, targets):
        """
        Searches for target cells in a maze with fog enabled.

        Args:
            start (tuple): The starting position of the robot.
            targets (list): A list of target cell positions.

        The main loop iterates until all targets are found:
            - Checks for remaining targets using check_targets.
            - If no targets remain, search is complete.
            - Finds adjacent discovered cells.
            - If an adjacent target is discovered, the robot moves directly to it and updates the internal maze.
            - Otherwise, the robot moves to a random adjacent discovered cell and updates the internal maze.

            - If all adjacent cells are visited, gathers all discovered target cells and discovered cells
            - Trains a Q-learning model.
            - Finds and process the shortest path to the next target cells or discovered cells (if no targets).
        """

        logging.info('task: SEARCH | fog: ON  | start: {} | targets: {}'.format(start, targets))
        start_time = datetime.now()
        step_count = 0

        start_cell = self.maze.cells[*start]
        target_cells = [self.maze.cells[*target]  for target in targets]
        self.reset(start_cell, target_cells)

        def update_cells(cells):
            view_keys = self.views.keys()
            for d in Direction:
                
                cells[*self.current_cell.pos].adj[d] = self.current_cell.adj[d]
                
                if self.current_cell.adj[d] is None:
                    continue
                
                c = self.maze.cells[*self.current_cell.adj[d]]
                for dd in Direction:
                    if c.adj[dd] is not None and c.adj[dd] in view_keys:
                        cells[*c.pos].adj[dd] = c.adj[dd]

        cells = np.array([[Cell((x,y), adj=[None, None, None, None]) for y in range(self.maze.height)] for x in range(self.maze.width)])
        maze = Maze(cells)
        update_cells(cells)

        def check_targets(targets):
            view_keys = self.views.keys()

            return [target for target in targets if target not in view_keys or self.views[target] != View.VISITED]
            
        while True:
            targets = check_targets(targets)
            if len(targets) == 0:
                break 

            discovered = []
            next = None
            for d in Direction:
                pos = self.current_cell.adj[d]

                if pos is None:
                    continue
                if self.views[pos] == View.DISCOVERED:
                    if pos in targets:
                        next = pos
                        break
                    discovered.append(pos)

            if next != None:
                self.step_to(next)
                step_count += 1
                update_cells(cells)
                continue

            if len(discovered) > 0:
                next = random.choice(discovered)
                self.step_to(next)
                step_count += 1
                update_cells(cells)
                continue
            
            discovered_cells = []
            next_target_cells = []

            for pos, view in self.views.items():
                if view == View.DISCOVERED:
                    if pos in targets:
                        next_target_cells.append(self.maze.cells[*pos])
                    else:
                        discovered_cells.append(self.maze.cells[*pos])

            next_cells = next_target_cells if len(next_target_cells) > 0 else discovered_cells
            rob = Robot(maze = maze, fog = False)
            model = QModel(rob)
            round = len(self.views)
            model.train(target = self.current_cell.pos, discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=round*10,
                            stop_at_convergence=True, check_convergence_every = round, check_list = [c.pos for c in next_cells])
            _, shortest_path = model.shortest_path(next_cells)
            if len(shortest_path) > 0:
                shortest_path.pop(-1)
                shortest_path.reverse()
                
                for next in shortest_path:
                    self.step_to(next)
                    step_count += 1
                    update_cells(cells)

        logging.info('task completed | time spent: {} | steps: {}'.format(datetime.now() - start_time, step_count))

    def search(self, start, targets):
        """
        Searches for target cells in a maze with fog disabled (full view).

        Args:
            start (tuple): The starting position of the robot.
            targets (list): A list of target cell positions.

        The main loop iterates until all targets are found:
            Check the number of remaining targets.
            Trains a Q-learning model.
            Finds and process the shortest path to remaining targets.
        """

        logging.info('task: SEARCH | fog: OFF | start: {} | targets: {}'.format(start, targets))
        start_time = datetime.now()
        step_count = 0
        
        start_cell = self.maze.cells[*start]
        target_cells = [self.maze.cells[*target]  for target in targets]
        self.reset(start_cell, target_cells)

        while len(targets) > 0:
            logging.info('search nearest target | len(targets): {}'.format(len(targets)))
            rob = Robot(maze = self.maze)
            model = QModel(rob)
            round = self.maze.width * self.maze.height
            model.train(target = start, discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=round*10,
                                stop_at_convergence=True, check_convergence_every = round, check_list = targets)

            target_cells = [self.maze.cells[*target] for target in targets]
            nearest_target, shortest_path = model.shortest_path(target_cells)
            
            logging.info('found nearest target: {} -> {}'.format(shortest_path[-1], shortest_path[0]))

            shortest_path.pop(-1)
            shortest_path.reverse()

            for next in shortest_path:
                self.step_to(next)
                step_count += 1

            start = nearest_target
            targets.remove(nearest_target)
        
        logging.info('task completed | time spent: {} | steps: {}'.format(datetime.now() - start_time, step_count))

    def render(self):
        """
        Renders the maze and the robot's view.
        """

        if self.ax is None:
            self.fig, self.ax = plt.subplots()
            self.fig.canvas.manager.set_window_title('Vu Dao - WGU CS Capstone 2024 ' + self.maze.info)
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ax.clear()
        self.maze.draw_maze(self.ax, alpha = 0.2 if self.fog else 1.0)
        self.draw_view(self.ax)
        plt.show(block=False)

    def reset(self, start_cell = None, target_cells = []):
        """
        Resets the robot's state to the starting position and clears the view.

        Args:
            start_cell (tuple, optional): The starting position of the robot. Defaults to None.
            target_cells (list, optional): A list of target cell positions. Defaults to [].

        Returns the position of the starting cell.
        """
        self.start_cell = start_cell if start_cell is not None else self.maze.cells[0, 0]
        self.previous_cell = self.current_cell = self.start_cell
        self.target_cells = target_cells
        self.reset_view()

        if self.render_mode != Render.NOTHING:
            self.ax.clear()
            self.maze.draw_maze(self.ax, alpha = 0.2 if self.fog else 1.0)
            self.draw_view(self.ax)

        return self.start_cell.pos
    
