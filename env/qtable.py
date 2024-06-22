"""
Project: WGU - CS Capstone - C964
File: env/qtable.py
Author: VD
Date: 6/22/2024
Description:

This file defines the QModel class used to train Robot's navigation system.
"""

import logging
import random
from datetime import datetime
import numpy as np
from .common import *


class QModel:
    """
    This class implements a Q-learning model for navigating an environment represented as a maze.

    The model learns to navigate the maze by associating a Q-value with each state-action pair.
    The Q-value represents the expected future reward of taking a specific action in a given state.
    The model employs exploration to balance between exploiting learned knowledge (high Q-value actions)
    and trying new actions to potentially discover better paths.
    """

    default_check_convergence_every = 5  # by default check for convergence every # episodes

    def __init__(self, environment, **kwargs):
        """
        Initializes the Q-learning model with the provided maze environment.

        Args:
            environment (Robot): An instance of the Robot class.
        """

        self.environment = environment
        self.name = "QTableModel"

        self.cells = self.environment.maze.cells
        self.minimum_reward = -0.5 * self.cells.size
        self.total_reward = 0

        self.Q = dict()  # table with Q per (state, action) combination
        for _ in self.cells:
            for cell in _:
                for action in Direction:
                    self.Q[(cell.pos, action)] = 0.0 if cell.adj[action] != None else -99.99

    
    def q(self, state):
        """
        Returns a NumPy array containing the Q-values for all possible actions in a given state.

        Args:
            state: A representation of the current state within the maze environment.

        Returns:
            np.ndarray: A NumPy array of Q-values corresponding to each available action.
        """
        
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])

    def predict(self, state):
        """
        Chooses an action to take based on the learned Q-values, incorporating exploration.

        The action with the highest Q-value is selected with a probability of (1 - exploration_rate).
        Otherwise, a random action from the available actions is chosen to encourage exploration.

        Args:
            state: A representation of the current state within the maze environment.

        Returns:
            int: The chosen action index from the environment's available actions.
        """
        q = self.q(state)

        logging.debug("q[] = {}".format(q))

        actions = np.nonzero(q == np.max(q))[0]
        return random.choice(actions)
    
    def play(self, start_cell, render_mode = Render.NOTHING):
        """
        Runs a gameplay episode starting from a specified cell within the maze, with an optional rendering mode.

        The model interacts with the environment, taking actions and learning from rewards received,
        until a terminal state (win or lose) is reached.

        Args:
            start_cell (Cell): The starting cell for the gameplay episode.
            render_mode (Render, optional): The rendering mode to visualize the gameplay. Defaults to Render.NOTHING.

        Returns:
            tuple: A tuple containing the terminal status (WIN or LOSE) and the episode path as a list of states.
        """

        self.total_reward = 0
        self.environment.render_mode = render_mode
        state = self.environment.reset(start_cell, [self.target_cell])
        path = [state]
        while True:
            action = self.predict(state=state)
            state, reward, status = self.step(action)
            path.append(state)
            if status in (Status.WIN, Status.LOSE):
                return status, path

    def check_win_all(self):
        """
        Evaluates the win rate from all cells in the check_list.

        This function iterates through the cells in the `check_list` (or all non-target cells if check_list is None)
        and plays a game starting from each cell. It then calculates the win rate (percentage of games won)
        across all these starting positions. This is used to assess convergence during training.

        Returns:
            tuple: A tuple containing a boolean indicating if all cells achieved wins (True) and the overall win rate (float).
        """
        render_mode = self.environment.render_mode 
        win = 0
        lose = 0

        for cell in self.check_list:
            status, _ = self.play(cell)
            if status == Status.WIN:
                win += 1
            else:
                lose += 1

        logging.info("[convergence] won: {} | lost: {} | win rate: {:.2f} %".format(win, lose, win / (win + lose) * 100))

        result = True if lose == 0 else False

        self.environment.render_mode = render_mode
        return result, win / (win + lose)

    reward_target = 100.0  # reward for reaching the target cell
    penalty_move = -0.05  # penalty for a move which did not result in finding the exit cell
    penalty_visited = -1.00  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -1.00  # penalty for trying to enter an occupied cell or moving out of the maze

    def step(self, action):
        """
        Executes an action within the environment
        Args:
            action (int): The action index to be taken by the agent.

        Returns:
            tuple: A tuple containing the new state, the received reward, and the current agent status.
        """
        current_cell = self.environment.current_cell
        reward = self.penalty_move

        possible = list(filter(lambda d: current_cell.adj[d] is not None, Direction))
        if action in possible:
            next_cell = self.environment.maze.cells[*current_cell.adj[action]]
            if next_cell.pos in self.environment.views.keys() and self.environment.views[next_cell.pos] == View.VISITED:
                reward = self.penalty_visited

            if next_cell == self.target_cell:
                reward = self.reward_target
        else:
            reward = self.penalty_impossible_move

        self.total_reward += reward
        state = self.environment.step(action)
        return state, reward, self.status()

    def status(self):    
        """
        Checks the current status
        """
        if self.target_cell == self.environment.current_cell:
            return Status.WIN
            
        if self.total_reward < self.minimum_reward:
            return Status.LOSE
        
        return Status.PLAYING
    
    def shortest_path(self, cells):
        m = 999999999999999999
        shortest_path = []
        nearest_cell = self.target_cell
        for cell in cells:
            status, path = self.play(start_cell= cell)
            if status == Status.WIN and len(path) < m:
                shortest_path = path
                nearest_cell = cell
                m = len(path)
                
        return nearest_cell.pos, shortest_path


    def train(self, target, render_mode = Render.NOTHING, stop_at_convergence=False, **kwargs):
        """
        Trains the Q-learning model on the provided maze environment.

        The model iterates through training episodes, starting from various cells and learning
        to reach the target cell efficiently. Hyperparameters such as learning rate, exploration rate,
        and discount factor can be adjusted to influence the training process.

        Args:
            target (tuple): The coordinates of the target cell to be reached within the maze.
            render_mode (Render, optional): The rendering mode to visualize the training process. Defaults to Render.NOTHING.
            stop_at_convergence (bool, optional): Flag to stop training early if all cells in `check_list` reach the target consistently. Defaults to False.

        **kwargs:
            discount (float, optional): Discount factor (gamma) used in Q-learning update rule to balance immediate vs. future rewards. Defaults to 0.9.
            exploration_rate (float, optional): Probability of choosing a random action for exploration during training. Defaults to 0.1.
            exploration_decay (float, optional): Rate at which exploration rate decreases over time (1 - exploration_decay). Defaults to 0.995.
            learning_rate (float, optional): Learning rate (alpha) used in Q-learning update rule to adjust Q-values based on new experiences. Defaults to 0.1.
            episodes (int, optional): The total number of training episodes to run. Defaults to 1000 (minimum 1).
            check_convergence_every (int, optional): Interval at which to check win rates from all cells in `check_list`. Defaults to the model's default check frequency.
            check_list (list, optional): A list of cell coordinates to use for convergence checks. If None, all non-target cells are used.

        Returns:
            tuple: A tuple containing the training history (cumulative reward per episode),
                win rate history per check interval, total training episodes, and training duration.
        """

        self.environment.render_mode = render_mode
        self.target_cell = self.environment.maze.cells[*target]
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.995)
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = max(kwargs.get("episodes", 1000), 1)
        check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)
        check_list = kwargs.get("check_list", None)
        
        logging.info('train {} | target: {}, render: {}, episodes: {}, check_convergence_every: {}, check_list: {}'
                     .format(self.name, target, render_mode.name, episodes, check_convergence_every, check_list if check_list is None or len(check_list) < 10 else len(check_list)))

        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()
        start_time = datetime.now()
        start_list = []
        for _ in self.environment.maze.cells:
            for cell in _:
                if cell != self.target_cell:
                    if cell.adj != [None, None, None, None]:
                        start_list.append(cell)
            
        bak_list = start_list.copy()
        self.check_list = start_list.copy() if check_list is None else [self.environment.maze.cells[*pos] for pos in check_list]

        # training starts here
        for episode in range(1, episodes + 1):
            if not start_list:
                start_list = bak_list.copy()
                random.shuffle(start_list)

            start_cell = start_list.pop()
            state = self.environment.reset(start_cell, [self.target_cell])
            self.total_reward = 0

            while True:
                
                
                discovered, visited = self.environment.possible_actions()
                if np.random.random() < exploration_rate:
                    action = random.choice(discovered if len(discovered) > 0 else visited) 
                else:
                    action = self.predict(state)


                next_state, reward, status = self.step(action)
                cumulative_reward += reward

                max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])
                self.Q[(state, action)] += learning_rate * (reward + discount * max_next_Q - self.Q[(state, action)])

                if status in (Status.WIN, Status.LOSE):  # terminal state reached, stop training episode
                    break

                state = next_state

            cumulative_reward_history.append(cumulative_reward)

            logging.debug("episode: {:d}/{:d} | status: {:4s} | e: {:.5f}"
                         .format(episode, episodes, status.name, exploration_rate))

            if episode % check_convergence_every == 0:
                w_all, win_rate = self.check_win_all()
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    logging.info("won from all cells in check_list, stop learning")
                    break

            exploration_rate *= exploration_decay

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time
