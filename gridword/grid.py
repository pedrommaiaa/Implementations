import numpy as np
from gym.envs.toy_text import discrete


class gridWorld(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the bottom right corner.

    For example, a 5x5 grid looks as follows:

    0 0 0 0 0
    0 0 0 0 0
    0 0 X 0 0 
    0 0 0 0 0
    0 0 0 0 T

    x is your position and T are the two terminal states.

    You can take actions in each direction [0=up, 1=Right, 2=Down, 3=Left].
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at eachs tep until you reach a terminal state.
    """
    
    def __init__(self, shape=[5, 5]):

        self.shape = shape

        nS = np.prod(shape)
        self.actions = [0, 1, 2, 3]
        nA = len(self.actions)

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(nA)}

            is_done = lambda s: s == (nS - 1)
            reward = 0.0 if is_done(s) else -1.0

            # We're stuck in a terminal state
            if is_done(s):
                P[s][self.actions[0]] = [(1.0, s, reward, True)]
                P[s][self.actions[1]] = [(1.0, s, reward, True)]
                P[s][self.actions[2]] = [(1.0, s, reward, True)]
                P[s][self.actions[3]] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][self.actions[0]] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][self.actions[1]] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][self.actions[2]] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][self.actions[3]] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(gridWorld, self).__init__(nS, nA, P, isd)

    def display(self):
        """ Display the current grid layout """ 
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                print('X', end=' ')
            elif s == self.nS - 1:
                print('T', end=' ')
            else:
                print('0', end=' ')

            if x == self.shape[1] - 1:
                print()

            it.iternext()


if __name__ == "__main__":

    env = gridWorld()

    env.display()
