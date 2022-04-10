import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from enum import Enum
class States(Enum):
    HEALTHY = 0
    SICK = 1
    RECOVERED = 2

class Cell(object):
    """
    """

    def __init__(self, **kwargs):
        self.set_defaults(self, **kwargs)
        self.__dict__.update(kwargs)

    def set_defaults(self, position, state, range):
        self.position = position
        self.state = state
        self.update = False
        self.range = range

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def update(self, N):
        if self.update:
            self.N += 1
        if self.N == N:
            self.N = 0




class Board(object):
    '''
    Implement the system and look for a combination of the parameters described above that will result in the behavior of waves (an increase and decrease in the number of patients), at least 3 times during the life of the simulation.
    The parameters are:
    N - number of creatures
    D - number of cells infected at start time
    R - percentage of creatures that can move faster
    X - number of generations until recovery
    P_1, P_2 - probability of infection during high and low infections
    T - threshold value for the change of P as a function of the disease state
    '''

    def __init__(self, **kwargs):
        self.set_defaults()
        self.__dict__.update(kwargs)

    def set_defaults(self):
        self.size = (200,200)
        # self.N = 6

        # self.board = plt.table()
        self.board = pd.DataFrame(data=np.random.randint(self.N, size=(self.hight, self.width)))
    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
    def initiate_board(self):
        np.random.randint(self.N, N = self.N, size=(self.width, self.hight))
        for row in range(board.shape[1]):
            for column in range(board.shape[0]):


    #
    # def print_board(self):
    #     for i in range(self.width):
    #         for i in range(self.hight):
    #             self.board[i,j]


if __name__ == '__main__':
    board = Board()


