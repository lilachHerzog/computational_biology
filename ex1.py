import random
from gc import collect
from inspect import getmembers
from operator import truediv
from random import Random
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from enum import Enum

from numpy import empty_like

percentage_of_fast_cells = 0.1
distance_of_fast_cells = 10
infection_probability_1 = 0.5
infection_probability_2 = 0.1
threshold = 0.1
recovery_time = 10
percent_of_initial_num_of_sicks = 0.5
generation = 0

class States(Enum):
    RECOVERED = 0
    HEALTHY = 1
    SICK = 2



class Movement(Enum):
    """
    The directions a cell can go to
    I chose prime numbers to account for the diagonals - multiplication of 2 dirrections
    so that even if 1 root direction of a diagonal is inaccessible the direction is inaccessible.
    """
    UP = 2
    RIGHT = 3
    DOWN = 5
    LEFT = 7
    STAY = 0


class Cell(object):
    """
    """
    x = None
    y = None
    state = None
    movement_capability = None
    sick_days = 0
    id = None

    def __init__(self, df):
        self.set_defaults(df)
        # self.__dict__.update(kwargs)

    def set_defaults(self, df):
        self.id = df['id']
        self.x = df['x']
        self.y = df['y']
        self.state = df['state']
        self.sick_days = df['sick_days']
        # self.update = False
        self.movement_capability = df['movement_capability']

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def generation(self, infection_probability):
        optional_moves, num_of_sick_neighbors = self.possible_movement( self.movement_capability)
        infection_probability *= num_of_sick_neighbors
        next_place = random.choice(optional_moves)
        self.move(next_place)
        change = self.infect(infection_probability)
        return self, change

    def infect(self, infection_probability):
        if self.state == States.RECOVERED.value:
            return 0
        if self.state == States.SICK.value:
            if self.sick_days == recovery_time:
                self.state = States.RECOVERED.value
                print("cell at {},{} recovered!!".format(self.x, self.y))
                return -1
            self.sick_days += 1
            return 0
        if random.random() < infection_probability:#Random.choices([0,1], weights=[1-infection_probability, infection_probability]):
            self.state = States.SICK.value
            print("cell at {},{} got sick".format(self.x, self.y))
            return 1
        return  0


    def move(self, movement):
        if movement == Movement.STAY:
            return
        # print("moved from ", self.x, self.y, end=" ")
        if movement % Movement.UP.value == 0:
            self.x += self.movement_capability
        elif movement % Movement.DOWN.value == 0:
            self.x -= self.movement_capability
        if movement % Movement.RIGHT.value == 0:
            self.y += self.movement_capability
        elif movement % Movement.LEFT.value == 0:
            self.y -= self.movement_capability
        # print("to ", self.x, self.y)


    def possible_movement(self, step_size):

        places = [(self.x - step_size, self.y),
                  (self.x, self.y - step_size),
                  (self.x+ step_size, self.y),
                  (self.x, self.y  + step_size),
                  (self.x, self.y)]
        neighbors = pd.DataFrame({  "places": places , "occupied": np.zeros(5).astype(bool)}, index=Movement._member_names_)
        prev_occupied = 0
        num_of_sick_neihbors = 0
        empty_neighbors = [Movement.STAY.value]
        for neighbor in neighbors.index:
            x, y = neighbors.loc[neighbor]["places"]

            if board.cell_place_in_range(x, y):
                direction = Movement[neighbor].value
                cell = board.get_cell_by_place(x, y)
                if cell.empty:
                    neighbors.loc[neighbor,"occupied"] = True
                    empty_neighbors.append(direction)
                    if prev_occupied:
                        empty_neighbors.append(direction * prev_occupied)
                    prev_occupied = direction
                else:
                    prev_occupied = 0
                    if cell.state.values[0] == States.SICK.value:
                        num_of_sick_neihbors += 1
        if neighbors["occupied"].values[0] and neighbors["occupied"].values[-1]:
            empty_neighbors.append(Movement[neighbors.index[0]].value * direction)

        return empty_neighbors, num_of_sick_neihbors


    def get_State(self):
        return self.state


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
        self.initiate_board()
        self.__dict__.update(kwargs)


    def set_defaults(self):
        self.hight = 10
        self.width = 10
        self.N = 30 # N < self.hight * self.width
        self.num_of_sick = {'0': 0}
        self.cells = pd.DataFrame(columns=[var for var in vars(Cell).keys() if Cell.__dict__[var] == None or Cell.__dict__[var] == 0])

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def generation(self):
        num_of_sick = self.num_of_sick[str(generation - 1)]
        percentage_of_sick =  num_of_sick/ self.N
        probability = infection_probability_1 if random.random() > percentage_of_sick else infection_probability_2
        for i in self.cells.index:
            old_cell = Cell(self.cells.loc[i])
            cell, change = old_cell.generation(probability)
            self.cells.loc[i,'x'] = cell.x
            self.cells.loc[i,'y'] = cell.y
            self.cells.loc[i,'state'] = cell.state
            self.cells.loc[i,'sick_days'] = cell.sick_days
            num_of_sick += change
        self.num_of_sick[str(generation)] =num_of_sick

    def cell_place_in_range(self,x, y):
        return x>=0 and y>=0 and x<self.width and y<self.hight

    def get_cell_by_place(self, x, y):
        return self.cells.loc[self.cells['x'] == x].loc[ self.cells['y'] == y]

    def initiate_board(self):
        for i in range(self.N):
            movement_capability = 1 if random.random()> percentage_of_fast_cells else distance_of_fast_cells
            if random.random()> (percent_of_initial_num_of_sicks):
                state = States.HEALTHY
            else:
                state = States.SICK
                self.num_of_sick[str(generation)] += 1
            while (True):
                row = random.randrange(0, self.width)
                column = random.randrange(0, self.hight)
                if self.cells.empty or self.cells.loc[self.cells["x"]==row].loc[self.cells["y"] == column].empty:
                    break
            cell = Cell({"id": i, "x": row, "y": column, "state":state.value, "movement_capability":movement_capability, "sick_days": 0})
            self.cells.loc[len(self.cells)] =cell.__dict__
            # self.board.loc[row,column] = cell.state


    def print(self):
        for row in range(self.hight):
            for column in range(self.width):
                cell = self.cells.loc[self.cells['x']==row].loc[self.cells['y']==column]
                if cell.empty:
                    print("*", end=" | ")
                else:
                    print(cell['state'].values[0], end=" | " )
            print("\n")


    def print_num_of_sick(self):
        plt.title("Line graph")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.plot(self.num_of_sick.keys(), self.num_of_sick.values(), color="red")
        plt.show()

    def scatter_plot(self):
        colors = {'0': 'blue', '1': 'green', '2':'red'}
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.title("generation {}:".format(generation))

        # Random data of 100×3 dimension

        # Scatter plot
        plt.scatter(self.cells['x'], self.cells['y'], c='red', alpha=[s/3 for s in self.cells['state']])

        # Display the plot
        plt.show()

if __name__ == '__main__':
    global board
    board = Board()
    for i in range(30):
        generation += 1

        board.generation()
        board.scatter_plot()
        # board.print()
    board.print_num_of_sick()
