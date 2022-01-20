import copy
import random
import numpy as np

class Box():
    # constructor of cell
    def __init__(self, x, y, isTrap, isBegin, isEnd, lstMove, K):
        self.x = x
        self.y = y
        self.isTrap = isTrap
        self.isBegin = isBegin
        self.isEnd = isEnd
        self.lstMove = lstMove
        self.reward = -1

        self.value = np.random.random(1)[0]
        self.Q = []
        self.K = K

        if isTrap == True:
            self.reward = -2 * (self.K - 1)

    #initialize Q_function
    def init_q(self):
        self.Q = {
            'N' : -100,
            'S' : -100,
            'E' : -100,
            'W' : -100
        }

        for i, val in enumerate(self.lstMove):
            if self.x == self.K - 1 and self.y == self.K - 1: 
                self.Q[val] = np.zeros(1)[0]
            else: 
                self.Q[val] = np.random.random(1)[0]

    # gets coord according to an action 
    def get_coord(self, action):
        next_x = copy.deepcopy(self.x)
        next_y = copy.deepcopy(self.y)

        try:
            b=self.lstMove.index(action)
        except ValueError:
            return next_x, next_y

        switch={
            'N' : -1,
            'S' : 1,
            'E' : -1,
            'W' : 1
        }

        result = switch.get(action)

        if(action == 'N' or action == 'S'):
            next_x += result
        elif(action == 'E' or action == 'W'):
            next_y += result
            
        return next_x, next_y

class Board():
    def __init__(self, K):
        self.K = K
        self.move = ['N', 'S', 'E', 'W']
        self.board = []
    
    # initialiaze grid for TD learning and value iteration
    def create_board(self):
        for x in range(self.K):
            tmp = []
            for y in range(self.K):
                move = self.init_move(x, y)
                isTrap = self.is_trap(x, y)
                box = Box(x, y, isTrap, self.is_begin(x, y), self.is_end(x, y), move, self.K)             
                box.init_q()
                tmp.append(box)
            self.board.append(tmp)
        self.board[self.K - 1][self.K - 1].reward = 2 * (self.K - 1)
    
    # define trap on the grid
    def is_trap(self, x, y):
        range_one = list(range(0, 8))
        range_two = list(range(4, 12))

        if(x == 3 and y in range_one):
            return True
        if(x == 7 and y in range_two):
            return True
        return False

    # beginning of the grid
    def is_begin(self, x, y):
        if x == 0 and y == 0:
            return True
        return False
    
    # end of the grid
    def is_end(self, x, y):
        if(x == self.K - 1 and y == self.K - 1):
            return True
        return False

    # initialize authorized moves
    def init_move(self, x, y):
        move = copy.deepcopy(self.move)
        
        if(y == 0):
            move.pop(2)
        elif(y == self.K - 1):
            move.pop(3)

        if(x == 0):
            move.pop(0)
        elif(x == self.K - 1):
            move.pop(1)

        return move

    # return a S for TD learning
    def give_me_an_s(self):
        return random.choice(random.choice(self.board))