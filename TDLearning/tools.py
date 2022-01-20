import board as m
import numpy as np
import random
import time
import os

match_direction = {
    0 : '^',
    1 : '!',
    2 : '<',
    3 : '>'
}

match_direction2 = {
    0 : 'N',
    1 : 'S',
    2 : 'E',
    3 : 'W'
}

def extract_policy_q(b):
    policy = []
    for i, ligne in enumerate(b.board):
        tmp = []
        for j, item in enumerate(ligne):
            char = match_direction[np.argmax(list(item.Q.values()))]
            if item.isTrap:
                char = '¤'
            if i == 0 and j == 0:
                char = 'b' + char
            if i == 11 and j == 11:
                char = 'e' + char
            tmp.append(char)
        policy.append(tmp)
    return np.asarray(policy)

def show_movement_agent(b, K):
    mat = []
    for i, ligne in enumerate(b.board):
        tmp = []
        for j, item in enumerate(ligne):
            char = ' '
            if item.isTrap:
                char = '¤'
            tmp.append(char)
        mat.append(tmp)

    mat = np.asanyarray(mat)
    state = b.board[0][0]
    x, y = 0, 0
    mat[x][y] = '*'

    while(state.isEnd == False):
        os.system('clear')
        move = match_direction2[np.nanargmax(list(state.Q.values()))]
        nx, ny = state.get_coord(move)
        state = b.board[nx][ny]
        mat[nx][ny] = '*'
        if nx == x and ny == y:
            break
        print(f'{mat}')
        x, y = nx, ny
        time.sleep(0.1)

  

def epsilon_greedy(epsilon, s):
    if np.random.rand() < epsilon:
        action =  list(s.Q.keys())[np.nanargmax(list(s.Q.values()))]
    else:
        action = random.choice(list(s.lstMove))
    return action