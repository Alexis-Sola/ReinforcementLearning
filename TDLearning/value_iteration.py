import board as m
import numpy as np
import tools

# match index with direction
match_direction = {
    0 : '^',
    1 : '!',
    2 : '<',
    3 : '>'
}

# list of actions
actions = ['N', 'S', 'E', 'W']

# return the values matrix computed
def get_values_matrix(list):
    mat = np.zeros((len(list), len(list)))
    for i, ligne in enumerate(list):
        for j, item in enumerate(ligne):
            mat[i][j] = round(item.value, 2)
    return mat

# compute values for each state and return the highter 
def value_function(state, board, gamma):

    # list of values corresponding with actions
    value_actions = [np.nan, np.nan, np.nan, np.nan]

    actions = {
                'N' : np.nan,
                'S' : np.nan,
                'E' : np.nan,
                'W' : np.nan
            }

    # iterate on all actions
    for i, val in enumerate(actions):
        #val_tmp = 0
        #for j, a in enumerate(state.lstMove):
            # ignore forbidden moves
        if val not in state.lstMove:
            continue

        x, y = state.get_coord(val)
        next_state = board.board[x][y]
        # compute value 
        val_tmp = (state.reward + (gamma * next_state.value))

            # gets all values of possible move
        value_actions[i] = val_tmp 
    
    #return val_tmp
    return np.nanmax(value_actions)

# convert best values into actions
def compute_best_actions(board, K):
    best_actions = np.zeros((K, K))
    # for all states
    for i in range(K):
        for j in range(K):
            actions = {
                'N' : np.nan,
                'S' : np.nan,
                'E' : np.nan,
                'W' : np.nan
            }
            state = board.board[i][j]

            # gets all values of possible move
            for a in state.lstMove:
                # if a not in actions.keys():
                #     actions[a] = np.NaN

                next_x, next_y = state.get_coord(a)
                actions[a] = board.board[next_x][next_y].value #matrix_value[next_x][next_y]
                #actions[a] = next_value

            # gets the index of the best move
            best_actions[i][j] = np.nanargmax(list(actions.values()))

    return best_actions

# algorithm of value iteration
def value_iteration(max_iter, board, gamma, K):  
    delta = 0
    cpt = 0
    for iter in range(max_iter):
    #while(delta < 0.01):
        # for all states
        for i in range(K):
            for j in range(K):
                current_state = board.board[i][j]
                # gets new value
                old = current_state.value
                v_pi = value_function(current_state, board, gamma)               
                current_state.value = v_pi

                delta = max(delta, abs(old - current_state.value))

    # gets matrix of values
    best_values = get_values_matrix(board.board)
    # gets best actions according to matrix's values
    best_actions = compute_best_actions(board, K)

    return best_values, best_actions, board

# print best stationnary policy in terminal 
def extract_policy(indexes, b):
    tab = []
    for i,ligne in enumerate(indexes):
        tmp = []
        for j, item in enumerate(ligne):
            char = match_direction[item]
            state = b[i][j]
            if state.isTrap:
                char = '¤'
            if i == 0 and j == 0:
                char = 'b' + char
            if i == 11 and j == 11:
                char = 'e' + char
            tmp.append(char)
        tab.append(tmp)
    # flip the matrix to begin at the bottom
    return np.asarray(tab)


if __name__ == "__main__":
    K = 12
    b = m.Board(K)
    b.create_board()

    values, indexes, b = value_iteration(100, b, 0.3, K)
    print(values)
    print(extract_policy(indexes, b.board))
