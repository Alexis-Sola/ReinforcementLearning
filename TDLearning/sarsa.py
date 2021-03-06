import board as m
import numpy as np
import tools

def sarsa(b, episodes, max_step, epsilon, gamma, alpha):
    reward_list = []
    for e in range(episodes):
        total_reward = 0
        s = b.give_me_an_s()
        a = tools.epsilon_greedy(epsilon, s)
        t = 0

        while(True):
            t += 1
            x, y = s.get_coord(a)
            next_state = b.board[x][y]
            total_reward += next_state.reward
            next_a = tools.epsilon_greedy(epsilon, next_state)

            if s.isTrap or t == max_step - 1 or s.isEnd:
                s.Q[a] += alpha * (next_state.reward - s.Q[a])
                reward_list.append(total_reward)
                break
            
            s.Q[a] += alpha * (next_state.reward + gamma * (next_state.Q[next_a] - s.Q[a]))
            
            s = next_state
            a = next_a
        print(f'Training model : {(e * 100) / episodes} %', end='\r')
    return reward_list, b

if __name__ == "__main__":
    K = 12
    b = m.Board(K)
    b.create_board()

    alpha = 0.1
    gamma = 0.7
    epsilon = 0.9
    episodes = 5000
    max_steps = 10000
    timestep_reward, new_board = sarsa(b, episodes, max_steps, epsilon, gamma, alpha)
    tools.show_movement_agent(b, K)
    print(tools.extract_policy_q(new_board))
