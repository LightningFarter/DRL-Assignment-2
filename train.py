import copy
import random
import math
import numpy as np

import time
import gym
import gym_bandits
import matplotlib.pyplot as plt

from student_agent import *

import pickle

from collections import defaultdict

opt = 1

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1, search_depth=2):
    final_scores = []
    success_flags = []
    
    print(f"alpha={alpha} gamma={gamma} epsilon={epsilon}")

    for episode in range(num_episodes):
        state = env.reset()
        previous_score = 0
        trajectory = []
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            if random.random() < epsilon:
                action = random.choice(legal_moves)
            else:
                _, action = expectimax_search(env, search_depth, approximator)
                if action is None or action not in legal_moves:
                    action = random.choice(legal_moves) if legal_moves else 0
                
            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = env.score
            trajectory.append((copy.deepcopy(state), incremental_reward))
            state = next_state
            max_tile = max(max_tile, np.max(state))
            
            # print(incremental_reward)

        # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.
        trajectory_length = len(trajectory)
        target = 0
        for i in reversed(range(trajectory_length - 1)):
            state, reward = trajectory[i]
            next_state, next_reward = trajectory[i+1]
            delta = target - approximator.value(next_state)
            approximator.update(next_state, delta, alpha)
            target = reward + approximator.value(next_state)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)
        
        

        if (episode + 1) % opt == 0:
            avg_score = np.mean(final_scores[-opt:])
            success_rate = np.sum(success_flags[-opt:]) / opt
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")

    return final_scores


if __name__ == "__main__":
    patterns_3_6 = [
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
    ]

    patterns_8_6 = [
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
        [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)]
    ]

    patterns_5_4 = [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 1), (0, 2), (1, 1), (1, 2)],
        [(1, 1), (1, 2), (2, 1), (2, 2)]
    ]

    patterns_2_4_2_6 = [
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(1, 0), (1, 1), (1, 2), (1, 3)]
    ]

    patterns_3_5 = [
        [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
        [(1, 2), (2, 2), (2, 3), (3, 1), (3, 2)],
        [(0, 1), (0, 2), (0, 3), (1, 1), (2, 1)]
    ]
    
    with open('approximator_p1_3_6_1000.pkl', 'rb') as f:
        approximator = pickle.load(f)

    # approximator = NTupleApproximator(board_size=4, patterns=patterns_3_6)

    env = Game2048Env()

    epis = 1000

    final_scores = td_learning(env, approximator, num_episodes=epis, alpha=0.1, gamma=1, epsilon=0, search_depth=1)
    
    epis = 2000
    
    with open(f'approximator_p1_3_6_{epis}_score.pkl', 'wb') as f:
        pickle.dump(final_scores, f)

    with open(f'approximator_p1_3_6_{epis}.pkl', 'wb') as f:
        pickle.dump(approximator, f)