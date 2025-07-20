import argparse
import numpy as np
import random
import pygame
from collections import defaultdict
from student_agent import *  # This should contain your modified Game2048Env class
import time

# Initialize Pygame globally
pygame.init()
window_size = (800, 800)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("2048 AI Agent")

def run_episode(env: Game2048Env, render=False, render_interval=50):
    state = env.reset()
    total_reward = 0
    steps = 0
    max_tile = 0

    while True:
        # Allow closing window during game loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Render using Pygame only (no saving)
        if render and (steps % render_interval == 0):
            env.pygame_render(screen)

        action = get_action(state, env.score)
        next_state, reward, done, _ = env.step(action)

        total_reward += reward
        steps += 1
        max_tile = max(max_tile, np.max(next_state))
        state = next_state

        if done:
            env.pygame_render(screen)
            time.sleep(2)
            break

    return env.score, max_tile, steps

if __name__ == "__main__":
    random.seed(0)
    for i in range(100):
        env = Game2048Env(window_size=window_size)  # Pass the window size to support Pygame font sizing
        env.reset()
        print(run_episode(env, render=True, render_interval=1))
    model = CPPModel('2048.bin')
