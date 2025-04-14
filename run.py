import argparse
import numpy as np
# from tqdm import tqdm
from collections import defaultdict
from student_agent import *

def run_episode(env, render=False, render_interval=50):
    state = env.reset()
    total_reward = 0
    steps = 0
    max_tile = 0
    
    while True:
        if render and (steps % render_interval == 0):
            env.render(action=None, savepath=f"frame_{steps:04d}.png")
        
        action = get_action(state, env.score)
        next_state, reward, done, _ = env.step(action)
        
        total_reward += reward
        steps += 1
        max_tile = max(max_tile, np.max(next_state))
        state = next_state

        # if steps % 10 == 0:
        #     print(state)
        #     print(reward)
        
        if done:
            break
            
    return env.score, max_tile, steps

# def main():
#     parser = argparse.ArgumentParser(description="Test 2048 AI Agent")
#     parser.add_argument("--episodes", type=int, default=10, help="Number of games to play")
#     parser.add_argument("--render", type=int, default=0, 
#                       help="Render first N episodes (0=disable rendering)")
#     args = parser.parse_args()

#     env = Game2048Env()
#     stats = {
#         'scores': [],
#         'max_tiles': [],
#         'steps': [],
#         'tile_dist': defaultdict(int)
#     }

#     # Run episodes with progress bar
#     for ep in tqdm(range(args.episodes), desc="Testing agent"):
#         render = ep < args.render if args.render > 0 else False
#         score, max_tile, steps = run_episode(env, render=render)
        
#         # Record statistics
#         stats['scores'].append(score)
#         stats['max_tiles'].append(max_tile)
#         stats['steps'].append(steps)
#         stats['tile_dist'][max_tile] += 1

#     # Print statistics
#     print("\n=== Test Results ===")
#     print(f"Episodes played: {args.episodes}")
#     print(f"Average score: {np.mean(stats['scores']):.1f}")
#     print(f"Max score: {np.max(stats['scores'])}")
#     print(f"Average steps: {np.mean(stats['steps']):.1f}")
    
#     print("\nMax Tile Distribution:")
#     tiles = sorted([t for t in stats['tile_dist'].keys() if t >= 128], reverse=True)
#     for tile in tiles:
#         count = stats['tile_dist'][tile]
#         print(f"{tile:5}: {count:4} games ({count/args.episodes:.1%})")

if __name__ == "__main__":
    # main()
    random.seed(0)
    for i in range(10):
        env = Game2048Env()
        env.reset()
        print(run_episode(env))