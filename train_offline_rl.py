import sys
sys.path.append('/home/edge/Desktop/Parallel_Exploration/PGCL-main')
import train_agent_utils as utils
import hydra
import numpy as np
import torch
from tqdm import tqdm
import random
from parallelrooms import ParallelRooms
from offline_q_learning import QLearningWithReplay
from replay_buffer import make_replay_loader
from pathlib import Path
from logger import Logger

from gymnasium.envs.registration import register 
import gymnasium
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
# Load YAML file
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

np.random.seed(0)

env_cfg = cfg["envs"]
policy_cfg = cfg["policy"]
learning_cfg = cfg["learning"]

register(
    id="FrozenLakeCustom-v0",  # Unique name for your custom environment
    entry_point="lake:FrozenLakeEnv",  # Entry point for the environment",
    autoreset=False,
    kwargs={
        "desc": ["SFFF", "FHFH", "FFFH", "HFFG"],
        "map_name": "4x4",
        "is_slippery": True,
        "entropy_mode": True,
    },
    max_episode_steps=100,
    )
register(
    id=env_cfg["env_id"],
    entry_point="parallelrooms:ParallelRooms",
    max_episode_steps=200,
    autoreset=False,
    kwargs={
        "desc": env_cfg["maps"],
        "is_slippery": env_cfg["stochastic"],
        "entropy_mode": False,
    },
)


plt.rcParams.update({
    'font.size': 12 * 3.0,  # base font size
    'axes.titlesize': 14 *5.0,  # title size
    'axes.labelsize': 12 *3.0,  # x and y labels size
    'xtick.labelsize': 10 *3.0,  # x tick labels size
    'ytick.labelsize': 10 * 3.0,  # y tick labels size
    'legend.fontsize': 10 * 3.0,  # legend size
})


env = gymnasium.make(env_cfg["env_id"],desc=env_cfg["maps"],is_slippery=env_cfg["stochastic"])
_,_ = env.reset(seed=0)

policy = ["parl","single_agent","random"]
policy_names = ["Parallel","Single Agent","Random"]
mode = "stoch" if env_cfg["stochastic"] else "det"
seeds = [133,42]
ags = [2,6]
dataset = 5

barplot = False

single_colors = sns.color_palette("dark6" ,3)
single_colors = single_colors[0],single_colors[2]
paralell_colors = sns.color_palette("husl", 3)
paralell_colors = paralell_colors[0],paralell_colors[2]
plot_colors = [[paralell_colors[i],single_colors[i],"#999999"] for i in range(len(ags))]

# add patch to wall positions
walls = 0
for i in range(env.unwrapped.nrow):
    for j in range(env.unwrapped.ncol):
        if env.unwrapped.desc[i][j] == b'-':
           walls += 1

for fa,ag in enumerate(ags):
    fig = plt.figure(figsize=(12, 12))
    for p,pi in enumerate(policy):
        reward = np.zeros((dataset,env.unwrapped.nrow, env.unwrapped.ncol))
        for j in range(dataset):
            file_path = Path(f"/home/edge/Desktop/Parallel_Exploration/PGCL-main/datasets/first/labirinth/{pi}/{mode}/{seeds[fa]}/episode_{seeds[fa]}_{j}_{ag}.npz")
            # Instantiate the Q-learning agent
            replay_loader = make_replay_loader(file_path, 20,20,1,0.99)
            replay_iter = iter(replay_loader)

            state_size = env.observation_space.n
            action_size = env.action_space.n

            agent = QLearningWithReplay(state_size, action_size, replay_iter,external_target=True)

            
            default_map = env_cfg["maps"].copy()
            for i,state in enumerate(env.valid_states):
                if state != env_cfg["initial_state"]:            
                    #convert i in row,col
                    row = state // env.unwrapped.ncol
                    col = state % env.unwrapped.ncol
                    # Convert the target row into a list (since strings are immutable)
                    print("actual goal position",row,col)
                    row_list = list(default_map[row])
                    # Modify the character
                    row_list[col] = 'G'
                    # Convert it back to a string and update the map
                    default_map[row] = ''.join(row_list)
                    env.update_map(default_map)
                    agent.set_new_target(state)
                    # Train the agent using the replay buffer
                    agent.train_from_replay(episodes=100)
                    # Evaluate the agent's policy
                    reward[j,row,col] = agent.evaluate_policy(env, test_episodes=10)

                    agent.reset()
                    default_map = env_cfg["maps"].copy()
                # Reset the map

            print(reward.sum())
            print("Training complete.")

        reward = reward.mean(axis=0)
        # Create a (5,11) matrix with example values
            # Set axis limits for the grid
        grid_size = (env.unwrapped.nrow, env.unwrapped.ncol)

        if barplot:
            plt.figure(figsize=(12, 12))
            #sort the reward in descending order
            reward = reward.flatten()
            reward = reward[np.argsort(reward)[::-1]]
            reward = reward[:-walls]
            plt.bar(np.arange(len(reward)),reward,color=plot_colors[fa][p],width=0.7)
            # add horizontal line for the average reward avoid considering the walls
            plt.axhline(y=reward.mean(), color=plot_colors[fa][p], linewidth=4,linestyle='--',label="Average Reward")
            plt.xticks(np.arange(0, len(reward), 1))
            plt.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
            plt.ylabel("Success Rate")
            plt.xlabel("Goal Position")
            plt.title(f"{policy_names[p]} Dataset")
            plt.savefig(f'/home/edge/Desktop/Parallel_Exploration/PGCL-main/plots/goal_occupancy_{policy_names[p]}_{env_cfg["env_id"]}_{ag}_{env_cfg["stochastic"]}.png')
        else:
            plt.xlim(-0.5, grid_size[1] - 0.5)  # X-axis spans columns
            plt.ylim(-0.5, grid_size[0] - 0.5)  # Y-axis spans rows
            plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
            plt.xticks(np.arange(-0.5, grid_size[1], 1))  # Columns
            plt.yticks(np.arange(-0.5, grid_size[0], 1))  # Rows
            plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            # Ensure the grid cells are square
            plt.title(f"{policy_names[p]}",fontsize=36)  # Set the title
            plt.imshow(reward, cmap="viridis")  # Store the image object
            plt.savefig(f'/home/edge/Desktop/Parallel_Exploration/PGCL-main/plots/heatmap_goal_occupancy_{policy_names[p]}_{env_cfg["env_id"]}_{ag}_{env_cfg["stochastic"]}.png')
        print(fa,p)