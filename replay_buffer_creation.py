import numpy as np
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('/home/edge/Desktop/Parallel_Exploration/PGCL-main')
import train_agent_utils as aut
import os
import time
import plot as pl
from tqdm import tqdm

import gymnasium
import datetime
from gymnasium.envs.registration import register
import trajs_utils as ut

import yaml
import Softmax as sm


# Load YAML file
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

env_cfg = cfg["replay_buffer"]
train_cfg = cfg["envs"]
policy_cfg = cfg["policy"]

# Check if the environment is already registered
if env_cfg["env_id"] not in gymnasium.registry:
    if "club" in env_cfg["env_id"]:
        register(
            id=env_cfg["env_id"],
            entry_point="club_env:club",
            max_episode_steps=300,
        )
    elif "FrozenLake" in env_cfg["env_id"]:
        
        register(
        id="FrozenLakeCustom-v0",  # Unique name for your custom environment
        entry_point="lake:FrozenLakeEnv",  # Entry point for the environment",
        autoreset=False,
        kwargs={
            "desc":  env_cfg["maps"],
            "map_name": "8x8",
            "is_slippery": True,
            "entropy_mode": True,
        },
        max_episode_steps=100
    )
    elif "Taxi" in env_cfg["env_id"]:
        register(
            id=env_cfg["env_id"],
            entry_point="taxi:TaxiEnv",
            max_episode_steps=200,
            autoreset=False,
        )
    elif "Parallel" in env_cfg["env_id"]:
        register(
            id=env_cfg["env_id"],
            entry_point="parallelrooms:ParallelRooms",
            max_episode_steps=200,
            autoreset=False,
            kwargs={
                "desc": env_cfg["maps"],
                "is_slippery": train_cfg["stochastic"],
                "entropy_mode": False,
            },
        )
else:
    print(f"Environment {env_cfg['env_id']} is already registered.")

### Environment creation
seed_array = train_cfg["seed"]

#remove everything from the save directory
# if os.path.exists(env_cfg['save_directory']):
#     os.system(f"rm -r {env_cfg['save_directory']}")
multiple_agents = env_cfg["agents"] 
for ag,num_agents in enumerate(multiple_agents):
    env_cfg["agents"] = num_agents
    env_cfg["num_envs"] = num_agents
    for seed in seed_array:
        envs = gymnasium.make_vec(env_cfg["env_id"],num_envs=env_cfg["num_envs"])
        envs.reset(seed=seed)
        obs_size = envs.observation_space.nvec[0]
        action_size = envs.action_space.nvec[0]

        policy = sm.SoftMaxPolicy(env_cfg["agents"],action_size,obs_size,policy_cfg["target_policy"],policy_cfg["zero_init"])

        for l in range(len(env_cfg["replay_size"])):
            
            ### Replay buffer creation
            size_replay = env_cfg["replay_size"][l]

            if not policy_cfg["zero_init"]:
                if policy_cfg["single_agent"]:
                    theta_init = np.load(env_cfg["load_single_theta"][ag])
                    policy.set_theta(theta_init)
                else:
                    theta_init = np.load(env_cfg["load_multiple_theta"][ag])
                    policy.set_theta(theta_init)

            max_steps = env_cfg['num_observations']

            sars = []

            for i in tqdm(range(size_replay)):
                s,_ = envs.reset()
                truncated = np.zeros(env_cfg["num_envs"],dtype=bool)
                done = np.zeros(env_cfg["num_envs"],dtype=bool)
                for j in range(0, env_cfg['num_observations']):
                    s_one_hot = ut.one_hot_encoding(s, obs_size)
                    a = policy.predict(s_one_hot)
                    next_s, reward, truncated, done, info = envs.step(a)
                    for env_idx in range(env_cfg["num_envs"]):
                        if truncated[env_idx] or done[env_idx]:
                            sars.append((s[env_idx], a[env_idx], reward[env_idx], truncated[env_idx], done[env_idx], info["final_observation"][env_idx]))
                        else:
                            sars.append((s[env_idx], a[env_idx], reward[env_idx], truncated[env_idx], done[env_idx], next_s[env_idx]))
                    # Update current state
                    s = next_s


            if not os.path.exists(env_cfg['save_directory'] + str(seed)):
                os.makedirs(env_cfg['save_directory'] + str(seed))
            
            output_dict = {"sars": np.array(sars)} 
            
            save_path = f"{env_cfg['save_directory']+ str(seed)}/episode_{str(seed)}_{ag}_{env_cfg['replay_size'][l]*env_cfg['num_envs']}.npz"

            np.savez(save_path, **output_dict)