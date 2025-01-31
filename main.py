import gymnasium
import datetime
from gymnasium.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import train_agent_utils as t_utils
import trajs_utils as ut
from club_env import club
import Softmax as sm
import plot as pl
import mdp as mdp
from lr_utils import AdamOptimizer,LearningRateScheduler
import eval
import time
import yaml
from lake import FrozenLakeEnv
import random
from logger import Logger
import os
import sys
sys.path.append('.')
# Load YAML file
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

np.random.seed(0)

env_cfg = cfg["envs"]
policy_cfg = cfg["policy"]
learning_cfg = cfg["learning"]

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
            "map_name": "4x4",
            "is_slippery": env_cfg["stochastic"],
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
                "is_slippery": env_cfg["stochastic"],
                "entropy_mode": True,
            },
        )
else:
    print(f"Environment {env_cfg['env_id']} is already registered.")


if __name__ == "__main__":

    real_objective_collection = []
    seed = env_cfg["seed"]
    multiple_agents = env_cfg["multiple_agents"]
    for multiple in multiple_agents:
        env_cfg["agents"] = multiple
        env_cfg["num_envs"] = multiple
        for episode in tqdm(range(learning_cfg["num_episode"])):
            
            if policy_cfg['single_agent']:
                sub_dir = f"{env_cfg['env_id']}_1{env_cfg['num_envs']}_{seed[episode]}"
            else:
                sub_dir = f"{env_cfg['env_id']}_{env_cfg['num_envs']}_{seed[episode]}"
            if env_cfg["stochastic"]:
                pt = Path("test/st")
            else:
                pt = Path("test/det") 
            logger = Logger(pt,sub_dir,True)

            np.random.seed(seed[episode])
            random.seed(seed[episode])

            envs = gymnasium.make_vec(env_cfg["env_id"],num_envs=env_cfg["num_envs"])
            envs.reset(seed=seed[episode])
            obs_size = envs.observation_space.nvec[0]
            effective_obs_size = len(envs.get_attr("valid_states")[0])
            action_size = envs.action_space.nvec[0]

            if policy_cfg["linear"]:
                policy = sm.LinearPolicy(env_cfg["agents"],action_size,obs_size,env_cfg["horizon"],policy_cfg["target_policy"],policy_cfg["zero_init"])
            else:
                if policy_cfg["non_stationary"]:
                    policy = sm.SoftMax_nst_Policy(env_cfg["agents"],action_size,obs_size,env_cfg["horizon"],policy_cfg["target_policy"],policy_cfg["zero_init"])
                else:
                    policy = sm.SoftMaxPolicy(env_cfg["agents"],action_size,obs_size,policy_cfg["target_policy"],policy_cfg["zero_init"],policy_cfg["single_agent"])

            if learning_cfg["Adam"]:
                optimizer = AdamOptimizer(policy.theta,learning_cfg["initial_lr"], learning_cfg["beta1"], learning_cfg["beta2"])
            else:
                scheduler =  LearningRateScheduler(learning_cfg["initial_lr"], learning_cfg["decay_rate"], learning_cfg["decay_steps"]) 
                alpha = learning_cfg["initial_lr"]#scheduler.get_lr(0)  # learning rate

            gradients = []
            cum_distance = []
            cum_overlaps = []
            entropy = []
            objective = []
            real_objective = []
            cum_support_h = []
            mean_entropy_single_agent_h = [] 
            history_heatmap = []
            array_history_single_heatmap = [[] for _ in range(env_cfg["agents"])]
            timer = t_utils.Timer()
            
            target_policy = policy_cfg["target_policy"]

            if policy_cfg["force_init"]:
                theta_init = ut.initialize_policy(env_cfg["agents"],env_cfg["horizon"],env_cfg["obs_dim"],env_cfg["deterministic"])
                policy.set_theta(theta_init)
            else:
                policy.set_force_probs(False)

            #history_probs = np.zeros((learning_cfg["iterations"],env_cfg["agents"],len(mdp_traj)))
            #empirical_actions = np.zeros((learning_cfg["iterations"],env_cfg["horizon"],env_cfg["agents"],action_size,obs_size))
            
            for itr in tqdm(range(learning_cfg["iterations"])):

                # Generate rollouts
                seed_multiple = [int(seed[episode]) for _ in range(env_cfg["num_envs"])] 
                if policy_cfg["single_agent"]:
                    pg,entropy_montecarlo,cum_support,mean_entropy_single_agent,obs = ut.collect_rollouts_states_single_agents(envs, policy, learning_cfg["trajectories"], env_cfg["total_time"],obs_size,learning_cfg["mini_batch"],seed=seed_multiple) 
                else:
                    pg,entropy_montecarlo,cum_support,mean_entropy_single_agent,obs = ut.collect_rollouts_states(envs, policy, learning_cfg["trajectories"], env_cfg["total_time"],obs_size,learning_cfg["mini_batch"],seed=seed_multiple)

                r_o = entropy_montecarlo
                
                # Get policy parameter
                theta = policy.get_theta()

                # Update policy parameters
                if learning_cfg["Adam"]:
                    pg = optimizer.update(pg)
                    theta = theta + pg
                else:
                    #alpha = scheduler.get_lr(itr)  # learning rate
                    theta = theta + (alpha * pg) 

                # Set policy parameters
                policy.set_theta(theta)
                
                #save gradients and empirical actions
                gradients.append(theta)

                elapsed_time, total_time = timer.reset()
                with logger.log_and_dump_ctx(itr, ty='train') as log:
                    log('Objective',r_o/(-np.log(1/effective_obs_size)))
                    log('Support', cum_support)
                    log('Step', itr)
                    log('Single_entropy', mean_entropy_single_agent)
                    log('fps', 100 / elapsed_time)
                    log('total_time', total_time)
                    log('step', itr)

                #if itr==0:
                    #first_emp_rollouts = np.zeros_like(agent_probs)
                    #first_emp_rollouts = agent_probs
                #    first_t_rollouts = np.zeros_like(p_t)
                #    first_t_rollouts = p_t
                
                if itr == learning_cfg["iterations"]-1:
                    now = datetime.datetime.now()
                    if not os.path.exists(f"theta/{env_cfg['env_id']}/{env_cfg['stochastic']}"):
                        os.makedirs(f"theta/{env_cfg['env_id']}/{env_cfg['stochastic']}")
                    # Format datetime as a string
                    datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")
                    if policy_cfg["single_agent"]:
                        np.save(f"theta/{env_cfg['env_id']}/{env_cfg['stochastic']}/{seed[episode]}_{env_cfg['num_envs']}_st_{env_cfg['stochastic']}_single_agent_theta.npy",policy.theta)
                    else:
                        np.save(f"theta/{env_cfg['env_id']}/{env_cfg['stochastic']}/{seed[episode]}_{env_cfg['num_envs']}_st_{env_cfg['stochastic']}_theta.npy",policy.theta)

            #################################################################################################################################
            ###########################################
            ###########################################
            ###########################################
            ###########################################
            #PLOT SECTION  PLOT SECTION  PLOT SECTION  PLOT SECTION  PLOT SECTION  PLOT SECTION  PLOT SECTION 
            ###########################################
            ###########################################
            ###########################################
            ###########################################
            ##################################################################################################################################

            if learning_cfg["plot"]:
                # Get current datetime
                now = datetime.datetime.now()
                # Format datetime as a string
                datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")

                
                #pl.plot_update_ns(learning_cfg["iterations"],env_cfg["agents"],obs_size,env_cfg["horizon"], gradients, datetime_str)

                #pl.plot_entropy(entropy,datetime_str)

                #pl.plot_ns_policy(learning_cfg["iterations"],env_cfg["horizon"],empirical_actions,env_cfg["agents"],obs_size,datetime_str)
                
                #pl.plot_agent_specific_distance(learning_cfg["iterations"],env_cfg["agents"],cum_distance,datetime_str)
                #pl.plot_agent_specific_distance(learning_cfg["iterations"],env_cfg["agents"],cum_overlaps,datetime_str+"_overlap")
                        
                #pl.barplot_references(agent_probs,env_cfg["agents"],first_emp_rollouts,f"{datetime_str}_barplot_emp_ref")
                #pl.barplot_references(p_t,env_cfg["agents"],first_t_rollouts,f"{datetime_str}_barplot_teor_ref")

                #print(p_t,first_t_rollouts,np.sum(p_t),np.sum(first_t_rollouts))
                #pl.barplot_references(p_t,1,first_t_rollouts,f"{datetime_stsr}_barplot_teor_ref")
                #save mdp_traj in a file.txt
                #np.savetxt(f"test/{datetime_str}_mdp_traj.txt", mdp_traj, fmt='%d')

                #pl.plot_objective(objective,datetime_str,"objective")
                #pl.plot_objective(real_objective,datetime_str,"real_objective")
                pl.plot_montecarlo_objective(real_objective,-np.log(1/obs_size),datetime_str,"real_objective")
                pl.plot_support(cum_support_h,datetime_str,"cum_support")
                pl.plot_mean_entropy_single_agent(mean_entropy_single_agent_h,datetime_str,"mean_entropy_single_agent")
                #pl.plot_3d_surface(history_heatmap,(env_cfg["row"],env_cfg["col"]))
                #
                # 
                # pl.plot_heatmap_new(history_heatmap,(env_cfg["row"],env_cfg["col"]))
            
                for agent in range(env_cfg["agents"]):
                    pl.plot_heatmap_new(array_history_single_heatmap[agent],(env_cfg["row"],env_cfg["col"]),str(agent))    
                    pl.plot_3d_surface(array_history_single_heatmap[agent],(env_cfg["row"],env_cfg["col"]),str(agent))

       



    