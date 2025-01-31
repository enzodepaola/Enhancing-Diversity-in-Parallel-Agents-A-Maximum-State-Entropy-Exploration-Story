import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import math
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
import itertools
import pandas as pd
import os
from joblib import Parallel, delayed
import time
import plot as pl
import concurrent.futures
from numba import jit, prange

##### FILE UTILS #####

def create_gif_trajs(foldername,filenames,name):

  """
  Create a gif from a folder of images
  """

  with imageio.get_writer(f'{name}.gif', mode='I') as writer:
    for filename in filenames:
      image = imageio.imread(os.path.join(foldername,filename))
      writer.append_data(image)
  return

def remove_duplicates(foldername,filenames):

  """
  Remove duplicate files from a folder.
  """
  for filename in set(filenames):
    os.remove(os.path.join(foldername,filename))
  return 

###########################
###########################


##### INITIALIZATION #####

def gen_random_trajs(K,m):

  """
  Generate random trajectories for the MDP.

  Parameters:
  K (int): The number of trajectories.
  m (int): The number of agents.

  Returns:
  np.array: The random trajectories.
  """

  x0 = np.random.rand(m*K)  # Initial guess
  random_matrix = np.random.rand(m, K)
  row_sums = random_matrix.sum(axis=1).reshape(m, 1)  # Calculate row sums and reshape for broadcasting
  x0 = random_matrix / row_sums

  return x0

def state_actions_trajs(options,horizon,initial_state,policy,transitions):
  
  """
  Input:
  options: list of possible states
  horizon: length of the trajectory
  initial_state: initial state of the trajectory
  policy: initial policy
  transitions: transition matrix of the MDP
  
  Output:
  possible_traj: possible trajectories
  probs: probabilities of the trajectories  
  """

  # MDP Transitions from state to next state CLUB ENV EXAMPLE

  combinations = np.array(list(itertools.product(options, repeat=horizon+1)))
  action_combinations = np.array(list(itertools.product([0,1], repeat=horizon)))
  possible_traj = combinations[np.where((combinations[:,0]==initial_state))]
  probs = []
  for t in possible_traj:
    p_t = np.ones(len(action_combinations))
    for a,act_traj in enumerate(action_combinations):
      for i,s in enumerate(t):
        if i< len(t)-1:
            p_t[a] *= transitions[s][t[i+1]][act_traj[i]] * policy[act_traj[i]]
          
    probs.append(np.sum(p_t))

  return np.array(possible_traj),np.array(probs)

def normalize_distributions(distributions, threshold=1e-10):
    # Ensure the input is a numpy array
    distributions = np.array(distributions, dtype=float)
    # Set small negative values to zero
    for d,distribution in enumerate(distributions):
      distribution = np.where(distribution < 0.00001, 0, distribution)
      distributions[d] = distribution
    
    #distributions[distributions < -threshold] = 0
    
    # Compute the sum along the last axis (typically the distributions are in rows)
    sums = distributions.sum(axis=1, keepdims=True)
    
    # Avoid division by zero in case of an all-zero distribution
    sums[sums == 0] = 1
    
    # Normalize by dividing each element by the sum of its distribution
    normalized_distributions = distributions / sums
    
    return normalized_distributions

def possible_trajs(options,horizon,initial_state,transitions):
  """
  Calculate possible discrete trajectories in a MDP.

  Parameters:
  options (list): The possible states in the MDP.
  horizon (int): The length of the trajectory minus the initial state.
  initial_state (int): The initial state of the MDP.
  transitions (np.array): The transition matrix of the MDP.
  
  Returns:
  np.array: The possible trajectories in the MDP.
  """
  np.random.default_rng(12345)
  
  combinations = np.array(list(itertools.product(options, repeat=horizon+1)))
  #action_combinations = np.array(list(itertools.product([0,1], repeat=horizon)))
  possible_traj = combinations[np.where((combinations[:,0]==initial_state))]
  
  #remove that combination where it's impossibile to reach the final state
  possible_traj = constrained_traj(possible_traj,transitions)

  return np.array(possible_traj)

def constrained_traj_o(possible_traj,transitions):
    new_traj = []
    flag = False
    for traj in possible_traj:
      for i,t in enumerate(traj):
        if i< len(traj)-1:
          if np.sum(transitions[t][traj[i+1]]) == 0:
            flag = True
            break
      if not flag:
        new_traj.append(traj)
      flag = False
    return np.array(new_traj)

def constrained_traj(possible_traj, transitions):
    new_traj = []
    for traj in possible_traj:
        valid = True
        for i in range(len(traj) - 1):
            if np.sum(transitions[traj[i]][traj[i + 1]]) == 0:
                valid = False
                break
        if valid:
            new_traj.append(traj)
    return np.array(new_traj)

def from_probs_to_theta(probs):
  """
  Convert a probability distribution to a theta softmax functions parameters.
  The theta parameter is a real number that represents the probability of the first action.

  Parameters:
  probs (np.array): The probability distribution of first action along horizon.

  Returns:
  float: The theta parameter.
  """
  total_theta = []
  for _ in range(len(probs)):
    theta = np.zeros((2,len(probs[_])))
    #solve a linear system to find the theta parameter
    A = np.array([[1, -1], [-1, -1]])
    for i,p in enumerate(probs[_]):
      if p == 1:
        p = 0.99
      elif p == 0:
        p = 0.01
      B = np.array([math.log(p/(1-p)), math.log((1-p)/p)])
      r = np.linalg.solve(A,B)
      theta[0,i],theta[1,i] = r[0],r[1]
    total_theta.append(theta)
  return np.array(total_theta)

def initialize_policy(agents,horizon,obs_dim,deterministic):
  
  optimal_theta = []
  # create random policy for each agent 
  if not deterministic:
    policies = np.random.rand(agents,horizon,obs_dim)
  else:
    policies = np.random.randint(low=0,high=1,size=(agents,horizon,obs_dim))

  for pi in policies:
    optimal_theta.append(from_probs_to_theta(pi))

  return np.array(optimal_theta)

###########################
###########################


##### ON STEP FUNCTIONS #####

def one_hot_encoding(s,obs_size):

  """
  Convert a state to a one-hot encoding.

  Input:
  s: state
  obs_size: number of states
  
  Output:
  one_hot: one-hot encoding of the state
  """

  one_hot = np.zeros((s.shape[0],obs_size))
  one_hot[np.arange(len(s)), s] = 1
  return one_hot

def collect_rollouts_grad(env, policy, K, T,obs_size,mdp_traj,transitions):
  agents = policy.agents
  grad = 0
  #exact_distribution  = calculate_exact2(agents, mdp_traj, policy, transitions) 
  #p_t,grad = calculate_exact(agents, mdp_traj, policy, transitions)
  grad,objective = calculate_distribution_gradient_exact(agents, mdp_traj, policy, transitions)
  return grad,objective

def collect_rollouts(env, policy, K, T,obs_size,mdp_traj,itr):
  agents = policy.agents
  obs = []
  acts = []
  #reward = []
  
  index = 0
  scores = np.zeros((agents, T-1, policy.theta.shape[-2], policy.theta.shape[-1]))
  cum_scores = np.zeros_like(scores)
  cum_entropy = 0.0
  grad = np.zeros_like(scores)
  rw = np.zeros_like(scores)

  old_counts_itr = np.zeros((agents, mdp_traj.shape[0]))
  old_probs_itr = np.zeros((agents, mdp_traj.shape[0]))

  act_traj = np.array(list(itertools.product([0,1], repeat=mdp_traj.shape[1]-1)))
  old_counts_act_itr = np.zeros((agents, mdp_traj.shape[0],act_traj.shape[0]))
  old_probs_act_itr = np.zeros((agents, mdp_traj.shape[0],act_traj.shape[0]))

  itr = itr

  for l in range(itr):
    old_counts = np.zeros((agents, mdp_traj.shape[0]))
    old_probs = np.zeros((agents, mdp_traj.shape[0]))
    for j in range(K):
      s, _ = env.reset()
      ob = np.zeros((agents, T))  # Pre-allocate for observations
      act = np.zeros((agents, T-1))  # Pre-allocate for actions
      ob[:, 0] = s.reshape(agents,)  # Set the initial state
      scores = np.zeros((agents, T-1, policy.theta.shape[-2], policy.theta.shape[-1]))
      t = 0
      while t < T - 1:
        s_one_hot = one_hot_encoding(s, obs_size)
        a = policy.predict(s_one_hot, t)
        s1, r, done, _, _ = env.step(a)

        ob[:, t+1] = s1.reshape(agents,)  # Update the observation array
        act[:, t] = a  # Update the action array
        scores[:, t] = policy.log_softmax_derivative(s_one_hot, a, t)

        s = s1
        t += 1

      obs.append(ob)
      acts.append(act)

      # Calculate trajectory indices and update counts
      played_traj = [np.argwhere(np.all(ob[agent] == mdp_traj, axis=1)).item() for agent in range(agents)]
      old_counts[np.arange(len(played_traj)), played_traj] += 1

      #played_act_traj = [np.argwhere(np.all(act[agent] == act_traj, axis=1)).item() for agent in range(agents)]
      #old_counts_act_itr[np.arange(len(played_act_traj)), played_traj, played_act_traj] += 1
      #old_probs_act_itr = old_counts_act_itr / (index + 1)
      #old_counts_itr[np.arange(len(played_traj)), played_traj] += 1
      #old_probs_itr = old_counts_itr / (index + 1)

      # Normalize old_probs
      #old_probs = old_counts / (index + 1)
      #index += 1
      #probs_played = old_probs.sum(axis=0) 
      probs_played = old_counts.sum(axis=0)
      probs_played = probs_played / probs_played.sum()

      cum_scores += update_trajectory_scores(agents,probs_played,scores)
      cum_entropy += calculate_traj_entropy(probs_played)
  
  # with open(f"test/action_sample_{itr}.txt", "w") as f:
  #   for s in np.mean(old_probs_act_itr,axis=0):
  #     f.write(str(s) +"\n")
  # with open(f"test/action_probs_sample_{itr}.txt", "w") as f:
  #   for s in np.mean(old_probs_itr,axis=0):
  #     f.write(str(s) +"\n") 
  #np.save(f"test/action_sample_{itr}.npy",np.mean(old_probs_act_itr,axis=0))
  cum_entropy = cum_entropy/itr  
  grad = cum_scores/itr
  return grad,cum_entropy

def collect_rollouts_ric(env, policy, K, T,obs_size,itr):
  agents = policy.agents
  obs = []
  acts = []
  #reward = []
  
  index = 0
  scores = np.zeros((agents, T-1, policy.theta.shape[-2], policy.theta.shape[-1]))
  cum_scores = np.zeros_like(scores)
  cum_entropy = 0.0
  cum_support = 0
  grad = np.zeros_like(scores)

  itr = itr

  for l in range(itr):
    for j in range(K):
      s, _ = env.reset()
      ob = np.zeros((agents, T))  # Pre-allocate for observations
      act = np.zeros((agents, T-1))  # Pre-allocate for actions
      ob[:, 0] = s.reshape(agents,)  # Set the initial state
      scores = np.zeros((agents, T-1, policy.theta.shape[-2], policy.theta.shape[-1]))
      t = 0
      while t < T - 1:
        s_one_hot = one_hot_encoding(s, obs_size)
        a = policy.predict(s_one_hot, t)
        s1, r, truncated, done, info = env.step(a)
        # Create masks for done or truncated agents
        ob[:, t+1] = s1.reshape(agents,)  # Update the observation array

        act[:, t] = a  # Update the action array
        scores[:, t] = policy.log_softmax_derivative(s_one_hot, a, t)

        s = s1
        t += 1

      obs.append(ob)
      acts.append(act)

      # count occurrency of the trajectory along agents
      
      trajs,counts=np.unique(ob, axis=0, return_counts=True)

      probs_played = counts / counts.sum()

      cum_scores += update_trajectory_scores(agents,probs_played,scores)
      cum_entropy += calculate_traj_entropy(probs_played)
      cum_support += len(counts)


  mean_entropy_single_agent = calculate_mean_entropy_single_agent(obs)
  cum_entropy = cum_entropy/itr  
  grad = cum_scores/itr
  return grad,cum_entropy,cum_support/itr,mean_entropy_single_agent



def collect_rollouts_states(env,policy, K, T,obs_size,itr,seed):
  start = time.time()
  agents = policy.agents
  obs = []
  acts = []
  
  index = 0
  scores = np.zeros((agents, policy.theta.shape[-2], policy.theta.shape[-1]))
  cum_scores = np.zeros_like(scores)
  cum_entropy = 0.0
  cum_support = 0
  grad = np.zeros_like(scores)

  itr = itr

  for l in range(itr):
    for j in range(K):
      s, _ = env.reset()
      ob = np.zeros((agents, T))  # Pre-allocate for observations
      act = np.zeros((agents, T-1))  # Pre-allocate for actions
      ob[:, 0] = s.reshape(agents,)  # Set the initial state
      scores = np.zeros((agents, policy.theta.shape[-2], policy.theta.shape[-1]))
      t = 0
      while t < T - 1:
        s_one_hot = one_hot_encoding(s, obs_size)
        a = policy.predict(s_one_hot)
        s1, r, truncated, done, info = env.step(a)
        # Create masks for done or truncated agents
        ob[:, t+1] = s1.reshape(agents,)  # Update the observation array

        act[:, t] = a  # Update the action array
        scores += policy.log_softmax_derivative(s_one_hot, a)

        s = s1
        t += 1

      obs.append(ob)
      acts.append(act)

      tmp_scores,tmp_entropy,counts = update_states_scores(agents,ob,scores)
      cum_scores += tmp_scores
      cum_entropy += tmp_entropy
      cum_support += len(counts)

  mean_entropy_single_agent = calculate_mean_entropy_single_agent_states(obs)
  cum_entropy = cum_entropy/itr  
  grad = cum_scores/itr
  print(f"Time elapsed: {time.time()-start}")
  return grad,cum_entropy,cum_support/itr,mean_entropy_single_agent,obs

def collect_rollouts_states_single_agents(env,policy, K, T,obs_size,itr,seed):
  
  agents = policy.agents
  scores = np.zeros((agents, policy.theta.shape[-2], policy.theta.shape[-1]))
  cum_scores =  np.zeros((1, policy.theta.shape[-2], policy.theta.shape[-1]))
  cum_entropy = 0.0
  cum_support = 0
  grad = np.zeros_like(scores)

  itr = itr
  obs = np.zeros((itr,K*agents,T))
  acts = np.zeros((itr,K*agents,T-1))
  for l in range(itr):
    for j in range(K):
      s, _ = env.reset()
      ob = np.zeros((agents, T))  # Pre-allocate for observations
      act = np.zeros((agents, T-1))  # Pre-allocate for actions
      scores = np.zeros((agents, policy.theta.shape[-2], policy.theta.shape[-1]))
      ob[:, 0] = s.reshape(agents,)  # Set the initial state
      t = 0
      while t < T - 1:
        s_one_hot = one_hot_encoding(s, obs_size)
        a = policy.predict(s_one_hot)
        s1, r, truncated, done, info = env.step(a)
        # Create masks for done or truncated agents
        ob[:, t+1] = s1.reshape(agents,)  # Update the observation array

        act[:, t] = a  # Update the action array
        scores[:] += policy.log_softmax_derivative(s_one_hot, a)

        s = s1
        t += 1

      obs[l]  = ob
      acts[l] = act

      # count occurrency of the trajectory along agents
      
    u_states,counts = np.unique(obs[l].flatten(), axis=-1, return_counts=True)
    probs_played = counts / counts.sum()

    cum_scores += update_states_scores_single_agents(probs_played,scores.sum(axis=0))
    cum_entropy += calculate_states_entropy(probs_played)
    cum_support += len(counts)

  mean_entropy_single_agent = 0 #calculate_mean_entropy_single_agent_states(obs)
  cum_entropy = cum_entropy/itr  
  grad = cum_scores/itr
  cum_support = cum_support/itr
  return grad,cum_entropy,cum_support,mean_entropy_single_agent,obs


def update_scores(scores,agent_probs,played_traj):
  
  agents = agent_probs.shape[0]
  update_scores = np.zeros_like(scores)
  
  for agent in range(agents):
    other_agents = 1
    update_scores[agent] = agent_probs[agent,played_traj[agent]]*scores[agent] 
    for n_agent in range(agents):
      if n_agent != agent:
        other_agents *= (1 - agent_probs[n_agent,played_traj[n_agent]])
    update_scores[agent] *= other_agents
  
  return -update_scores

def update_trajectory_scores(agents,probs_played, scores):

  #calculate the overall probability of the trajectory played by the agents at this iteration
  cum_scores = np.zeros_like(scores)

  entropy = calculate_traj_entropy(probs_played)

  for agent in range(agents):
    cum_scores[agent] = entropy * scores[agent]

  return cum_scores

def update_trajectory_scores_o(old_counts,played_traj, scores):

  #calculate the overall probability of the trajectory played by the agents at this iteration
  agents = old_counts.shape[0]
  cum_scores = np.zeros_like(scores)
  rewards = np.zeros(agents)
  probs_played = old_counts.sum(axis=0) 
  probs_played = probs_played / probs_played.sum()

  entropy = calculate_traj_entropy(probs_played)

  for agent in range(agents):
    r =  np.prod([probs_played[played_traj[agent]] for i in range(agents) if i != agent])
    rewards[agent] = entropy * r #[1 - probs_played[played_traj[agent]] for i in range(agents) if i != agent]
    cum_scores[agent] = rewards[agent] * scores[agent]

  return cum_scores

def update_trajectory_scores_exact(exact_probs,played_traj, scores):

  #calculate the overall probability of the trajectory played by the agents at this iteration
  agents = exact_probs.shape[0]
  cum_scores = np.zeros_like(scores)
  rewards = np.zeros(agents)
  
  probs_played = exact_probs.sum(axis=0) 
  probs_played = probs_played / probs_played.sum()

  for agent in range(agents):
    rewards[agent] = probs_played[played_traj[agent]]
    cum_scores[agent] = rewards[agent] * scores[agent]

  return -cum_scores

def update_trajectory_scores_ric(old_counts, scores):

  #calculate the overall probability of the trajectory played by the agents at this iteration
  agents = old_counts.shape[0]
  cum_scores = np.zeros_like(scores)

  #entropy = calculate_traj_entropy(probs_played)

  for agent in range(agents):
    entropy = calculate_traj_entropy(old_counts[agent])
    cum_scores[agent] = entropy * scores[agent]

  return cum_scores

def calculate_mean_entropy_single_agent(obs_trajs):
  obs_trajs = np.array(obs_trajs)
  agents = obs_trajs.shape[1]
  mean_entropy = 0

  for agent in range(agents):
    tmp_obs = obs_trajs[:, agent]
    _,count = np.unique(tmp_obs, axis=0,return_counts=True)
    probs = count / count.sum()
    mean_entropy += calculate_traj_entropy(probs)
  
  mean_entropy = mean_entropy / agents
  
  return mean_entropy



####### STATES IMPLEMENTATION #######
@jit()
def update_states_scores(agents,ob, scores):
  flattened = ob.flatten()
  unique_vals = np.unique(flattened)
  counts = np.zeros_like(unique_vals, dtype=np.int32)
  
  # Count occurrences of each unique value
  for i in range(len(unique_vals)):
      counts[i] = np.sum(flattened == unique_vals[i])
  
  probs_played = counts / counts.sum()
  cum_scores = np.zeros_like(scores)
  entropy = calculate_states_entropy(probs_played)
  for agent in range(agents):
    cum_scores[agent] = entropy * scores[agent]  
  return cum_scores,entropy,counts

def update_states_scores_single_agents(played_states,scores):
  cum_scores = np.zeros_like(scores)
  entropy = calculate_states_entropy(played_states)
  cum_scores = entropy * scores
  return cum_scores

@jit()
def calculate_states_entropy(states):
  #states = np.array(states)  # Convert to numpy array if needed
  #sum over axis 0 and divide by the number of agents
  states = states[states > 0]  # Rimuovi valori di probabilità zero
  entropy = -np.sum(states * np.log(states))
  return entropy

def calculate_mean_entropy_single_agent_states(obs_trajs):
  obs_trajs = np.array(obs_trajs)
  agents = obs_trajs.shape[1]
  mean_entropy = 0

  for agent in range(agents):
    tmp_obs = obs_trajs[:, agent]
    _,count = np.unique(tmp_obs, axis=0,return_counts=True)
    probs = count / count.sum()
    mean_entropy += calculate_states_entropy(probs)
  
  mean_entropy = mean_entropy / agents

  return mean_entropy

def decode_trajectory(trajs,grid_size):
    """
    Decodes a list of trajectories represented as linear indices into row-column coordinates.

    Args:
        trajs (list of list of int): List of trajectories, where each trajectory is a list of linear indices.
        grid_size (tuple): Tuple of (rows, cols) representing the grid dimensions.

    Returns:
        list of list of tuple: Decoded trajectories as lists of (row, col) coordinates.
    """
    rows, cols = grid_size
    decoded_trajs = []

    for t in trajs:
        decoded_t = [(index // cols, index % cols) for index in t]
        decoded_trajs.append(decoded_t)

    return decoded_trajs

def decode_trajectory_taxi(trajs,grid_size):
  rows, cols = grid_size
  decoded_trajs = [] 
  for t in trajs:
      pass_position = trj % 5
      trj = trj // 5
      col = trj % 5
      trj = trj // 5
      row = trj 
      decoded_t = [calculate_taxi_position(index) for index in t]
      decoded_trajs.append(decoded_t)
  return decoded_trajs

def calculate_taxi_position(value):
  pass_position = value % 5
  value = value // 5
  col = value % 5
  value = value // 5
  row = value
  return (int(row),int(col))

###########################
###########################

##### EVALUATION ##########

def calculate_distances_among_agents(agent_probs):
    
    """
    Calculate the distances among agents' probability distributions.

    Parameters:
    agent_probs (np.array): The probability distributions of the agents.

    Returns:
    distance (np.array): The distances among agents' probability distributions.
    overlap (np.array): The overlap among agents' probability distributions.
    """

    distance = np.zeros((agent_probs.shape[0], agent_probs.shape[0]))
    overlap = np.zeros_like(distance)
    for i in range(agent_probs.shape[0]):
        for j in range(agent_probs.shape[0]):
            distance[i, j] = np.sum(np.abs(agent_probs[i] - agent_probs[j]))
            overlap[i,j] = bhattacharyya(agent_probs[i],agent_probs[j])
    return distance,overlap

def hellinger_distance(P, Q):
    """
    Calculate the Hellinger distance between two discrete probability distributions.
    
    Parameters:
    P (list or np.array): The first probability distribution.
    Q (list or np.array): The second probability distribution.
    
    Returns:
    float: The Hellinger distance.
    """
    P = np.asarray(P)
    Q = np.asarray(Q)
    
    # Ensure that the distributions are valid probability distributions
    assert np.all(P >= 0), "All elements in P must be non-negative"
    assert np.all(Q >= 0), "All elements in Q must be non-negative"
    assert np.isclose(np.sum(P), 1), "Elements in P must sum to 1"
    assert np.isclose(np.sum(Q), 1), "Elements in Q must sum to 1"
    
    # Compute the Hellinger distance
    distance = np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q))**2)) / np.sqrt(2)
    
    return distance
 
def bhattacharyya(P, Q):
    """
    Calcola l'indice di Bhattacharyya tra due distribuzioni di probabilità.

    Parameters:
    P (array-like): Prima distribuzione di probabilità.
    Q (array-like): Seconda distribuzione di probabilità.

    Returns:
    float: Indice di Bhattacharyya tra P e Q.
    """
    # Convertire in array NumPy
    P = np.asarray(P)
    Q = np.asarray(Q)

    # Assicurarsi che le distribuzioni siano normalizzate
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    # Calcolare l'indice di Bhattacharyya
    bc_index = np.sum(np.sqrt(P * Q))

    return bc_index

def evaluate_policies(horizon,policy):

  """
  retrieve the action probabilities for each agent along the horizon

  Parameters:
  horizon: length of the trajectory
  policy: policy object

  Returns:
  empirical_actions: action probabilities for each agent along the horizon
  """
  st = np.zeros(policy.obs_dim)
  empirical_actions = np.zeros((horizon,policy.agents,policy.action_dim,policy.obs_dim))
  agents = policy.agents

  for i in range(horizon):
      for j in range(policy.obs_dim):
        st[j] = 1
        st = np.repeat(np.expand_dims(st,axis=0),agents,axis=0)
        empirical_actions[i,:,:,j] = policy.get_action_probabilities(st,i)
        st = np.zeros(policy.obs_dim)
  return empirical_actions

def calculate_bernoulli_entropy(probs,no_probs):
  entropy = np.zeros((len(probs)))
  i = 0
  for p,no_p in zip(probs,no_probs):
      if p >0:
        entropy[i] -= p*math.log2(p)
      if no_p>0:
        entropy[i] -= no_p*math.log2(no_p)
      i +=1

  return entropy

def calculate_traj_entropy(initial_prob):
    initial_prob = np.array(initial_prob)  # Convert to numpy array if needed
    #sum over axis 0 and divide by the number of agents
    initial_prob = initial_prob[initial_prob > 0]  # Rimuovi valori di probabilità zero
    entropy = -np.sum(initial_prob * np.log(initial_prob + 1e-8))
    return entropy

def calculate_objective(K,m,agent_probs):
    
    """
    Calculate the objective function J(theta) of the maximum entropy exploration problem
    over multiple parallel agents based on the empirical agents distributions.

    Parameter:
    K (int): The number of trajectories.
    m (int): The number of agents.
    agent_probs (np.array): The probability distributions of the agents.
    
    Returns:
    float: The objective function of the agents.
    """
    
    J = 0
    for k in range(K):
      prod = 1
      for i in range(m):
        prod *= (1 - agent_probs[i,k])
      J += prod
    return J


def montecarlo_entropy(cum_entropy,iterations):
  "Calculate montecarlo version of the objective function"
  
  return np.sum(cum_entropy,axis=0)/iterations


def real_entropy_performance(agents, trajs, policy, transitions):

  # Pre-allocate for agent probabilities and actions
  p_t = np.ones((agents, len(trajs)))  # Probabilities for each agent on each trajectory
  cum_entropy = []
  action_combinations = np.array(list(itertools.product([0,1], repeat=trajs.shape[1]-1)))

  for one,t_1 in enumerate(trajs):
    for a_one,a_1 in enumerate(action_combinations):
      for two,t_2 in enumerate(trajs):
        for a_two,a_2 in enumerate(action_combinations):
          p_l = np.ones(agents)
          complete_t = np.array([t_1,t_2])
          counts = np.zeros((agents, trajs.shape[0]))
          #probs = np.zeros((agents, trajs.shape[0]))
          for i,(s1, s2) in enumerate(zip(t_1,t_2)):
            complete_s = np.array([s1,s2])
            if i< len(t_1)-1:
              complete_a = np.array([a_1[i],a_2[i]])
              st = np.zeros(agents,policy.obs_dim)
              st = one_hot_encoding(complete_s,policy.obs_dim)
              f = policy.get_action_probabilities(st, i)
              for a in range(agents):
                p_l[a] *= transitions[complete_s[a]][complete_t[a,i+1]][complete_a[a]] * f[a,complete_a[a]]
          # multiply p_l along the agents
          p_l = np.prod(p_l,axis=0)
          played_traj = [np.argwhere(np.all(complete_t[agent] == trajs, axis=1)).item() for agent in range(agents)]
          counts[np.arange(len(played_traj)), played_traj] += 1
          probs = counts.sum(axis=0) 
          probs = probs / probs.sum()
          cum_entropy.append(p_l * calculate_traj_entropy(probs))

  return np.sum(cum_entropy,axis=0)/len(cum_entropy)

def calculate_entropy_performance(agents, trajs, policy, transitions):
    # Pre-allocate for agent probabilities and actions
  action_combinations = np.array(list(itertools.product([0,1], repeat=trajs.shape[1]-1)))
  matrix_state_action = np.zeros((len(trajs)*len(action_combinations),2,trajs.shape[1]-1))
  matrix_state_action[:,0] = np.repeat(trajs[:,1:],len(action_combinations),axis=0)
  matrix_state_action[:,1] = np.tile(action_combinations,(len(trajs),1))
  entropy = 0

  results = Parallel(n_jobs=-1)(
    delayed(inner_loop_performance)(agents,policy,matrix_state_action,t,transitions) for t in matrix_state_action
  )

  results = np.array(results).flatten()
  #results =results[np.where(results != 0)]
  entropy = np.sum(results,axis=0)
  return entropy

def inner_loop_performance(agents,policy,matrix_state_action,t,transitions):
  cum_entropy = []
  t_1 = t[0]
  a_1 = t[1]
  for t_j in matrix_state_action:
    t_2 = t_j[0]
    a_2 = t_j[1]
    p_l = np.ones(agents)
    complete_t = np.array([t_1,t_2],dtype=int)
    #counts = np.zeros((agents, trajs.shape[0]))
    #probs = np.zeros((agents, trajs.shape[0]))
    for i,(s1, s2) in enumerate(zip(t_1,t_2)):
      if i == 0:
        complete_s = np.array([0,0],dtype=int)
      else:
        complete_s = np.array([t_1[i-1],t_2[i-1]],dtype=int)
      if i< len(t_1):
        complete_a = np.array([a_1[i],a_2[i]],dtype=int)
        st = np.zeros(agents,policy.obs_dim)
        st = one_hot_encoding(complete_s,policy.obs_dim)
        f = policy.get_action_probabilities(st, i)
        for a in range(agents):
          p_l[a] *= transitions[complete_s[a]][complete_t[a,i]][complete_a[a]] * f[a,complete_a[a]]
    # multiply p_l along the agents
    p_l = np.prod(p_l,axis=0)
    if np.all(t_1 == t_2):
      cum_entropy.append(0)
    else:
      cum_entropy.append(p_l*(-np.log(1/agents)))
  return cum_entropy


###########################
###########################

def calculate_exact(agents, trajs, policy, transitions):

  # Pre-allocate for agent probabilities and actions
  p_t = np.ones((agents, len(trajs)))  # Probabilities for each agent on each trajectory
  grads = np.zeros_like(policy.theta)
  new_score = np.zeros_like(policy.theta)
  action_combinations = np.array(list(itertools.product([0,1], repeat=trajs.shape[1]-1)))
  test_probs = np.load("test/action_sample_900000.npy")
  record_probs = np.zeros_like(test_probs)

  for agent in range(agents):
    for t_ind,t in enumerate(trajs):
      p_l = np.ones(len(action_combinations))
      p_l_test = np.ones(len(action_combinations))
      scores = np.zeros_like(policy.theta[0])
      scores = np.repeat(np.expand_dims(scores,axis=0),len(action_combinations), axis=0)
      cum_scores = np.zeros_like(policy.theta)
      cum_scores_test = np.zeros_like(policy.theta)
      for a,act_traj in enumerate(action_combinations):
        for i,s in enumerate(t):
          if i< len(t)-1:
            st = np.zeros(policy.obs_dim)
            st[s] = 1
            st = np.repeat(st[None, :], agents, axis=0)
            tmp_a = np.repeat(act_traj[None, i], agents, axis=0)
            scores[a,i] = policy.log_softmax_derivative(st,tmp_a, i)[agent]
            f = policy.get_action_probabilities(st, i)[agent]
            #p_l[a] *= transitions[s][t[i+1]][act_traj[i]] * f[act_traj[i]]
            p_l_test[a] *= transitions[s][t[i+1]][act_traj[i]] * f[act_traj[i]]
        p_l[a] = test_probs[t_ind,a]
        record_probs[t_ind] = p_l_test
      print("")
      p_t[agent,t_ind] = np.sum(p_l)
      p_l = p_l / p_t[agent,t_ind]
      cum_scores[agent] = np.sum(p_l[:,None,None,None]*scores,axis=0)
      cum_scores_test[agent] = np.sum(p_l_test[:,None,None,None]*scores,axis=0)
      #p_t[agent,t_ind] = np.sum(p_l)
      new_score[agent] += cum_scores_test[agent] /agents
      grads[agent] += (cum_scores[agent]) / agents

  return p_t,grads

def calculate_distribution_gradient_exact_p(agents, trajs, policy, transitions):

  # Pre-allocate for agent probabilities and actions
  p_t = np.ones((agents, len(trajs)))  # Probabilities for each agent on each trajectory
  grads = np.zeros_like(policy.theta)
  action_combinations = np.array(list(itertools.product([0,1], repeat=trajs.shape[1]-1)))

  test_probs = np.load("test/action_sample_900000.npy")

  for one,t_1 in enumerate(trajs):
    for a_one,a_1 in enumerate(action_combinations):
      for two,t_2 in enumerate(trajs):
        for a_two,a_2 in enumerate(action_combinations):
          for three,t_3 in enumerate(trajs):
            for a_three,a_3 in enumerate(action_combinations):
              scores = np.zeros_like(policy.theta)
              p_l = np.ones(agents)
              complete_t = np.array([t_1,t_2,t_3])
              counts = np.zeros((agents, trajs.shape[0]))
              #probs = np.zeros((agents, trajs.shape[0]))
              for i,(s1, s2, s3) in enumerate(zip(t_1,t_2,t_3)):
                complete_s = np.array([s1,s2,s3])
                if i< len(t_1)-1:
                  complete_a = np.array([a_1[i],a_2[i],a_3[i]])
                  st = np.zeros(agents,policy.obs_dim)
                  st = one_hot_encoding(complete_s,policy.obs_dim)
                  scores[:,i] = policy.log_softmax_derivative(st,complete_a, i)
                  f = policy.get_action_probabilities(st, i)
                  for a in range(agents):
                    p_l[a] *= transitions[complete_s[a]][complete_t[a,i+1]][complete_a[a]] * f[a,complete_a[a]]
              # multiply p_l along the agents
              p_l = np.prod(p_l,axis=0)
              scores *= p_l
              played_traj = [np.argwhere(np.all(complete_t[agent] == trajs, axis=1)).item() for agent in range(agents)]
              counts[np.arange(len(played_traj)), played_traj] += 1
              probs = counts.sum(axis=0) 
              probs = probs / probs.sum()
              entropy = calculate_traj_entropy(probs)
              grads +=  entropy * scores

  return p_t,grads

def calculate_distribution_gradient_exact_0(agents, trajs, policy, transitions):

  # Pre-allocate for agent probabilities and actions
  p_t = np.ones((agents, len(trajs)))  # Probabilities for each agent on each trajectory
  grads = np.zeros_like(policy.theta)
  action_combinations = np.array(list(itertools.product([0,1], repeat=trajs.shape[1]-1)))

  test_probs = np.load("test/action_sample_900000.npy")

  for one,t_1 in enumerate(trajs):
    for a_one,a_1 in enumerate(action_combinations):
      for two,t_2 in enumerate(trajs):
        for a_two,a_2 in enumerate(action_combinations):
          scores = np.zeros_like(policy.theta)
          p_l = np.ones(agents)
          complete_t = np.array([t_1,t_2])
          counts = np.zeros((agents, trajs.shape[0]))
          #probs = np.zeros((agents, trajs.shape[0]))
          for i,(s1, s2) in enumerate(zip(t_1,t_2)):
            complete_s = np.array([s1,s2])
            if i< len(t_1)-1:
              complete_a = np.array([a_1[i],a_2[i]])
              st = np.zeros(agents,policy.obs_dim)
              st = one_hot_encoding(complete_s,policy.obs_dim)
              scores[:,i] = policy.log_softmax_derivative(st,complete_a, i)
              f = policy.get_action_probabilities(st, i)
              for a in range(agents):
                p_l[a] *= transitions[complete_s[a]][complete_t[a,i+1]][complete_a[a]] * f[a,complete_a[a]]
          # multiply p_l along the agents
          p_l = np.prod(p_l,axis=0)
          scores *= p_l
          played_traj = [np.argwhere(np.all(complete_t[agent] == trajs, axis=1)).item() for agent in range(agents)]
          counts[np.arange(len(played_traj)), played_traj] += 1
          probs = counts.sum(axis=0) 
          probs = probs / probs.sum()
          entropy = calculate_traj_entropy(probs)
          grads +=  entropy * scores

  return p_t,grads

def calculate_distribution_gradient_exact(agents, trajs, policy, transitions):

  # Pre-allocate for agent probabilities and actions
  p_t = np.ones((agents, len(trajs)))  # Probabilities for each agent on each trajectory
  grads = np.zeros_like(policy.theta)
  action_combinations = np.array(list(itertools.product([0,1], repeat=trajs.shape[1]-1)))
  matrix_state_action = np.zeros((len(trajs)*len(action_combinations),2,trajs.shape[1]-1))
  matrix_state_action[:,0] = np.repeat(trajs[:,1:],len(action_combinations),axis=0)
  matrix_state_action[:,1] = np.tile(action_combinations,(len(trajs),1))
  objective = 0

  results = Parallel(n_jobs=-1)(
    delayed(inner_loop)(agents,policy,matrix_state_action,t,transitions) for t in matrix_state_action
  )

  for result in results:
    grads += result[0]
    objective += np.sum(result[1])
  return grads, objective

def inner_loop(agents,policy,matrix_state_action,t,transitions):
  local_objective_arr = []
  local_grads = np.zeros_like(policy.theta)
  t_1 = t[0]
  a_1 = t[1]
  for t_j in matrix_state_action:
    scores = np.zeros_like(policy.theta)
    t_2 = t_j[0]
    a_2 = t_j[1]
    p_l = np.ones(agents)
    complete_t = np.array([t_1,t_2],dtype=int)
    for i,(s1, s2) in enumerate(zip(t_1,t_2)):
      if i == 0:
        complete_s = np.array([0,0],dtype=int)
      else:
        complete_s = np.array([t_1[i-1],t_2[i-1]],dtype=int)
      if i< len(t_1):
        complete_a = np.array([a_1[i],a_2[i]],dtype=int)
        st = np.zeros(agents,policy.obs_dim)
        st = one_hot_encoding(complete_s,policy.obs_dim)
        scores[:,i] = policy.log_softmax_derivative(st,complete_a, i)
        f = policy.get_action_probabilities(st, i)
        for a in range(agents):
          p_l[a] *= transitions[complete_s[a]][complete_t[a,i]][complete_a[a]] * f[a,complete_a[a]]
    p_l = np.prod(p_l,axis=0)
    if np.all(t_1 == t_2):
      local_objective = 0
    else:
      local_objective = p_l*(-np.log(1/agents))
    local_objective_arr.append(local_objective)
    local_grads +=  local_objective * scores
  return np.array(local_grads),np.array(local_objective_arr)

def inner_loop_old(trajs,action_combinations,agents,policy,t_1,a_1,transitions):
  local_grads = np.zeros_like(policy.theta)
  local_objective_arr = []
  for two,t_2 in enumerate(trajs):
    for a_two,a_2 in enumerate(action_combinations):
      scores = np.zeros_like(policy.theta)
      p_l = np.ones(agents)
      local_objective = 0
      complete_t = np.array([t_1,t_2])
      for i,(s1, s2) in enumerate(zip(t_1,t_2)):
        complete_s = np.array([s1,s2])
        if i< len(t_1)-1:
          complete_a = np.array([a_1[i],a_2[i]])
          st = np.zeros(agents,policy.obs_dim)
          st = one_hot_encoding(complete_s,policy.obs_dim)
          scores[:,i] = policy.log_softmax_derivative(st,complete_a, i)
          f = policy.get_action_probabilities(st, i)
          for a in range(agents):
            p_l[a] *= transitions[complete_s[a]][complete_t[a,i+1]][complete_a[a]] * f[a,complete_a[a]]
      # multiply p_l along the agents
      p_l = np.prod(p_l,axis=0)
      if np.all(t_1 == t_2):
        local_objective = 0
      else:
        local_objective = p_l*(-np.log(1/agents))
      local_objective_arr.append(local_objective)
      local_grads +=  local_objective * scores
  return np.array(local_grads),np.array(local_objective_arr)

def calculate_exact2(agents, trajs, policy, transitions):
  # Pre-allocate for agent probabilities and actions
  p_t = np.ones((agents, len(trajs)))  # Probabilities for each agent on each trajectory
  # Loop over agents and trajectories
  for agent in range(agents):
      for index, t in enumerate(trajs):
          for i, s in enumerate(t[:-1]):  # Skip the last element (no next state)
              next_state = t[i + 1]
              # One-hot encoding for the current state
              st = np.zeros(policy.obs_dim)
              st[s] = 1
              # Get action probabilities for the current state (for all agents)
              f = policy.get_action_probabilities(np.repeat(st[None, :], agents, axis=0), i)[agent]
              # Update probability for this agent and trajectory
              p_t[agent, index] *= (transitions[s][next_state][0] * f[0]) + (transitions[s][next_state][1] * f[1])

  return p_t