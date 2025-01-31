import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.signal import savgol_filter
import datetime
import seaborn as sns  
import time 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from collections import Counter
import matplotlib.animation as animation


def plot_empirical_vs_target(history_probs, mdp_probs, max_probs, datetime_str,agents):
    
    """
    plot probability history respect the target and a optimal solution
    
    """
    for p in range(len(mdp_probs[0])):
        #create a subplot of two plot for both agents, to compare the probabilities
        fig, axs = plt.subplots(agents, 1, figsize=(10, 8))  # Adjust figsize as needed

        for a in range(agents):
            # Plot on the first subplot
            axs[a].plot(history_probs[:,a,p], label=f'S{p}')
            axs[a].axhline(y=mdp_probs[a][p], color='r', linestyle='-', label=f'S{p} Target')
            axs[a].axhline(y=max_probs[a][p], color='g', linestyle='-', label=f'S{p} Optimal')
            axs[a].grid()
            axs[a].set_ylabel("Probs")
            axs[a].set_xlabel("Time Step")
            axs[a].legend()

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        fig.savefig(f"test/{datetime_str}_{p}_combined_probs.png")
    return

def plot_abs_error(error_1,optimal_norm,datetime_str,agent_probs):

    """
    plot norm_1 error for each agent

    """    
    error_1 = np.array(error_1)
    fig, axs = plt.subplots(len(agent_probs),1, figsize=(10, 8)) 

    for i in range(len(agent_probs)):
        axs[i].plot(error_1)#moving_average(error_1[:,i],10))
        axs[i].axhline(y=optimal_norm[i], color='r',linestyle='--', label=f'Optimal Norm')
        axs[i].grid()
        axs[i].set_ylabel("Total Absolute Error")
        axs[i].set_xlabel("Time Step")
        axs[i].legend([f"Agent {i}"])
   
    fig.savefig(f"test/{datetime_str}_norm1.png")

    return

def plot_policy(K,action_size,horizon,empirical_actions,ref_policy,agents,datetime_str):
    """
    plot action probabilities along training for agent 0
    """   
    
    for i in range(agents):
        #fig = plt.figure(figsize=(10, 6))

        fig, axs = plt.subplots(nrows=action_size, ncols=horizon, figsize=(15, 10))  # Adjust nrows and ncols based on the number of theta values
        fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots

        for l in range(action_size):
            theta_values = np.array(empirical_actions)[:,:,i,l].reshape((K,horizon))
            for j in range(horizon):
                # Determine the row of the subplot go below every 3 columns for 4 rows
                tmp_theta = theta_values[:,j] #moving_average(theta_values[:,j],10)
                row =  l + j // horizon #(i*2) + l + j // horizon
                col = j % horizon   # Determine the column of the subplot
                ax = axs[row, col]  # Get the specific subplot axis
                ax.plot(tmp_theta)
                tmp_ref_policy = ref_policy[i,j] if l == 0 else 1 - ref_policy[i,j]
                ax.axhline(y=tmp_ref_policy, color='r', linestyle='-', label=f'Target')
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Value")
                ax.grid(True)

            plt.suptitle("Actions Across Epochs and Agents")
            #plt.show()

            fig.savefig(f"test/{datetime_str}_action_values_{i}.png")
    return

def plot_ns_policy(K,horizon,empirical_actions,agents,obs,datetime_str):
    """
    plot action probabilities along training for agent 0
    """   
    fig, axs = plt.subplots(nrows=agents, ncols=obs, figsize=(15, 10))  # Adjust nrows and ncols based on the number of theta values  
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots
    for l in range(horizon):

        for i in range(agents):
            #round i to the nearest integer
            theta_values = np.array(empirical_actions)[:,l,i,0,:].reshape((K,obs))
            for j in range(obs):
                # Determine the row of the subplot go below every 3 columns for 4 rows
                tmp_theta = theta_values[:,j]
                ax = axs[i, j]  # Get the specific subplot axis
                ax.plot(tmp_theta,label=f"horizon {l + 1}")
                ax.grid(True)
                ax.legend()
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Value")

    plt.suptitle(f"Actions probabilities Across Epochs and Agents horizon")
    #plt.show()

    plt.grid()
    plt.ylabel("Actions Value")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Actions value Across Epochs")

    fig.savefig(f"test/{datetime_str}_actions_values.png")
    return

def plot_updates(K,agents, horizon, results, datetime_str):
    """
    plot of theta updates, during training steps
    """
    
    fig = plt.figure(figsize=(10, 6))

    fig, axs = plt.subplots(nrows=2*agents, ncols=horizon, figsize=(15, 10))  # Adjust nrows and ncols based on the number of theta values
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots

    for i in range(agents):
        theta_values = np.array(results)[:,i,0,:,:].reshape((K, 2*horizon))
        for j in range(2*horizon):
            # Determine the row of the subplot go below every 3 columns for 4 rows
            tmp_theta = theta_values[:,j] #moving_average(theta_values[:,j],10)
            row = (i*2) + j // horizon
            col = j % horizon   # Determine the column of the subplot
            ax = axs[row, col]  # Get the specific subplot axis
            ax.plot(tmp_theta)
            ax.set_title(f"Theta {j+1}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.grid(True)

    plt.suptitle("Theta Values Across Epochs and Agents")
    #plt.show()

    plt.grid()
    plt.ylabel("Theta Value")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Theta Values Across Epochs")

    fig.savefig(f"test/{datetime_str}_theta_values.png")
    return

def plot_error(error, datetime_str):

    """
    plot error probability for each trajectory

    """
    fig = plt.figure()
    plt.plot(error)#moving_average(error,10))
    plt.grid()
    plt.ylabel("Absolute Error")
    plt.xlabel("Time Step")
    fig.savefig(f"test/{datetime_str}_abs_error.png")
    return

def plot_bar_diff(agent_probs, mdp_probs, datetime_str):

    """
    bar plot of the difference of single probability for each agent
    """

    fig = plt.figure(figsize=(10, 6))
    for i in range(len(agent_probs)):
        plt.bar(np.arange(len(mdp_probs[0])),agent_probs[i]-mdp_probs[i])
    
    plt.ylabel("Probs Error")
    plt.xlabel("Time Step")
    plt.legend([f"Agent {i}" for i in range(len(agent_probs))])
    fig.savefig(f"test/{datetime_str}_bar_error.png")
    
    return

def plot_distances_agents(agents,distances, datetime_str):
    
        """
        plot of the distances between the agent and the target
        """
        fig, axs = plt.subplots(nrows=agents, ncols=agents, figsize=(15, 10))
        distances = np.array(distances)
        for i in range(agents):
            for j in range(agents):
                ax = axs[i,j]  # Get the specific subplot axis
                ax.plot(distances[:,i,j])#moving_average(r[:,i],10))
                ax.set_title(f"Cumulative difference reward on Agent {i}")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Distance")
                ax.grid(True)

        fig.savefig(f"test/{datetime_str}_distances.png")
        return

def plot_update_ns(K,agents, obs,horizon, results, datetime_str):
    """
    plot of theta updates, during training steps
    """
    
    for l in range(horizon):
        fig, axs = plt.subplots(nrows=agents, ncols=obs, figsize=(15, 10))  # Adjust nrows and ncols based on the number of theta values  
        fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots
        for i in range(agents):
            #round i to the nearest integer
            theta_values = np.array(results)[:,i,l,0,:].reshape((K,obs))
            for j in range(obs):
                # Determine the row of the subplot go below every 3 columns for 4 rows
                tmp_theta = theta_values[:,j] #moving_average(theta_values[:,j],10)
                ax = axs[i, j]  # Get the specific subplot axis
                ax.plot(tmp_theta)
                ax.grid(True)
                ax.set_title(f"Theta {j+1}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Value")

        plt.suptitle(f"Theta Values Across Epochs and Agents {l} horizon")
        #plt.show()

        plt.grid()
        plt.ylabel("Theta Value")
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("Theta Values Across Epochs")

        fig.savefig(f"test/{datetime_str}_theta_values_{l}.png")
    return

def plot_perfomance(K,m,agents,cum_rewards, datetime_str):

    """
    plot of the cummulative rewards for each agent
    """
    r = np.array(cum_rewards).reshape((K,agents))
    fig = plt.figure(figsize=(10, 6))

    fig, axs = plt.subplots(nrows=1, ncols=agents, figsize=(15, 10))  # Adjust nrows and ncols based on the number of theta values
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots
    #subplot of the cummulative rewards for each agent
    for i in range(agents):
        ax = axs[i]  # Get the specific subplot axis
        ax.plot(r[:,i])#moving_average(r[:,i],10))
        ax.set_title(f"Cumulative difference reward on Agent {i}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)

    plt.suptitle("Cummulative Rewards Across Epochs 500 Trajs")
    fig.savefig(f"test/{datetime_str}_cum_rewards.png")

    return

# Define a moving average function
def moving_average(data, window_size):

    window_size = 1
    poly_order = 1
    data = np.array(data)
    if data.ndim == 1:  # If data is a 1D array
        return savgol_filter(data, window_size, poly_order)
    elif data.ndim == 2:  # If data is a 2D array

        return np.array([savgol_filter(data[:,i], window_size, poly_order) for i in range(data.shape[1])]).T
    else:
        raise ValueError("Data should be 1D or 2D numpy array")

def moving_interquartile_mean(data, window_size):
    window_size = 1
    data = np.array(data)
    iqm_values = []

    if data.ndim == 1:  # If data is a 1D array
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            window_data = data[start:end]
            Q1 = np.percentile(window_data, 25)
            Q3 = np.percentile(window_data, 75)
            iq_data = window_data[(window_data >= Q1) & (window_data <= Q3)]
            iqm_values.append(np.mean(iq_data))
    elif data.ndim == 2:  # If data is a 2D array
        for i in range(data.shape[1]):
            for j in range(data.shape[0]):
                start = max(0, j - window_size // 2)
                end = min(data.shape[0], j + window_size // 2 + 1)
                window_data = data[start:end, i]
                Q1 = np.percentile(window_data, 25)
                Q3 = np.percentile(window_data, 75)
                iq_data = window_data[(window_data >= Q1) & (window_data <= Q3)]
                iqm_values.append(np.mean(iq_data))
        iqm_values = np.array(iqm_values).reshape(data.shape)

    return np.array(iqm_values)

def barplot_references(references,agents,initialization,name_file):
    figure,ax = plt.subplots(nrows=agents,ncols=2,figsize=(10, 10))
    # tight_layout() adjusts subplots to fit into figure area.
    figure.tight_layout()
    figure.subplots_adjust(hspace=0.5)
    for i in range(agents):
        #add random color
        if agents>1:
            color = np.random.rand(3,)
            #set the same y axis for all the subplots
            ax[i,0].set_title(f'Agent {i} Init')
            ax[i,0].set_xlabel('Trajectories')
            ax[i,0].set_ylabel('Probabilities')
            ax[i,0].set_ylim([0,1.0])
            ax[i,0].set_xticks(np.arange(len(initialization[0]))) 
            ax[i,0].bar(np.arange(len(initialization[i])),initialization[i],color=color)  
            ax[i,0].grid(True)  
            # add outside xlabel
            #set the same y axis for all the subplots
            ax[i,1].set_title(f'Agent {i} last distr' )
            ax[i,1].set_xlabel('Trajectories')
            ax[i,1].set_ylabel('Probabilities')
            ax[i,1].set_ylim([0,1.0])
            ax[i,1].set_xticks(np.arange(len(references[0]))) 
            ax[i,1].bar(np.arange(len(references[i])),references[i],color=color)
            ax[i,1].grid(True)      
            # add outside xlabel
        else:
            color = np.random.rand(3,)
            #set the same y axis for all the subplots
            ax[0].set_title(f'Agent {i} Init')
            ax[0].set_xlabel('Trajectories')
            ax[0].set_ylabel('Probabilities')
            ax[0].set_ylim([0,1.0])
            ax[0].set_xticks(np.arange(len(initialization))) 
            ax[0].bar(np.arange(len(initialization)),initialization,color=color)  
            ax[0].grid(True)  
            # add outside xlabel
            #set the same y axis for all the subplots
            ax[1].set_title(f'Agent {i} last distr' )
            ax[1].set_xlabel('Trajectories')
            ax[1].set_ylabel('Probabilities')
            ax[1].set_ylim([0,1.0])
            ax[1].set_xticks(np.arange(len(references))) 
            ax[1].bar(np.arange(len(references)),references,color=color)
            ax[1].grid(True)      
            # add outside xlabel

    figure.savefig("test/"+name_file+".png")
    
    #plt.show()

def plot_entropy(entropy,datetime_str):
    """
    plot of the entropy for each trajectory
    """
    fig = plt.figure()
    plt.plot(entropy)#moving_average(entropy,10))
    plt.grid()
    plt.ylabel("Entropy")
    plt.xlabel("Time Step")
    fig.savefig(f"test/{datetime_str}_entropy.png")
    return

def plot_probs(K,m,x0,plot_name):

  
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection='3d')
  # Coordinates of the point where you want to plot the spike
  #x = np.linspace(0,1,K)
  #y = np.linspace(0,1,m)
  x = np.arange(K)
  y = np.arange(m)
  z = x0.reshape((m, K))
  for j in range(K):
    for k in range(m):
      ax.plot([x[j], x[j]], [y[k], y[k]], [0, z[k,j]], marker='_')


  # Setting labels for better understanding
  ax.set_xlabel('Trajectories')
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.set_ylabel('Parallel Agents')
  ax.yaxis.set_major_locator(MaxNLocator(integer=True))
  ax.set_zlabel('Probability')
  ax.zaxis.set_major_locator(MaxNLocator(integer=True))
  #ax.set_title(f'Distribution at episode {episodes}')
  #fig.savefig("moretraj_init.png")

  # Show plot
  #plt.show()
  fig.savefig(f"results/{plot_name}.png")

  plt.close(fig)
  return

def plot_agent_specific_distance(K, agents, cum_rewards, datetime_str):
    # Initialize an array to store distances for each agent pair
    agent_distances = np.zeros((K, int(agents * (agents - 1) / 2)))  # For n agents, n(n-1)/2 unique pairs
    cum_rewards = np.array(cum_rewards)

    ##### Agents distance selection
    # Use itertools to get unique pairs of agents
    pairs = list(itertools.combinations(range(agents), 2))  # Get all unique pairs

    # Extract the distances for each pair
    for idx, (i, j) in enumerate(pairs):
        agent_distances[:, idx] = cum_rewards[:, i, j]

    # Plotting the distances between the agent pairs
    figure_distance = plt.figure()
    plt.plot(agent_distances)
    plt.grid()
    plt.ylabel("Distance")
    plt.xlabel("Time Step")
    plt.title("Distance between agents")
    
    # Generate legend dynamically based on agent pairs
    legend_labels = [f"{i+1}-{j+1}" for i, j in pairs]
    plt.legend(legend_labels)

    # Save the figure
    figure_distance.savefig(f"test/{datetime_str}_agent_distances.png")

def plot_objective(objective, datetime_str,graph_name):
    fig = plt.figure()
    plt.plot(np.array(objective))
    plt.grid()
    plt.ylabel("Objective")
    plt.xlabel("Time Step")
    fig.savefig(f"test/{datetime_str}_{graph_name}.png")
    return

def plot_montecarlo_objective(objective,max_value, datetime_str,graph_name):
    fig = plt.figure()
    moving_average = []
    for i in range(0,len(objective),10):
        moving_average.append(np.mean(objective[i:i+10])/max_value)
    # convert to a percentege respect to the maximum
    plt.plot(np.array(moving_average))
    plt.grid()
    plt.ylabel("State Entropy")
    plt.xlabel("Time Step")
    plt.title("Objective as a percentage of the maximum")
    fig.savefig(f"test/{datetime_str}_{graph_name}.png")
    return

def plot_support(support, datetime_str,graph_name):
    fig = plt.figure()
    moving_average = []
    for i in range(0,len(support),10):
        moving_average.append(np.mean(support[i:i+10]))
    plt.plot(np.array(moving_average))
    plt.grid()
    plt.ylabel("Support")
    plt.xlabel("Time Step")
    fig.savefig(f"test/{datetime_str}_{graph_name}.png")

    return

def plot_mean_entropy_single_agent(mean_entropy,datetime_str,graph_name):
    fig = plt.figure()
    moving_average = []
    for i in range(0,len(mean_entropy),10):
        moving_average.append(np.mean(mean_entropy[i:i+10]))
    plt.plot(np.array(moving_average))
    plt.grid()
    plt.ylabel("Mean Entropy")
    plt.xlabel("Time Step")
    fig.savefig(f"test/{datetime_str}_{graph_name}.png")
    return

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def decode_trajectory_l(codified_trajectory, grid_size):
    """
    Decodes a trajectory from codified cell values to (row, col) indices.
    
    Parameters:
        codified_trajectory (list of int): List of codified cell values.
        grid_size (tuple): The size of the gridworld as (rows, cols).
    
    Returns:
        list of tuple: Decoded trajectory as (row, col) indices.
    """
    rows, cols = grid_size
    collection_trajectory = []
    for value in codified_trajectory:
        tmp_value = []
        for state in value:
            tmp_value.append((state // cols, state % cols))
        collection_trajectory.append(tmp_value)
    return collection_trajectory


def plot_gridworld_trajectory(grid_size, codified_trajectory, background_image_path=None):
    """
    Plots a gridworld showing the trajectory of visited states.
    
    Parameters:
        grid_size (tuple): The size of the gridworld as (rows, cols).
        codified_trajectory (list of int): The trajectory as a list of codified cell values.
        background_image_path (str): Path to the background image (optional).
    
    Returns:
        None
    """
    # Decode the codified trajectory to (row, col) indices
    trajectories = decode_trajectory(codified_trajectory, grid_size)

    plt.figure(figsize=(8, 6))

    # Plot the background image if provided
    if background_image_path:
        background = Image.open(background_image_path)
        background = background.resize((grid_size[1], grid_size[0]))  # Resize to (cols, rows)
        plt.imshow(background, extent=[0, grid_size[1], 0, grid_size[0]], origin="upper")
    
    # Plot grid lines
    for x in range(grid_size[1] + 1):
        plt.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    for y in range(grid_size[0] + 1):
        plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)
    
    # Plot trajectory
    for trajectory in trajectories:
        rows, cols = zip(*trajectory)
    #rows, cols = zip(*trajectory)  # Unpack rows and cols
        plt.plot([c + 0.5 for c in cols], [r + 0.5 for r in rows], marker='o', color='red', markersize=8, label="Trajectory")
        
        # Add arrows to indicate direction
        for i in range(len(trajectory) - 1):
            start = trajectory[i]
            end = trajectory[i + 1]
            plt.arrow(start[1] + 0.5, start[0] + 0.5, 
                    (end[1] - start[1]) * 0.8, (end[0] - start[0]) * 0.8,
                    head_width=0.3, head_length=0.3, fc='blue', ec='blue')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Formatting
    plt.title("Gridworld Trajectory")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.xlim(0, grid_size[1])
    plt.ylim(grid_size[0], 0)
    plt.xticks(range(grid_size[1]))
    plt.yticks(range(grid_size[0]))
    #plt.legend()
    plt.grid(False)
    plt.savefig(f"heatmap/{timestamp}-trajectory.png")
    return


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def decode_trajectory(codified_trajectory, grid_size):
    """
    Decodes a trajectory from codified cell values to (row, col) indices.
    
    Parameters:
        codified_trajectory (list of int): List of codified cell values.
        grid_size (tuple): The size of the gridworld as (rows, cols).
    
    Returns:
        list of tuple: Decoded trajectory as (row, col) indices.
    """
    rows, cols = grid_size
    return [(int(value // cols), int(value % cols) ) for value in codified_trajectory]

def decode_taxi_trajectory(codified_trajectory, grid_size):
    """
    Decodes a trajectory from codified cell values to (row, col) indices.
    
    Parameters:
        codified_trajectory (list of int): List of codified cell values.
        grid_size (tuple): The size of the gridworld as (rows, cols).
    
    Returns:
        list of tuple: Decoded trajectory as (row, col) indices.
    """
    rows, cols = grid_size
    value = []
    for trj in codified_trajectory:
        pass_position = trj % 5
        trj = trj // 5
        col = trj % 5
        trj = trj // 5
        row = trj 
        value.append([int(row),int(col)])
    return value


def plot_gridworld_heatmap_from_trajectory(grid_size, codified_trajectory, counts,background_image_path=None, cmap="viridis"):
    """
    Plots a heatmap showing the visit frequencies of states in a gridworld.
    
    Parameters:
        grid_size (tuple): The size of the gridworld as (rows, cols).
        codified_trajectory (list of int): The trajectory as a list of codified cell values.
        background_image_path (str): Path to the background image (optional).
        cmap (str): Colormap to use for the heatmap.
    
    Returns:
        None
    """
    # Decode the codified trajectory to (row, col) indices
    trajectory = decode_trajectory(codified_trajectory, grid_size)
    
    # Initialize visit count grid
    visit_counts = np.zeros(grid_size, dtype=int)
    
    # Count visits to each state
    for i, (row, col) in enumerate(trajectory):
        visit_counts[row, col] = counts[i]

    # Plot the background image if provided
    if background_image_path:
        background = Image.open(background_image_path)
        background = background.resize((grid_size[1], grid_size[0]))  # Resize to (cols, rows)
        plt.imshow(background, extent=[0, grid_size[1], 0, grid_size[0]], origin="upper", alpha=0.6)

    plt.figure(figsize=(8, 6))
    
    svm = sns.heatmap(visit_counts, annot=True, fmt="d", cmap=cmap, cbar=False, square=True)
    
    # Formatting
    plt.title("Heatmap of Visited States in Gridworld")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.xlim(0, grid_size[1])
    plt.ylim(grid_size[0], 0)
    plt.xticks(range(grid_size[1]))
    plt.yticks(range(grid_size[0]))
    plt.grid(False)
    #plt.show()
    timestamp = time.time_ns()
    figure = svm.get_figure()    
    figure.savefig(f'heatmap/{timestamp}-heatmap.png', dpi=400)
    plt.close()
   
    return


def plot_gridworld_heatmap(history_map,grid_size):
    history_map = np.array(history_map)
    fig, ax = plt.subplots(grid_size[0], grid_size[1], figsize=(8, 6))
    ylim = history_map.max()    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Plot the history of each cell
            raw_history = history_map[:, i, j]
            smoothed_history = np.convolve(raw_history, np.ones(50) / 50, mode='valid')
            # Apply moving average to smooth the data
            ax[i, j].plot(smoothed_history)
            ax[i, j].set_title(f"State {i},{j}")
            ax[i, j].set_xlabel("Time Step")
            ax[i, j].set_ylabel("Value")
            ax[i, j].grid(True)
            ax[i, j].set_ylim([0, ylim])
    
    plt.tight_layout()
    t = time.time_ns()
    fig.savefig(f"heatmap/{t}-heatmap.png")
    return

def plot_3d_surface(heatmap,grid_size,name = None):
    # Set up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(grid_size[1]), np.arange(grid_size[0])[::-1])
    # Create a surface plot
    history_heatmap = np.array(heatmap)
    #moving avarage of the heatmap
    moving_average = []
    for i in range(0,len(history_heatmap),50):
        moving_average.append(np.mean(history_heatmap[i:i+50],axis=0))

    #surface = [ax.plot_surface(X, Y, moving_average[0], cmap='viridis')]
    for j in range(grid_size[1]):
        for k in range(grid_size[0]):
            ax.plot([X[k, j], X[k, j]], [Y[k, j], Y[k, j]], [0, moving_average[0][k,j]], marker='_')

    # Setting labels for better understanding
    #ax.set_xlabel('Trajectories')
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.set_ylabel('Parallel Agents')
    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.set_zlabel('Probability')
    #ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    if name:
        ax.set_title(f"Step 0 {name}")
    else:
        ax.set_title("Step 0 All Agents")

    # Function to update the surface for each frame
    def update(frame):
        ax.clear()
        #ax.set_zlim(-2, 2)  # Keep z-axis limits consistent
        Z = moving_average[frame]
        for j in range(grid_size[1]):
            for k in range(grid_size[0]):
                ax.plot([X[k,j], X[k,j]], [Y[k,j], Y[k,j]], [0, Z[k,j]], marker='_')
                if name:
                    ax.set_title(f"Step {frame} {name}")
                else:
                    ax.set_title(f"Step {frame} All Agents")

    # Create an animation
    ani = FuncAnimation(fig, update, frames=len(moving_average), repeat=False)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if name:
        ani.save(f'heatmap/{timestamp}_{name}.gif', writer='imagemagick', fps=5)
    else:
        ani.save(f'heatmap/{timestamp}_surface.gif', writer='imagemagick', fps=5)
    return


def plot_heatmap_new(heatmap, grid_size, name=None):
    # Convert the heatmap data into a moving average
    history_heatmap = np.array(heatmap)
    moving_average = [
        np.mean(history_heatmap[i:i + 10], axis=0)
        for i in range(0, len(history_heatmap), 10)
    ]

    # Set up the figure
    fig, ax = plt.subplots()
    
    # Function to update the heatmap for each frame
    def update_heatmap(frame):
        ax.clear()
        sns.heatmap(
            moving_average[frame],
            ax=ax,
            cmap='viridis',
            annot=True,  # Add values to the cells
            fmt=".2f",   # Format for numbers in the cells
            cbar=False   # Remove the colorbar
        )
        if name:
            ax.set_title(f"Agent {name} Trajectories with Visit Frequency Heatmap frame {frame}")
        else:
            ax.set_title(f" Trajectories with Visit Frequency Heatmap frame {frame} All Agents")
        
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_ylabel('Rows', fontsize=12)
    # Save the animation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'heatmap/{timestamp}_{name if name else "surface"}.gif'

    def animate(frame):
        update_heatmap(frame)
        return ax

    ani = FuncAnimation(fig, animate, frames=len(moving_average), repeat=False)
    ani.save(filename, writer='Pillow', fps=15)

    plt.close(fig)
    return


def plot_trajectory_heatmap(agent,plain_trajs,codified_trajs,batch,mini_batch,horizon,name=None):
    # Count visits to each cell
    plain_trajs = np.array(plain_trajs)
    trajs = np.array(codified_trajs)

    # Function to compute the mean trajectory across mini-batches for each batch
    def compute_significative_trajectory(plain_trajs,codified_trajs):
        # Calculate the mean trajectory across mini-batches (axis 1)
        uni_trjs = []
        for i in range(plain_trajs.shape[0]):
            unique_trjs,_ = np.unique(plain_trajs[i], axis=-2, return_counts=True)
            most_visited = np.argmax(_)
            #retrive original most visited trajectory not reordered
            original = np.argwhere(np.all(plain_trajs[i] == unique_trjs[most_visited], axis=-1))
            uni_trjs.append(codified_trajs[i,original[0][0]])
        return uni_trjs

    # Compute the mean trajectory
    mean_trajs = compute_significative_trajectory(plain_trajs,trajs)

    trajs = trajs.reshape(batch*mini_batch,horizon+1,2)

    grid_size = (4, 4)
    rows, cols = grid_size

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Overlay trajectories with distinct colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(mean_trajs)))  # Use a colormap for distinct trajectory colors

    def update_frame_history(frame,horizon):
        start_index = frame*mini_batch
        end_index = (frame +1)*mini_batch
        tmp_traj = trajs[start_index:end_index].copy()
        visit_counter = Counter(tuple(cell) for traj in tmp_traj for cell in traj)
        visit_grid = np.zeros(grid_size)
        for (r, c), count in visit_counter.items():
            visit_grid[int(r), int(c)] = count/(mini_batch * horizon+1)
        ax.clear()

        # Plot the visit heatmap using seaborn's heatmap
        sns.heatmap(
            visit_grid,
            ax=ax,
            cmap='viridis',
            annot=True,  # Add values to the cells
            fmt=".2f",   # Format for numbers in the cells
            cbar=False,  # Remove the colorbar
            square=True, # Ensure cells are square
            xticklabels=False, yticklabels=False  # Hide labels
        )

        # Plot the trajectory
        for i, trajectory in enumerate(mean_trajs[frame:frame + 1]):
            y,x = zip(*trajectory)  # Extract x and y coordinates
            # Shift coordinates by 0.5 to center them in each grid cell
            x_centered = [xi + 0.5 for xi in x]
            y_centered = [yi + 0.5 for yi in y]
            
            #if i == frame:
            ax.plot(x_centered, y_centered, marker="o", linestyle="--", color='red', alpha=0.7, label=f'Trajectory {i + 1}', linewidth=4)
            #else:
            #    ax.plot(x_centered, y_centered, marker='o', color=colors[i], alpha=0.7, label=f'Trajectory {i + 1}', linewidth=2)

            # Add an arrow to indicate the direction at the last point
            if len(x_centered) > 1:  # Ensure there are at least 2 points for the arrow
                dx = x_centered[-1] - x_centered[-2]
                dy = y_centered[-1] - y_centered[-2]
                ax.arrow(x_centered[-2], y_centered[-2], dx, dy, head_width=0.1, head_length=0.1, fc=colors[i], ec=colors[i])
        
        # Customize grid
        ax.set_xticks(np.arange(0, cols, 1))
        ax.set_yticks(np.arange(0, rows, 1))
        ax.grid(color='gray', linestyle='-', linewidth=0.5)

        # Invert y-axis if needed (for top-left origin)
        # ax.invert_yaxis()

        # Labels and title
        ax.set_title(f'Agent {agent} Trajectories with Visit Frequency Heatmap frame {frame}', fontsize=14)
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_ylabel('Rows', fontsize=12)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'heatmap/{timestamp}_{agent}_trajectory.gif'

    def animate(frame,horizon):
        update_frame_history(frame,horizon)
        return ax

    ani = FuncAnimation(fig, animate, frames=len(mean_trajs)-1, repeat=False, fargs=(horizon,))
    ani.save(filename, writer='Pillow', fps=10)
    
    return