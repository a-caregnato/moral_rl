from tqdm import tqdm
from ppo import *
import torch
from envs.gym_wrapper import *
import wandb
import argparse
import os
from matplotlib import pyplot as plt
import matplotlib.patches as patches

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    
    # Create Environment
    env_id = 'small_multitarget_v0'
    vec_env = GymWrapper(env_id)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    ppo.load_state_dict(torch.load('./policies/test_with_one_target.pt'))

    return_per_ep = []
    traj_per_ep = []
    tar_A_cells = []
    tar_B_cells = []
    timesteps_per_ep = []   
    obs_cells = []
    n_episodes = 1
    n_steps = 75
    for ep in range(n_episodes):
        return_ = 0
        traj = []

        # Save environment matrice
        tar_A_cells.append(states[2]) # not sure which is which for the targets
        tar_B_cells.append(states[3])
        obs_cells.append(states[4])

        for k in range(n_steps):

            # Dynamics
            actions, log_probs = ppo.act(states_tensor)
            next_states, rewards, done, info = vec_env.step(actions)

            if sum(rewards) > 0.5:
                stop = 1
            # Metrics
            return_ = return_ + np.sum(rewards)

            y = np.where(states[1]==1)[0][0]
            y = -y+7
            x = np.where(states[1]==1)[1][0]
            traj.append([x,y])

            # termination
            if done or k == n_steps-1:
                break

            # Prepare state input for next time step
            states = next_states.copy()
            states_tensor = torch.tensor(states).float().to(device)

        # Save episode data   
        return_per_ep.append(return_)
        traj_per_ep.append(traj)
        timesteps_per_ep.append(k)

    print('Mean return: ',np.mean(return_per_ep))
        
    ep_idx = 0 # each ep has a different env/traj
    print('Return of trajectory[',ep_idx,']: ',return_per_ep[ep_idx])
    print('Number of timesteps: ',timesteps_per_ep[ep_idx])
    #print(traj_per_ep[ep_idx])

    
    ####
    # Plots ------------------------------------------------------ 
    fig, ax = plt.subplots()
    plt.grid(axis='both', color='k',linewidth=2)
    plt.xticks(np.arange(0.5, 7.5, step=1))
    plt.yticks(np.arange(0.5, 7.5, step=1))
    plt.xlim((0.5,5.5))
    plt.ylim((0.5,6.5))
    # plot obstacles
    obs_coords = np.stack(np.where(obs_cells[ep_idx]==1),axis=0) 
    for i in range(len(obs_coords[0])):
        rect = patches.Rectangle((obs_coords[1,i]-0.5, -obs_coords[0,i]+7-0.5), 1, 1, linewidth=1, edgecolor='k', facecolor='k')
        ax.add_patch(rect)
       
    # plot targets
    tar_A_coords =np.stack(np.where(tar_A_cells[ep_idx]==1),axis=0)
    for i in range(len(tar_A_coords[0])):
           rect = patches.Rectangle((tar_A_coords[1,i]-0.5, -tar_A_coords[0,i]+7-0.5), 1, 1, linewidth=1, edgecolor='k', facecolor='g')
           ax.add_patch(rect)

    tar_B_coords =np.stack(np.where(tar_B_cells[ep_idx]==1),axis=0)
    for i in range(len(tar_B_coords[0])):
           rect = patches.Rectangle((tar_B_coords[1,i]-0.5, -tar_B_coords[0,i]+7-0.5), 1, 1, linewidth=1, edgecolor='k', facecolor='r')
           ax.add_patch(rect)

    # traj
    x_coord = [x for [x,y] in traj_per_ep[ep_idx]]
    y_coord = [y for [x,y] in traj_per_ep[ep_idx]]
    plt.plot(x_coord[0]-0.4, y_coord[0],'ms')
    plt.plot(x_coord[1:], y_coord[1:],'bo')

    plt.show()


