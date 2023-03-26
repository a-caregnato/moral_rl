from tqdm import tqdm
from airl import *
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

    M_wall = states[0]
    M_obs = states[4]
    M_agt = np.zeros((8,7))
    M_agt[1,1] = 1
    M_tar_A = np.zeros((8,7))
    #M_tar_B = M_agt
    M_tar_B = np.zeros((8,7))
    M_tar_B[3,4] = 1
    
    manual_state = np.array([M_wall,M_agt,M_tar_A,M_tar_B,M_obs], dtype=np.float32)


   # states_tensor = torch.tensor(manual_state).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]
    
    discriminator = DiscriminatorMLP(state_shape=state_shape, in_channels=in_channels).to(device)
    discriminator.load_state_dict(torch.load('./saved_discriminators/teste.pt'))

    correct_in_state = []
    correct_in_state.append(manual_state)

    tensor_correct_input_state = torch.tensor(correct_in_state).float().to(device)

    reward = discriminator.g(tensor_correct_input_state)[0]
    print('Target A reward: ',reward)

    a=1