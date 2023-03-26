import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pickle
import os


if __name__ == '__main__':
    
    #cwd = os.getcwd()
    dataset,tar_A_cells,tar_B_cells,obs_cells = pickle.load(open('./demos/ppo_demos_small_multitarget_v0.pk', 'rb'))



    traj = []
    traj_per_ep = []
    ep_idx = np.random.choice(100,1)[0] # each ep has a different env/traj
    print(ep_idx)
    for k in range(len(dataset[ep_idx]['states'])):
        y = np.where(dataset[ep_idx]['states'][k][1]==1)[0][0]
        y = -y+7
        x = np.where(dataset[ep_idx]['states'][k][1]==1)[1][0]
        traj.append([x,y])
        traj_per_ep.append(traj)


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
    x_coord = [x for [x,y] in traj_per_ep[0]]
    y_coord = [y for [x,y] in traj_per_ep[0]]
    plt.plot(x_coord[0]-0.4, y_coord[0],'ms')
    plt.plot(x_coord[1:], y_coord[1:],'bo')

    plt.show()

