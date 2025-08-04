## Remove trajectories of length 50 (unsuccessful tasks) from data (demo.py)

import sys
import os
import numpy as np
sys.path.append("/datasets/sxx/flexskill/reskill")

data = np.load("dataset/fetch_block_100000/demos.npy", allow_pickle=True)

print("Original data shape:", data.shape)

filtered_data = []
max_timestep = 0

for traj in data[:100000]:
    timesteps = len(traj["obs"])

    if timesteps > max_timestep:
        max_timestep = timesteps
    
    if timesteps != 50:
        filtered_data.append(traj)

print("Number of trajectories after filtering:", len(filtered_data))


np.save("dataset/fetch_block_100000/demos1.npy", np.array(filtered_data), allow_pickle=True)
print("Filtered data saved to demos1.npy")

data1 = np.load("dataset/fetch_block_100000/demos1.npy", allow_pickle=True)

print("Filtered data shape:", data1.shape)