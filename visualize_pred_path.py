
'''
Script to visualize results of Capstone Project
'''

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import torch
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  
import torch
import numpy as np
import sys

## Specify data filepaths
testset_dir = '/Users/mithunjothiravi/Repos/argoverse-api/demo_usage/data/test_obs/data'
preds_dir = '/Users/mithunjothiravi/Repos/argoverse-api/demo_usage/data/test_preds.tar'

## Load the Argoverse dataset and the prediction information
testset_load = ArgoverseForecastingLoader(testset_dir)
pred_load = torch.load(preds_dir)[25024] # Change the index to be the file number

# Test set trajectories
test_trajectories = testset_load.agent_traj


ax = plt.gca()
plt.plot(pred_load[0,0,0],pred_load[0,0,1],'-o',c='r') #starting point here
plt.plot(pred_load[0,:21,0],pred_load[0,:21,1],'-',c='b')

for i in range(len(pred_load)):
    plt.plot(pred_load[i,20:,0],pred_load[i,20:,1],'-',c=np.random.rand(3,))

plt.xlabel('map_x_coord (m)')
plt.ylabel('map_y_coord (m)')
ax.set_aspect('equal')
plt.show()