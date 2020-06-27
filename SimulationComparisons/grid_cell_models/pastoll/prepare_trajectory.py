# Prepares a trajectory data file for use in the pastoll 2013 model

import scipy.io
import numpy as np

pos = scipy.io.loadmat('../trajectory_data.mat')['pos']

t  = pos[2, :]
x = pos[0, :] - 50
y = pos[1, :] - 50

traj = {
        'pos_timeStamps': t.T,
        'pos_x': x.T,
        'pos_y': y.T,
        'dt': np.array(0.02)
        }


scipy.io.savemat('./model/grid_cell_model/kg_trajectory.mat', traj)
